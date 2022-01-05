from semantic_val.decision.decide import (
    make_group_decision,
    prepare_las_for_decision,
    split_idx_by_dim,
    update_las_with_decisions,
)

from enum import Enum
import pickle
import glob
import os
import os.path as osp
from typing import List, Tuple
from omegaconf.base import SCMode
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import numpy as np
import optuna
import laspy
from sklearn.metrics import confusion_matrix
from semantic_val.utils.db_communication import ConnectionData
from semantic_val.datamodules.processing import ChannelNames
from semantic_val.decision.codes import (
    MTS_AUTO_DETECTED_CODE,
    MTS_FALSE_POSITIVE_CODE_LIST,
    FinalClassificationCodes,
    reset_classification,
)
from semantic_val.utils import utils
from semantic_val.decision.codes import (
    DECISION_CODES_LIST_FOR_CONFUSION,
    MTS_TRUE_POSITIVE_CODE_LIST,
)

log = utils.get_logger(__name__)

# Optimization is performed under those constrains:
MIN_PRECISION_CONSTRAINT = 0.98
MIN_RECALL_CONSTRAINT = 0.98
MIN_AUTOMATION_CONSTRAINT = 0.35
# Solution that meets them and maximizes the product of metrics is selected.


def optimize(config: DictConfig) -> Tuple[float]:
    """
    Parse predicted las, cluster groups of points, ottimize decision thresholds, then use them to product final las.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    Pseudocode (temporary):

    for las in las_list:
        prepare with pdal (cluster + overlay)
        save to PREPARED_las_file
        add the X and target informations to lists
    pickle.dump the lists

    pickle.load the lists
    Run an hyperoptimisation
    Find the best solution
    Pickle.dump best solution it where las are saved.

    pickle.load the lists
    pickle.load the best solution
    evaluate_decision

    load best solution
    for PREPARED_las_file in las_list, using prepared las:
        reset_classif
        update with thresholds of solution
        save to POST_IA_las_file
    """
    if "seed" in config:
        seed_everything(config.seed, workers=True)
    input_dir = config.optimize.predicted_las_dirpath
    output_dir = config.optimize.results_output_dir
    data_connexion_db = ConnectionData(
        config.prediction.host,
        config.prediction.user,
        config.prediction.pwd,
        config.prediction.bd_name,
    )

    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Best trial and outputs will be saved in {output_dir}")
    log.info(f"Logs will be saved in {os.getcwd()}")
    las_filepaths = glob.glob(osp.join(input_dir, "*.las"))
    # DEBUG
    las_filepaths = [
        "/var/data/cgaydon/data/202110_building_val/logs/good_checkpoints/V2.0/validation_preds/test_870000_6649000.las",
        # "/var/data/cgaydon/data/202110_building_val/logs/good_checkpoints/V2.0/validation_preds/test_792000_6272000.las"
    ]
    print(las_filepaths)

    ### CLUSTER AND GET PROBAS AND TARGETS FOR LATER OPTIMIZATION
    if "prepare" in config.optimize.todo:
        (
            group_probas,
            group_topo_overlay_bools,
            mts_gt,
        ) = get_group_info_and_label(las_filepaths, data_connexion_db, output_dir)
        probas_target_groups_filepath = osp.join(output_dir, "probas_target_groups.pkl")
        with open(probas_target_groups_filepath, "wb") as f:
            pickle.dump(
                (
                    group_probas,
                    group_topo_overlay_bools,
                    mts_gt,
                ),
                f,
            )

    ### OPTIMIZE THRESHOLDS WITH OPTUNA
    if "optimize" in config.optimize.todo:
        probas_target_groups_filepath = osp.join(output_dir, "probas_target_groups.pkl")
        with open(probas_target_groups_filepath, "rb") as f:
            (
                group_probas,
                group_topo_overlay_bools,
                mts_gt,
            ) = pickle.load(f)
        log.info(f"Optimizing on N={len(mts_gt)} groups of points.")

        sampler_kwargs = OmegaConf.to_container(
            config.optimize.sampler_kwargs, structured_config_mode=SCMode.DICT_CONFIG
        )
        sampler_kwargs.update({"constraints_func": constraints_func})
        sampler = eval(config.optimize.sampler_class)(**sampler_kwargs)
        study = optuna.create_study(
            sampler=sampler,
            directions=config.optimize.study.directions,
            study_name=config.optimize.study.study_name,
        )

        def objective(trial):
            """Objective function. Sets the group data for quick optimization."""
            return _objective(
                trial,
                group_probas,
                group_topo_overlay_bools,
                mts_gt,
            )

        study.optimize(objective, n_trials=config.optimize.study.n_trials)

        best_trial = select_best_trial(study)
        log.info("Best_trial: \n")
        log.info(best_trial)
        with open(config.optimize.best_trial_pickle_path, "wb") as f:
            pickle.dump(best_trial, f)
            log.info(f"Best trial stored in: {config.optimize.best_trial_pickle_path}")

    ### EVALUATE WITH BEST PARAMS
    if "evaluate" in config.optimize.todo:
        with open(config.optimize.best_trial_pickle_path, "rb") as f:
            best_trial = pickle.load(f)
            log.info(f"Using best trial from: {config.optimize.best_trial_pickle_path}")
        probas_target_groups_filepath = osp.join(output_dir, "probas_target_groups.pkl")
        with open(probas_target_groups_filepath, "rb") as f:
            (
                group_probas,
                group_topo_overlay_bools,
                mts_gt,
            ) = pickle.load(f)
        log.info(f"Evaluating best trial on N={len(mts_gt)} groups of points.")
        decisions = [
            make_group_decision(
                probas,
                topo_overlay_bools,
                **best_trial.params,
            )
            for probas, topo_overlay_bools in zip(
                group_probas, group_topo_overlay_bools
            )
        ]
        metrics_dict = evaluate_decisions(mts_gt, np.array(decisions))
        log.info(
            f"\n Metrics and Confusion Matrices: {get_results_logs_str(metrics_dict)}"
        )

    ### UPDATING LAS
    if "update" in config.optimize.todo:
        log.info(f"Validated las will be saved in {output_dir}")
        for las_filepath in tqdm(las_filepaths, desc="Update Las"):
            # we reuse post_ia path that already contains clustered las.
            basename = osp.basename(las_filepath)
            cluster_path = osp.join(output_dir, "PREPARED_" + basename)
            las = laspy.read(cluster_path)
            with open(config.optimize.best_trial_pickle_path, "rb") as f:
                best_trial = pickle.load(f)
                log.info(
                    f"Using best trial from: {config.optimize.best_trial_pickle_path}"
                )
            las.classification = reset_classification(las.classification)
            las = update_las_with_decisions(
                las, best_trial.params, use_final_classification_codes=False
            )
            out_path = osp.join(output_dir, "POST_IA_" + basename)
            las.write(out_path)
            log.info(f"Saved update las to {out_path}")


# Functions


def define_MTS_ground_truth_flag(frac_true_positive):
    """Helper : Based on the fraction of confirmed building points, set the nature of the shape or declare an ambiguous case"""
    FP_FRAC = 0.05
    TP_FRAC = 0.95
    if frac_true_positive >= TP_FRAC:
        return FinalClassificationCodes.BUILDING.value
    elif frac_true_positive < FP_FRAC:
        return FinalClassificationCodes.NOT_BUILDING.value
    else:
        return FinalClassificationCodes.UNSURE.value


def get_group_info_and_label(
    las_filepaths: List[str],
    data_connexion_db: ConnectionData,
    output_dir: str,
) -> Tuple[List[np.array], List[str]]:
    """
    From a folder of las with probabilities, cluster the clouds and append both list of predicted probas
    and MTS ground truth to lists.
    group_probas: the probas of each group of points
    mts_gt: the group label based on ground truths
    """
    group_probas = []
    group_topo_overlay_bools = []
    mts_gt = []
    for las_filepath in tqdm(las_filepaths, desc="Preparing  ->"):
        log.info(f"Preparing tile: {las_filepath}...")
        basename = osp.basename(las_filepath)
        out_path = osp.join(output_dir, "PREPARED_" + basename)
        structured_array = prepare_las_for_decision(
            las_filepath,
            data_connexion_db,
            out_path,
            candidate_building_points_classification_code=[MTS_AUTO_DETECTED_CODE]
            + MTS_TRUE_POSITIVE_CODE_LIST
            + MTS_FALSE_POSITIVE_CODE_LIST,
        )
        if len(structured_array) == 0:
            log.info("/!\ Skipping tile: there are no candidate building points.")
            continue
        split_idx = split_idx_by_dim(structured_array[ChannelNames.ClusterID.value])
        split_idx = split_idx[1:]  # remove default group with ClusterID = 0
        for pts_idx in tqdm(split_idx, desc="Append probas and targets  ->"):
            group_probas.append(
                structured_array[ChannelNames.BuildingsProba.value][pts_idx]
            )
            group_topo_overlay_bools.append(
                structured_array[ChannelNames.BDTopoOverlay.value][pts_idx]
            )
            # ChannelNames.Classification.value is "classification" not "Classification"
            frac_true_positive = np.mean(
                np.isin(
                    structured_array["Classification"][pts_idx],
                    MTS_TRUE_POSITIVE_CODE_LIST,
                )
            )
            mts_gt.append(define_MTS_ground_truth_flag(frac_true_positive))
    return (
        group_probas,
        group_topo_overlay_bools,
        np.array(mts_gt),
    )


class MetricsNames(Enum):
    # Shapes info
    MTS_GROUPS_COUNT = "P_MTS_GROUPS_COUNT"
    MTS_GROUP_BUILDING_LABEL = "P_MTS_BUILDINGS"
    MTS_GROUP_NO_BUILDINGS_LABEL = "P_MTS_NO_BUILDINGS"
    MTS_GROUP_UNSURE_LABEL = "P_MTS_UNSURE"

    # Amount of each deicison
    PROPORTION_OF_UNCERTAINTY = "P_UNSURE"
    PROPORTION_OF_REFUTATION = "P_REFUTE"
    PROPORTION_OF_CONFIRMATION = "P_CONFIRM"
    CONFUSION_MATRIX_NORM = "CONFUSION_MATRIX_NORM"
    CONFUSION_MATRIX_NO_NORM = "CONFUSION_MATRIX_NO_NORM"

    # To maximize:
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    PROPORTION_OF_AUTOMATED_DECISIONS = "P_AUTO"

    # Constraints:
    REFUTATION_ACCURACY = "A_REFUTE"
    CONFIRMATION_ACCURACY = "A_CONFIRM"


def evaluate_decisions(mts_gt, ia_decision):
    """
    Get dict of metrics to evaluate how good module decisions were in reference to ground truths.
    Targets: U=Unsure, N=No (not a building), Y=Yes (building)
    PRedictions : U=Unsure, C=Confirmation, R=Refutation
    Confusion Matrix :
            predictions
            [Uu Ur Uc]
    target  [Nu Nr Nc]
            [Yu Yr Yc]

    Maximization criteria:
      Proportion of each decision among total of candidate groups.
      We want to maximize it.
    Accuracies:
      Confirmation/Refutation Accuracy.
      Accurate decision if either "unsure" or the same as the label.
    Quality
      Precision and Recall, assuming perfect posterior decision for unsure predictions.
      Only candidate shapes with known ground truths are considered (ambiguous labels are ignored).
      Precision :
      Recall : (Yu + Yc) / (Yu + Yn + Yc)
    """
    metrics_dict = dict()

    # VECTORS INFOS
    num_shapes = len(ia_decision)
    metrics_dict.update({MetricsNames.MTS_GROUPS_COUNT.value: num_shapes})

    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_CODES_LIST_FOR_CONFUSION, normalize=None
    )
    metrics_dict.update({MetricsNames.CONFUSION_MATRIX_NO_NORM.value: cm.copy()})

    # CRITERIA
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_CODES_LIST_FOR_CONFUSION, normalize="all"
    )
    P_MTS_U, P_MTS_N, P_MTS_C = cm.sum(axis=1)
    metrics_dict.update(
        {
            MetricsNames.MTS_GROUP_UNSURE_LABEL.value: P_MTS_U,
            MetricsNames.MTS_GROUP_NO_BUILDINGS_LABEL.value: P_MTS_N,
            MetricsNames.MTS_GROUP_BUILDING_LABEL.value: P_MTS_C,
        }
    )
    P_IA_u, P_IA_r, P_IA_c = cm.sum(axis=0)
    PAD = P_IA_c + P_IA_r
    metrics_dict.update(
        {
            MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value: PAD,
            MetricsNames.PROPORTION_OF_UNCERTAINTY.value: P_IA_u,
            MetricsNames.PROPORTION_OF_REFUTATION.value: P_IA_r,
            MetricsNames.PROPORTION_OF_CONFIRMATION.value: P_IA_c,
        }
    )

    # ACCURACIES
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_CODES_LIST_FOR_CONFUSION, normalize="pred"
    )
    RA = cm[1, 1]
    CA = cm[2, 2]
    metrics_dict.update(
        {
            MetricsNames.REFUTATION_ACCURACY.value: RA,
            MetricsNames.CONFIRMATION_ACCURACY.value: CA,
        }
    )

    # NORMALIZED CM
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_CODES_LIST_FOR_CONFUSION, normalize="true"
    )
    metrics_dict.update({MetricsNames.CONFUSION_MATRIX_NORM.value: cm.copy()})

    # QUALITY
    non_ambiguous_idx = mts_gt != FinalClassificationCodes.UNSURE.value
    ia_decision = ia_decision[non_ambiguous_idx]
    mts_gt = mts_gt[non_ambiguous_idx]
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_CODES_LIST_FOR_CONFUSION, normalize="all"
    )
    final_true_positives = cm[2, 0] + cm[2, 2]  # Yu + Yc
    final_false_positives = cm[1, 2]  # Nc
    precision = final_true_positives / (
        final_true_positives + final_false_positives
    )  #  (Yu + Yc) / (Yu + Yc + Nc)

    positives = cm[2, :].sum()
    recall = final_true_positives / positives  # (Yu + Yc) / (Yu + Yn + Yc)

    metrics_dict.update(
        {
            MetricsNames.PRECISION.value: precision,
            MetricsNames.RECALL.value: recall,
        }
    )

    return metrics_dict


def get_results_logs_str(metrics_dict: dict):
    """Format all metrics as a str for logging."""
    results_logs = "  |  ".join(
        f"{metric_enum.value}={metrics_dict[metric_enum.value]:{'' if type(metrics_dict[metric_enum.value]) is int else '.3'}}"
        for metric_enum in MetricsNames
        if metric_enum
        not in [
            MetricsNames.CONFUSION_MATRIX_NORM,
            MetricsNames.CONFUSION_MATRIX_NO_NORM,
        ]
    )
    results_logs = (
        results_logs
        + "\n"
        + str(metrics_dict[MetricsNames.CONFUSION_MATRIX_NO_NORM.value].round(3))
        + "\n"
        + str(metrics_dict[MetricsNames.CONFUSION_MATRIX_NORM.value].round(3))
    )
    return results_logs


def compute_OPTIMIZATION_penalty(auto, precision, recall):
    """
    Positive float indicative of how much a solution violates the constraint of minimal auto/precision/metrics
    """
    penalty = 0
    if precision < MIN_PRECISION_CONSTRAINT:
        penalty += MIN_PRECISION_CONSTRAINT - precision
    if recall < MIN_RECALL_CONSTRAINT:
        penalty += MIN_RECALL_CONSTRAINT - recall
    if auto < MIN_AUTOMATION_CONSTRAINT:
        penalty += MIN_AUTOMATION_CONSTRAINT - auto
    return [penalty]


def constraints_func(trial):
    return trial.user_attrs["constraint"]


def _objective(
    trial,
    group_probas,
    group_topo_overlay_bools,
    mts_gt,
):
    """Objective function for optuna optimization. Inner definition to access list of array of probas and list of targets."""
    params = {
        "min_confidence_confirmation": trial.suggest_float(
            "min_confidence_confirmation", 0.0, 1.0
        ),
        "min_frac_confirmation": trial.suggest_float("min_frac_confirmation", 0.0, 1.0),
        "min_confidence_refutation": trial.suggest_float(
            "min_confidence_refutation", 0.0, 1.0
        ),
        "min_frac_refutation": trial.suggest_float("min_frac_refutation", 0.0, 1.0),
        "min_overlay_confirmation": trial.suggest_float(
            "min_overlay_confirmation", 0.50, 1.0
        ),
    }
    decisions = [
        make_group_decision(probas, topo_overlay_bools, **params)
        for probas, topo_overlay_bools in zip(group_probas, group_topo_overlay_bools)
    ]
    metrics_dict = evaluate_decisions(mts_gt, np.array(decisions))

    values = (
        metrics_dict[MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value],
        metrics_dict[MetricsNames.PRECISION.value],
        metrics_dict[MetricsNames.RECALL.value],
    )
    auto, precision, recall = (value if not np.isnan(value) else 0 for value in values)
    trial.set_user_attr(
        "constraint", compute_OPTIMIZATION_penalty(auto, precision, recall)
    )
    return auto, precision, recall


def select_best_trial(study):
    """Find the trial that meets constraints and that maximizes automation."""
    TRIALS_BELOW_ZERO_ARE_VALID = 0
    sorted_trials = sorted(study.best_trials, key=lambda x: x.values[0], reverse=True)
    good_enough_trials = [
        s
        for s in sorted_trials
        if s.user_attrs["constraint"][0] <= TRIALS_BELOW_ZERO_ARE_VALID
    ]
    try:
        best_trial = good_enough_trials[0]
    except:
        log.warning(
            "No trial meeting constraint - returning best metrics-products instead."
        )
        sorted_trials = sorted(
            study.best_trials, key=lambda x: np.product(x.values), reverse=True
        )
        best_trial = sorted_trials[0]
    return best_trial
