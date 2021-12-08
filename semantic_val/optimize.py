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
from semantic_val.datamodules.processing import ChannelNames
from semantic_val.utils import utils
from semantic_val.decision.decide import (
    CODE_TO_LABEL_MAPPER,
    MTS_TRUE_POSITIVE_CODE_LIST,
    DecisionLabels,
    MetricsNames,
    prepare_las_for_decision,
    evaluate_decisions,
    get_results_logs_str,
    make_group_decision,
    reset_classification,
    split_idx_by_dim,
    update_las_with_decisions,
)

log = utils.get_logger(__name__)


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
    input_bd_topo_shp = config.optimize.input_bd_topo_shp
    input_bd_parcellaire_shp = config.optimize.input_bd_parcellaire_shp
    output_dir = config.optimize.results_output_dir

    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Best trial and outputs will be saved in {output_dir}")
    log.info(f"Logs will be saved in {os.getcwd()}")
    las_filepaths = glob.glob(osp.join(input_dir, "*.las"))
    # DEBUG
    las_filepaths = [
        # "/var/data/cgaydon/data/202110_building_val/logs/good_checkpoints/V2.0/validation_preds/test_870000_6649000.las",
        "/var/data/cgaydon/data/202110_building_val/logs/good_checkpoints/V2.0/validation_preds/test_792000_6272000.las"
    ]
    print(las_filepaths)

    ### CLUSTER AND GET PROBAS AND TARGETS FOR LATER OPTIMIZATION
    if "prepare" in config.optimize.todo:
        (
            group_probas,
            group_topo_overlay_bools,
            group_parcellaire_overlay_bools,
            mts_gt,
        ) = get_group_info_and_label(
            las_filepaths, input_bd_topo_shp, input_bd_parcellaire_shp, output_dir
        )
        probas_target_groups_filepath = osp.join(output_dir, "probas_target_groups.pkl")
        with open(probas_target_groups_filepath, "wb") as f:
            pickle.dump(
                (
                    group_probas,
                    group_topo_overlay_bools,
                    group_parcellaire_overlay_bools,
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
                group_parcellaire_overlay_bools,
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
                group_parcellaire_overlay_bools,
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
                group_parcellaire_overlay_bools,
                mts_gt,
            ) = pickle.load(f)
        log.info(f"Evaluating best trial on N={len(mts_gt)} groups of points.")
        decision_codes = [
<<<<<<< HEAD
            make_group_decision(probas, overlay_bools, **best_trial.params)
            for probas, overlay_bools in zip(group_probas, group_overlay_bools)
=======
            make_group_decision(
                probas,
                topo_overlay_bools,
                parcellaire_overlay_bools,
                **best_trial.params,
            )
            for probas, topo_overlay_bools, parcellaire_overlay_bools in zip(
                group_probas, group_topo_overlay_bools, group_parcellaire_overlay_bools
            )
>>>>>>> 0c0a676... Integrate BDParcellaire as additional source of proof
        ]
        decision_labels = [
            CODE_TO_LABEL_MAPPER[decision_code] for decision_code in decision_codes
        ]
<<<<<<< HEAD
=======

>>>>>>> 0c0a676... Integrate BDParcellaire as additional source of proof
        metrics_dict = evaluate_decisions(mts_gt, np.array(decision_labels))
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
            las.classification = reset_classification(las.classification)
            with open(config.optimize.best_trial_pickle_path, "rb") as f:
                best_trial = pickle.load(f)
                log.info(
                    f"Using best trial from: {config.optimize.best_trial_pickle_path}"
                )
            las = update_las_with_decisions(las, best_trial.params)
            out_path = osp.join(output_dir, "POST_IA_" + basename)
            las.write(out_path)
            log.info(f"Saved update las to {out_path}")


# Functions


def define_MTS_ground_truth_flag(frac_true_positive):
    """Helper : Based on the fraction of confirmed building points, set the nature of the shape or declare an ambiguous case"""
    FP_FRAC = 0.05
    TP_FRAC = 0.95
    if frac_true_positive >= TP_FRAC:
        return DecisionLabels.BUILDING.value
    elif frac_true_positive < FP_FRAC:
        return DecisionLabels.NOT_BUILDING.value
    else:
        return DecisionLabels.UNSURE.value


def get_group_info_and_label(
    las_filepaths: List[str],
    input_bd_topo_shp: str,
    input_bd_parcellaire_shp: str,
    output_dir: str,
) -> Tuple[List[np.array], List[str]]:
    """
    From a folder of las with probabilities, cluster the clouds and append both list of predicted probas and MTS ground truth to lists.
    group_probas: the probas of each group of points
    mts_gt: the group label based on ground truths
    """
    group_probas = []
    group_topo_overlay_bools = []
    group_parcellaire_overlay_bools = []
    mts_gt = []
    for las_filepath in tqdm(las_filepaths, desc="Preparing  ->"):
        log.info(f"Preparing tile: {las_filepath}...")
        basename = osp.basename(las_filepath)
        out_path = osp.join(output_dir, "PREPARED_" + basename)
        structured_array = prepare_las_for_decision(
            las_filepath, input_bd_topo_shp, input_bd_parcellaire_shp, out_path
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
            group_parcellaire_overlay_bools.append(
                structured_array[ChannelNames.BDParcellaireOverlay.value][pts_idx]
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
        group_parcellaire_overlay_bools,
        np.array(mts_gt),
    )


# TODO: extract constraints as global constant
def compute_OPTIMIZATION_penalty(auto, precision, recall):
    """
    Positive float indicative of how much a solution violates the constraint of minimal auto/precision/metrics
    """
    penalty = 0
    if precision < 0.98:
        penalty += 0.98 - precision
    if recall < 0.98:
        penalty += 0.98 - recall
    if auto < 0.35:
        penalty += 0.35 - auto
    return [penalty]


def constraints_func(trial):
    return trial.user_attrs["constraint"]


def _objective(
    trial,
    group_probas,
    group_topo_overlay_bools,
    group_parcellaire_overlay_bools,
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
        "min_topo_overlay_confirmation": trial.suggest_float(
            "min_topo_overlay_confirmation", 0.50, 1.0
        ),
        "min_parcellaire_overlay_confirmation": trial.suggest_float(
            "min_parcellaire_overlay_confirmation", 0.50, 1.0
        ),
    }
    decision_codes = [
        make_group_decision(
            probas, topo_overlay_bools, parcellaire_overlay_bools, **params
        )
        for probas, topo_overlay_bools, parcellaire_overlay_bools in zip(
            group_probas, group_topo_overlay_bools, group_parcellaire_overlay_bools
        )
    ]
    decision_labels = [
        CODE_TO_LABEL_MAPPER[decision_code] for decision_code in decision_codes
    ]
    metrics_dict = evaluate_decisions(mts_gt, np.array(decision_labels))

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
        best_trial = sorted_trials[0]
    return best_trial
