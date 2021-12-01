import pickle
import glob
import os
import os.path as osp
from typing import List, Tuple
import hydra
from omegaconf.base import SCMode
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import numpy as np
import optuna
import laspy
from semantic_val.callbacks.predictions_callbacks import ChannelNames
from semantic_val.utils import utils
from semantic_val.decision.decide import (
    MTS_TRUE_POSITIVE_CODE_LIST,
    DecisionLabels,
    MetricsNames,
    get_post_ia_las_filepath,
    cluster,
    evaluate_decisions,
    get_results_logs_str,
    make_decisions,
    reset_classification,
    split_idx_by_dim,
    update_las_with_decisions,
)

log = utils.get_logger(__name__)


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


def get_probas_and_target_by_group(
    las_filepaths: List[str],
) -> Tuple[List[np.array], List[str]]:
    """
    From a folder of las with probabilities, cluster the clouds and append both list of predicted probas and MTS ground truth to lists.
    group_probas: the probas of each group of points
    mts_gt: the group label based on ground truths
    """
    group_probas = []
    mts_gt = []
    for las_filepath in tqdm(las_filepaths, desc="Clustering  ->"):
        log.info(f"Clustering tile: {las_filepath}...")
        structured_array = cluster(las_filepath, get_post_ia_las_filepath(las_filepath))
        if len(structured_array) == 0:
            log.info("/!\ Skipping tile: there are no candidate building points.")
            continue
        split_idx = split_idx_by_dim(structured_array["ClusterID"])
        split_idx = split_idx[1:]  # remove large group with ClusterID = 0
        for pts_idx in tqdm(split_idx, desc="Append probas and targets  ->"):
            # TODO: optimize splitting with https://stackoverflow.com/a/28156406/8086033
            probas = structured_array[ChannelNames.BuildingsProba.value][pts_idx]
            group_probas.append(probas)
            frac_true_positive = np.mean(
                np.isin(
                    structured_array["Classification"][pts_idx],
                    MTS_TRUE_POSITIVE_CODE_LIST,
                )
            )
            mts_gt.append(define_MTS_ground_truth_flag(frac_true_positive))
    mts_gt = np.array(mts_gt)
    return group_probas, mts_gt


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


def select_best_trial(study):
    """Find the trial that meets constraints and that optimizes the product of the three maximized metrics."""
    TRIALS_BELOW_ZERO_ARE_VALID = 0
    sorted_trials = sorted(study.best_trials, key=lambda x: np.product(x), reverse=True)
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


def optimize(config: DictConfig) -> Tuple[float]:
    """
    Parse predicted las, cluster groups of points, ottimize decision thresholds, then use them to product final las.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    """ 
    for las in las_list:
        cluster with pdal, save to output_las_file
        add the X and target informations in a list
        pickle the list

    Run an hyperoptimisation
    Find the best solution
    Pickle it where las are saved.
    evaluate_decision with this set of solution.

    for las in las_list:
        reset_classif
        update with thresholds
        save to output_las_file_BIS
        delete previous one and rename output_las_file_BIS to output_las_file (to be safe)
     """
    ### GROUP ALL POINTS WITH PDAL,
    os.chdir(config.optimize.results_output_dir)
    os.makedirs(os.getcwd(), exist_ok=True)
    log.info(f"Best trial and outputs will be saved in {os.getcwd()}")

    las_filepaths = glob.glob(osp.join(config.optimize.predicted_las_dirpath, "*.las"))
    # DEBUG

    las_filepaths = [
        "/var/data/cgaydon/data/202110_building_val/logs/good_checkpoints/V2.0/validation_preds/test_870000_6649000.las",
        # "/var/data/cgaydon/data/202110_building_val/logs/good_checkpoints/V2.0/validation_preds/test_792000_6272000.las"
    ]

    print(las_filepaths)
    probas_target_groups_filepath = osp.join(os.getcwd(), "probas_target_groups.pkl")
    if config.optimize.resume_from_pickled_groups:
        with open(probas_target_groups_filepath, "rb") as f:
            group_probas, mts_gt = pickle.load(f)
    else:
        group_probas, mts_gt = get_probas_and_target_by_group(las_filepaths)
        with open(probas_target_groups_filepath, "wb") as f:
            pickle.dump((group_probas, mts_gt), f)

    ### OPTIMIZE WITH OPTUNA
    log.info(f"Optimizing on N={len(mts_gt)} groups of points.")

    def objective(trial):
        """Objective function for optuna optimization. Inner definition to access list of probas-targets."""
        min_confidence_confirmation = trial.suggest_float(
            "min_confidence_confirmation", 0.0, 1.0
        )
        min_frac_confirmation = trial.suggest_float("min_frac_confirmation", 0.0, 1.0)
        min_confidence_refutation = trial.suggest_float(
            "min_confidence_refutation", 0.0, 1.0
        )
        min_frac_refutation = trial.suggest_float("min_frac_refutation", 0.0, 1.0)
        ia_decision = [
            make_decisions(
                probas,
                min_confidence_confirmation=min_confidence_confirmation,
                min_frac_confirmation=min_frac_confirmation,
                min_confidence_refutation=min_confidence_refutation,
                min_frac_refutation=min_frac_refutation,
            )
            for probas in group_probas
        ]
        metrics_dict = evaluate_decisions(mts_gt, np.array(ia_decision))

        values = (
            metrics_dict[MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value],
            metrics_dict[MetricsNames.PRECISION.value],
            metrics_dict[MetricsNames.RECALL.value],
        )
        auto, precision, recall = (
            value if not np.isnan(value) else 0 for value in values
        )
        trial.set_user_attr(
            "constraint", compute_OPTIMIZATION_penalty(auto, precision, recall)
        )
        return auto, precision, recall

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
    study.optimize(objective, n_trials=config.optimize.study.n_trials)

    best_trial = select_best_trial(study)
    log.info("Best_trial: \n")
    log.info(best_trial)
    best_trial_path = osp.join(os.getcwd(), "best_trial.pkl")
    with open(best_trial_path, "wb") as f:
        pickle.dump(best_trial, f)
        log.info(f"Best trial stored in: {best_trial_path}")

    ### EVALUATE WITH BEST PARAMS
    ia_decision = np.array(
        [make_decisions(probas, **best_trial.params) for probas in group_probas]
    )
    mts_gt = mts_gt
    metrics_dict = evaluate_decisions(mts_gt, ia_decision)
    log.info(f"\n Metrics and Confusion Matrices: {get_results_logs_str(metrics_dict)}")

    ### UPDATING LAS
    if config.optimize.update_las:
        log.info(f"Validated las will be saved in {os.getcwd()}")
        log.info("The following params are used :")
        log.info(best_trial.params)
        for las_filepath in tqdm(las_filepaths, desc="Validate Las"):
            # we reuse post_ia path that already contains clustered las.
            out_path = get_post_ia_las_filepath(las_filepath)
            las = laspy.read(out_path)

            las.classification = reset_classification(las.classification)
            # TODO: delete leftovers fields : "BuildingsPreds", BuildingsConfusion, BuildingsHasPredictions = "BuildingsHasPredictions"
            las = update_las_with_decisions(las, best_trial.params)
            las.write(out_path)
