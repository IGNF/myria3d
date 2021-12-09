from enum import Enum

import numpy as np


# INITIAL TERRA-SOLID CLASSIFICATION CODES
DEFAULT_CODE = 1
MTS_AUTO_DETECTED_CODE = 6
MTS_TRUE_POSITIVE_CODE_LIST = [19]
MTS_FALSE_POSITIVE_CODE_LIST = [20, 110, 112, 114, 115]
MTS_FALSE_NEGATIVE_CODE_LIST = [21]
LAST_ECHO_CODE = 104


# FINAL CLASSIFICATION CODES - DETAIL BY SOURCE OF DECISION
class DetailedClassificationCodes(Enum):
    """Points code after decision for further analysis."""

    UNCLUSTERED = 6  # refuted
    IA_REFUTED = 33  # refuted

    IA_REFUTED_AND_DB_OVERLAYED = 34  # unsure
    BOTH_UNSURE = 35  # unsure

    IA_CONFIRMED_ONLY = 36  # confirmed
    DB_OVERLAYED_ONLY = 37  # confirmed
    BOTH_CONFIRMED = 38  # confirmed


# FINAL CLASSIFICATION CODES - SIMPLIFIED
class FinalClassificationCodes(Enum):
    """Points code for use in production."""

    UNSURE = 33  # unsure
    NOT_BUILDING = 35  # refuted
    BUILDING = 38  # confirmed


DECISION_CODES_LIST_FOR_CONFUSION = [l.value for l in FinalClassificationCodes]


DETAILED_CODE_TO_FINAL_CODE = {
    DetailedClassificationCodes.UNCLUSTERED.value: FinalClassificationCodes.NOT_BUILDING.value,
    DetailedClassificationCodes.IA_REFUTED.value: FinalClassificationCodes.NOT_BUILDING.value,
    DetailedClassificationCodes.IA_REFUTED_AND_DB_OVERLAYED.value: FinalClassificationCodes.UNSURE.value,
    DetailedClassificationCodes.BOTH_UNSURE.value: FinalClassificationCodes.UNSURE.value,
    DetailedClassificationCodes.IA_CONFIRMED_ONLY.value: FinalClassificationCodes.BUILDING.value,
    DetailedClassificationCodes.DB_OVERLAYED_ONLY.value: FinalClassificationCodes.BUILDING.value,
    DetailedClassificationCodes.BOTH_CONFIRMED.value: FinalClassificationCodes.BUILDING.value,
}


def reset_classification(classification: np.array):
    """
    Set the classification to pre-correction codes. This is not needed for production.
    FP+TP -> set to auto-detected code
    FN -> set to background code
    LAST_ECHO -> set to background code
    """
    candidate_building_points_mask = np.isin(
        classification, MTS_TRUE_POSITIVE_CODE_LIST + MTS_FALSE_POSITIVE_CODE_LIST
    )
    classification[candidate_building_points_mask] = MTS_AUTO_DETECTED_CODE
    forgotten_buillding_points_mask = np.isin(
        classification, MTS_FALSE_NEGATIVE_CODE_LIST
    )
    classification[forgotten_buillding_points_mask] = DEFAULT_CODE
    last_echo_index = classification == LAST_ECHO_CODE
    classification[last_echo_index] = DEFAULT_CODE
    return classification
