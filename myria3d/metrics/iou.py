from torch import Tensor

EPSILON = 1e-8


def iou(confmat: Tensor):
    """Computes the Intersection over Union of each class in the
    confusion matrix

    Return:
        (iou, missing_class_mask) - iou for class as well as a mask
        highlighting existing classes
    """
    true_positives_and_false_negatives = confmat.sum(dim=0)
    true_positives_and_false_positives = confmat.sum(dim=1)
    true_positives = confmat.diag()
    union = (
        true_positives_and_false_negatives + true_positives_and_false_positives - true_positives
    )
    iou = EPSILON + true_positives / (union + EPSILON)
    return iou
