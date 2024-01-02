from torch import Tensor


def iou(confmat: Tensor):
    """Computes the Intersection over Union of each class in the
    confusion matrix

    Return:
        (iou, missing_class_mask) - iou for class as well as a mask
        highlighting existing classes
    """
    TP_plus_FN = confmat.sum(dim=0)
    TP_plus_FP = confmat.sum(dim=1)
    TP = confmat.diag()
    union = TP_plus_FN + TP_plus_FP - TP
    iou = 1e-8 + TP / (union + 1e-8)
    return iou
