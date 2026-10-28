"""Boundary-to-boundary distances used by the manuscript's HD95 and ASSD."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def surface_distance(pred, gt):
    """Return pooled bidirectional HD95 and boundary-point-weighted ASSD.

    Boundaries use one erosion with SciPy's default connectivity (4-neighbour
    in 2D). Distances are in pixels. Either empty foreground gives NaN for both
    metrics, preserving the manuscript's explicitly reported exclusion rule.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError('Prediction and target must have the same shape.')
    if not pred.any() or not gt.any():
        return float('nan'), float('nan')
    pred_boundary = pred & ~binary_erosion(pred)
    gt_boundary = gt & ~binary_erosion(gt)
    pred_to_gt = distance_transform_edt(~gt_boundary)[pred_boundary]
    gt_to_pred = distance_transform_edt(~pred_boundary)[gt_boundary]
    distances = np.concatenate((pred_to_gt, gt_to_pred))
    return float(np.percentile(distances, 95)), float(distances.mean())
