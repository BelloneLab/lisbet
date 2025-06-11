"""LISBET"""

import imageio.v3 as iio
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score


def permute_predictions(predictions, labels):
    """ """
    # Permute HMM ids to match human labels
    n_states = np.max(predictions) + 1
    n_classes = np.max(labels) + 1

    hist2d = []
    for s in range(n_states):
        bin_pred = np.array(predictions == s, dtype=int)
        score = []
        for lbl in range(n_classes):
            bin_lab = np.array(labels == lbl, dtype=int)
            score.append(f1_score(bin_lab, bin_pred))
        hist2d.append(score)
    hist2d = np.array(hist2d)

    # Sort motifs to best match labels
    behavior_ids, motif_ids = linear_sum_assignment(hist2d.T, maximize=True)

    # Complete the assignment by adding the missing motifs
    missing_ids = list(set(np.unique(predictions)) - set(motif_ids))
    matched_ids = np.concatenate((motif_ids, missing_ids))

    # Match labels
    predictions_matched = np.array(predictions)
    for bhv, mtf in enumerate(matched_ids):
        predictions_matched[np.where(predictions == mtf)] = bhv

    return predictions_matched


def export_pose_video(frames, filename, fps=30):
    """
    Export a sequence of frames (numpy arrays, shape [N, H, W, 3]) to mp4 using imageio.

    Args:
        frames: numpy array of shape (num_frames, height, width, 3), dtype=uint8
        filename: output mp4 file path
        fps: frames per second
    """
    iio.imwrite(filename, frames, fps=fps, codec="libx264", quality=8)
