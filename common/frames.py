"""Frame bookkeeping shared by the MVDR baseline and the BAME detector."""

from common.utils.utils_tch import list_json


def ordered_frames(prefix):
    """List the preprocessed frames of one trajectory, ordered by frame number.

    Returns (interval, frame) as two int lists, sorted by frame number.
    """
    interval, frame = list_json(prefix)
    interval = [int(i) for i in interval]
    frame = [int(i) for i in frame]
    order = sorted(range(len(frame)), key=lambda i: frame[i])
    return [interval[i] for i in order], [frame[i] for i in order]


def batch_indices(n_frames, batch=1, max_frames=None):
    """Split frame indices into consecutive batches, dropping the ragged tail."""
    eliminate = n_frames % batch
    if eliminate != 0:
        n_frames -= eliminate
    batches = [list(range(i, min(i + batch, n_frames))) for i in range(0, n_frames, batch)]
    if max_frames is not None:
        batches = batches[:max_frames]
    return batches
