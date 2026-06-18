import numpy as np

def compute_rmse(reference, target):
    """Return RMSE for each state component between two trajectories."""
    reference = np.asarray(reference)
    target = np.asarray(target)
    if reference.shape != target.shape:
        raise ValueError("Reference and target must have the same shape")

    diff = reference - target
    return np.sqrt(np.mean(diff**2, axis=0))


def compute_position_rmse(reference, target):
    """Return scalar RMSE for position components [x, y]."""
    reference = np.asarray(reference)
    target = np.asarray(target)
    if reference.shape != target.shape:
        raise ValueError("Reference and target must have the same shape")

    diff = reference[:, :2] - target[:, :2]
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def compute_momentum_rmse(reference, target):
    """Return scalar RMSE for momentum components [px, py]."""
    reference = np.asarray(reference)
    target = np.asarray(target)
    if reference.shape != target.shape:
        raise ValueError("Reference and target must have the same shape")

    diff = reference[:, 2:] - target[:, 2:]
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def max_position_error(reference, target):
    """Return the maximum Euclidean position error over all steps."""
    reference = np.asarray(reference)
    target = np.asarray(target)
    if reference.shape != target.shape:
        raise ValueError("Reference and target must have the same shape")

    diff = reference[:, :2] - target[:, :2]
    return np.max(np.linalg.norm(diff, axis=1))


def compare_trajectories(reference, target):
    """Compute a summary of trajectory differences."""
    return {
        "rmse_full": compute_rmse(reference, target),
        "position_rmse": compute_position_rmse(reference, target),
        "momentum_rmse": compute_momentum_rmse(reference, target),
        "max_position_error": max_position_error(reference, target),
    }
