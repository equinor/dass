import numpy as np
import numpy.typing as npt


def ES(
    Y: npt.NDArray[np.float_],
    D: npt.NDArray[np.float_],
    Cdd: npt.NDArray[np.float_],
    taper_coeff: float = 1.0,
) -> npt.NDArray[np.float_]:
    """Ensemble Smoother based on `evensen2009` section 9.5.

    Parameters
    ----------
    Y : npt.NDArray[np.float_]
        Measured responses
    D : npt.NDArray[np.float_]
        Perturbed observations
    Cdd : npt.NDArray[np.float_]
        Diagonal observation error covariance
    taper_coeff: float
        Coefficient from tapering function used in localisation.
        Scales innovation (`Dprime`) down and the observation error covariance (`Cdd`) up.

    Returns
    -------
    npt.NDArray[np.float_]
        Array to multiply with prior to get posterior.
    """
    N = D.shape[1]
    # Eq. 9.24 with `taper_coeff` added
    Dprime = taper_coeff * (D - Y)
    S = Y - Y.mean(axis=1, keepdims=True)  # Eq. 9.25

    # Eq. 9.26 with `taper_coeff` added
    C = S @ S.T + (N - 1) * (1 / taper_coeff**2) * Cdd

    # Eq. 9.27
    X = np.identity(N) + S.T @ np.linalg.inv(C) @ Dprime

    return X
