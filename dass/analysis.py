"""Collection of analysis steps like ES, IES etc.
"""

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


def IES(
    Y: npt.NDArray[np.float_],
    D: npt.NDArray[np.float_],
    Cdd: npt.NDArray[np.float_],
    W: npt.NDArray[np.float_],
    gamma: float,
) -> npt.NDArray[np.float_]:
    """Iterative Ensemble Smoother based on `evensen2019`.

    Parameters
    ----------
    Y : npt.NDArray[np.float_]
        Measured responses
    D : npt.NDArray[np.float_]
        Perturbed observations
    Cdd : npt.NDArray[np.float_]
        Diagonal observation error covariance
    W : npt.NDArray[np.float_]
        Coefficient matrix
    gamma : float
        Step length

    Returns
    -------
    npt.NDArray[np.float_]
        Array such that prior multiplied by (I + W) yields posterior
    """
    N = W.shape[0]
    # Line 4 of `Algorithm 1`.
    Y_centered = Y - Y.mean(axis=1, keepdims=True)

    Omega = np.identity(N) + (W - W.mean(axis=1, keepdims=True))

    S = np.linalg.solve(Omega.T, Y_centered.T).T

    H = S @ W + D - Y

    # Eq. 42 - Inversion in measurement space
    # W = W - gamma * (W - S.T @ np.linalg.inv(S @ S.T + (N - 1) * Cdd) @ H)

    # Eq. 50 - Exact inversion without assuming a diagonal error covariance
    # notice that the inversion is computed in the ensemble subspace.
    W = W - gamma * (
        W
        - np.linalg.inv(S.T @ np.linalg.inv(Cdd) @ S + (N - 1) * np.identity(N))
        @ S.T
        @ np.linalg.inv(Cdd)
        @ H
    )

    return W
