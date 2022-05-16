from typing import Optional

import numpy as np

import numpy.typing as npt


def heat_equation(
    u: npt.NDArray[np.float_],
    alpha: npt.NDArray[np.float_],
    dx: int,
    dt: float,
    k_start: int,
    k_end: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float_]:
    """2D heat equation that supports field of heat coefficients.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    assert (dt <= dx**2 / (4 * alpha)).all(), "Choise of dt not numerically stable"
    nx = u.shape[1]  # number of grid cells
    assert alpha.shape == (nx, nx)

    gamma = (alpha * dt) / (dx**2)
    plate_length = u.shape[1]
    for k in range(k_start, k_end - 1, 1):
        for i in range(1, plate_length - 1, dx):
            for j in range(1, plate_length - 1, dx):
                if scale is not None:
                    noise = rng.normal(scale=scale)
                else:
                    noise = 0
                u[k + 1, i, j] = (
                    gamma[i, j]
                    * (
                        u[k][i + 1][j]
                        + u[k][i - 1][j]
                        + u[k][i][j + 1]
                        + u[k][i][j - 1]
                        - 4 * u[k][i][j]
                    )
                    + u[k][i][j]
                    + noise
                )

    return u
