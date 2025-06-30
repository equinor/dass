"""Partial Differential Equations to use as forward models."""

from typing import Optional

import numpy as np
import numpy.typing as npt


def heat_equation(
    u0: npt.NDArray[np.float_],
    alpha: npt.NDArray[np.float_],
    dx: float,
    dt: float,
    num_steps: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float_]:
    """2D heat equation that supports field of heat coefficients.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    ny, nx = u0.shape
    assert alpha.shape == (ny, nx)

    # Pre-allocate solution array
    u = np.zeros((num_steps + 1, ny, nx))
    u[0] = u0  # Set initial condition

    gamma = (alpha * dt) / (dx**2)

    for k in range(num_steps):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                noise = rng.normal(scale=scale) if scale is not None else 0
                u[k + 1, i, j] = (
                    gamma[i, j]
                    * (
                        u[k, i + 1, j]
                        + u[k, i - 1, j]
                        + u[k, i, j + 1]
                        + u[k, i, j - 1]
                        - 4 * u[k, i, j]
                    )
                    + u[k, i, j]
                    + noise
                )

    return u
