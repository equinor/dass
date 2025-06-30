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
    u = np.zeros((num_steps + 1, ny, nx))
    u[0] = u0
    gamma = (alpha * dt) / (dx**2)

    for k in range(num_steps):
        # Vectorized finite difference
        u[k + 1, 1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (
                u[k, 2:, 1:-1]  # i+1
                + u[k, :-2, 1:-1]  # i-1
                + u[k, 1:-1, 2:]  # j+1
                + u[k, 1:-1, :-2]  # j-1
                - 4 * u[k, 1:-1, 1:-1]
            )
            + u[k, 1:-1, 1:-1]
        )

        # Add noise if needed
        if scale is not None:
            noise = rng.normal(0, scale, size=(ny - 2, nx - 2))
            u[k + 1, 1:-1, 1:-1] += noise

    return u
