"""Partial Differential Equations to use as forward models."""
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


def burgers(nx: int, ny: int, nt: float, nu: float) -> npt.NDArray[np.float_]:
    """
    Based on https://github.com/Masod-sadipour/Burgers-equation-convection-diffusion-in-2D
    """
    dt = 0.001

    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    u = np.ones((ny, nx))
    v = np.ones((ny, nx))
    un = np.ones((ny, nx))
    vn = np.ones((ny, nx))
    uf = np.ones((nt, nx, ny))
    vf = np.ones((nt, nx, ny))

    # assigning initial conditions
    u[int(0.75 / dy) : int(1.25 / dy + 1), int(0.75 / dx) : int(1.25 / dx + 1)] = 5
    v[int(0.75 / dy) : int(1.25 / dy + 1), int(0.75 / dx) : int(1.25 / dx + 1)] = 5

    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = (
                    (
                        un[i, j]
                        - (un[i, j] * dt / dx * (un[i, j] - un[i - 1, j]))
                        - vn[i, j] * dt / dy * (un[i, j] - un[i, j - 1])
                    )
                    + (nu * dt / (dx**2))
                    * (un[i + 1, j] - 2 * un[i, j] + un[i - 1, j])
                    + (nu * dt / (dx**2))
                    * (un[i, j - 1] - 2 * un[i, j] + un[i, j + 1])
                )
                v[i, j] = (
                    (
                        vn[i, j]
                        - (un[i, j] * dt / dx * (vn[i, j] - vn[i - 1, j]))
                        - vn[i, j] * dt / dy * (vn[i, j] - vn[i, j - 1])
                    )
                    + (nu * dt / (dx**2))
                    * (vn[i + 1, j] - 2 * vn[i, j] + vn[i - 1, j])
                    + (nu * dt / (dx**2))
                    * (vn[i, j - 1] - 2 * vn[i, j] + vn[i, j + 1])
                )
                uf[n, i, j] = u[i, j]  # U in every time-step
                vf[n, i, j] = v[i, j]  # V in every time-step
        # Velocity boundary conditions
        u[:, 0] = 1
        u[:, -1] = 1
        u[0, :] = 1
        u[-1, :] = 1
        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    return uf
