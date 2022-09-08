# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EnKF using 2D heat equation
#
# **Note about coordinate system**
#
# Matplotlib's `pcolormesh` follows the standard matrix convention: "An array C with shape (nrows, ncolumns) is plotted with the column number as X and the row number as Y."
#
# This means that to get values at the point `(k, x, y)` of a field `u`, we must do `u[k, y, x]`.

# %%
# %load_ext autoreload

# %%
import numpy as np

np.set_printoptions(suppress=True)
rng = np.random.default_rng()

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams.update({"font.size": 12})
from ipywidgets import interact
import ipywidgets as widgets

from p_tqdm import p_map

# %%
# %autoreload 2
from dass import pde, utils, analysis, taper

# %% [markdown]
# # EnKF
#
# Constant parameter, unknown field.
# Stop forward model at every point in time where we have observations.

# %% [markdown]
# ## Define parameters, set true initial conditions and calculate the true temperature field

# %%
N = 100

k_start = 0
k_end = 300

# Number of grid-cells in x and y direction
nx = 20

alpha = np.ones((nx, nx)) * 5.0
# alpha[:, nx // 2 :] = 100.0

dx = 1
dt = dx**2 / (4 * np.max(alpha))  # Max dt
# Amount of noise in the heat equation is dependent on the value of `dt`.
# Making this smaller adds less noise which yields less change from one step to the other.
# dt = dt / 10.0

# True initial temperature field.
u_top = 100.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0
u = np.empty((k_end, nx, nx))
u.fill(0.0)
# Set the boundary conditions
u[:, (nx - 1) :, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (nx - 1) :] = u_right

u = pde.heat_equation(u, alpha, dx, dt, k_start, k_end, rng=rng, scale=None)


# %% [markdown]
# ## Interactive plot of true temperature field

# %%
def interactive_truth(k):
    fig, ax = plt.subplots()
    fig.suptitle("True temperature field")
    p = ax.pcolormesh(u[k], vmin=0, vmax=100)
    ax.set_title(f"k = {k}")
    utils.colorbar(p)
    fig.tight_layout()


interact(
    interactive_truth,
    k=widgets.IntSlider(min=k_start, max=k_end - 1, step=1, value=0),
)

# %% [markdown]
# ## Define random seeds because multiprocessing
#
# https://numpy.org/doc/stable/reference/random/parallel.html#seedsequence-spawning

# %%
ss = np.random.SeedSequence(12345)
child_seeds = ss.spawn(N)
streams = [np.random.default_rng(s) for s in child_seeds]

# %% [markdown]
# ## Generate truth and observations based on truth

# %%
u = pde.heat_equation(
    u, alpha, dx, dt, k_start, k_end, np.random.default_rng(12345), np.sqrt(dt)
)

# placement of sensors, i.e, where the observations are done
# padding = int(0.15 * nx)
# x = np.linspace(padding, nx - padding, 3, dtype=int)
# y = np.linspace(padding, nx - padding, 3, dtype=int)
# obs_coordinates = [utils.Coordinate(xc, yc) for xc in x for yc in y]

obs_coordinates = [utils.Coordinate(nx // 2, nx - 2)]

# At which times observations are taken
obs_times = np.linspace(5, k_end, 50, endpoint=False, dtype=int)

d = utils.observations(obs_coordinates, obs_times, u, lambda value: abs(0.01 * value))
# number of measurements
m = d.shape[0]
print("Number of observations: ", m)

# %%
# Plot temperature field and show placement of sensors.
obs_xy = set(zip(d.index.get_level_values("x"), d.index.get_level_values("y")))
x, y = zip(*obs_xy)

fig, ax = plt.subplots()
p = ax.pcolormesh(u[100], cmap=plt.cm.viridis)
fig.colorbar(p)
ax.plot(x, y, "s", color="white", markersize=15)


# %%
def gen_field():
    u = np.empty((k_end, nx, nx))
    u.fill(0.0)

    # Set the boundary conditions
    # u[:, (nx - 1) :, :] = (u_top / 2) + rng.normal(0, 20, nx)
    u[:, (nx - 1) :, :] = (u_top / 2) + rng.normal(0, 20)
    u[:, :, :1] = u_left
    u[:, :1, 1:] = u_bottom
    u[:, :, (nx - 1) :] = u_right

    return u


fields = [gen_field() for _ in range(N)]


# %%
def matrix_from_fields(fields, k):
    nx = fields[0][0].shape[0]
    A = np.zeros(shape=(nx * nx, N))
    for f in range(len(fields)):
        A[:, f] = fields[f][k].ravel()
    return A


# %%
d.index.get_level_values("k").unique()

# %% [markdown]
# ## Plot tapering function used for localisation

# %%
fig, ax = plt.subplots()
ax.plot(taper.gauss(np.linspace(-nx, nx), 3.0))

# %%
A_no_update = {}
localize = True
k_start = 0
for k_obs in d.index.get_level_values("k").unique().to_list():
    fields = p_map(
        pde.heat_equation,
        fields,
        [alpha] * N,
        [dx] * N,
        [dt] * N,
        [k_start] * N,
        [k_obs + 1] * N,
        streams,
        [np.sqrt(dt)] * N,
        desc=f"Running forward model from {k_start} to {k_obs}",
    )

    d_k = d.query(f"k == {k_obs}")
    m = d_k.shape[0]

    A = matrix_from_fields(fields, k_obs)
    A_no_update[k_obs] = A.copy()

    # measure response
    Y = np.zeros(shape=(m, N))
    for i in range(N):
        # It's A[y, x] and not A[x, y] due to convention followed by matplotlib's pcolormesh.
        Y[:, i] = A[:, i].reshape(nx, nx)[
            d_k.index.get_level_values("y").to_list(),
            d_k.index.get_level_values("x").to_list(),
        ]

    Cdd = np.diag(d_k.sd.values**2)

    E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
    E = E - E.mean(axis=1, keepdims=True)
    assert E.shape == (m, N)

    D = np.ones((m, N)) * d_k.value.values.reshape(-1, 1) + E

    S = Y - Y.mean(axis=1, keepdims=True)

    A_centered = A - A.mean(axis=0, keepdims=True)
    Dprime = D - Y

    if localize:
        for i in range(nx**2):
            state_idx = np.unravel_index(i, shape=(nx, nx))
            dist = np.sqrt(
                (state_idx[0] - obs_coordinates[0].y) ** 2
                + (state_idx[1] - obs_coordinates[0].x) ** 2
            )
            taper_coeff = taper.gauss(dist, 2.0)

            K = A_centered[i, :] @ S.T @ np.linalg.pinv(S @ S.T + (N - 1) * Cdd)

            # K = (
            #    A_centered[i, :]
            #    @ S.T
            #    @ np.linalg.pinv(S @ S.T + (N - 1) * (1 / taper_coeff**2) * Cdd)
            # )

            A[i, :] = A[i, :] + np.sqrt(taper_coeff) * K @ (Dprime)
    else:
        K = A_centered @ S.T @ np.linalg.pinv(S @ S.T + (N - 1) * Cdd)
        A = A + K @ Dprime

    for i in range(N):
        fields[i][k_obs:] = A[:, i].reshape(nx, nx)

    k_start = k_obs

# %% [markdown]
# ## Plot difference between prior and posterior of a single update

# %%
k_obs = 5
prior_mean_field = A_no_update[k_obs].mean(axis=1).reshape(nx, nx)
posterior_mean_field = matrix_from_fields(fields, k_obs).mean(axis=1).reshape(nx, nx)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
p1 = axes[0].pcolormesh(posterior_mean_field - prior_mean_field)
utils.colorbar(p1)

axes[1].plot(posterior_mean_field.ravel() - prior_mean_field.ravel())

fig.tight_layout()


# %% [markdown]
# ## Interactive plotting

# %%
def updated_vs_truth(k):
    vmin = 0
    vmax = 100
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f"k = {k}")

    A_with_update = matrix_from_fields(fields, k)

    axes[0].set_title("With update")
    p0 = axes[0].pcolormesh(
        A_with_update.mean(axis=1).reshape(nx, nx),
        cmap=plt.cm.viridis,
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title("Truth")
    p1 = axes[1].pcolormesh(u[k], cmap=plt.cm.viridis, vmin=vmin, vmax=vmax)

    utils.colorbar(p1)

    fig.tight_layout()


interact(
    updated_vs_truth,
    k=widgets.IntSlider(
        min=0, max=d.index.get_level_values("k").max(), step=1, value=0
    ),
)

# %%
