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

# %%
# %load_ext autoreload

# %%
import numpy as np

np.set_printoptions(suppress=True)
rng = np.random.default_rng()

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams.update({"font.size": 10})
# Ignore error when drawing many figures
plt.rcParams.update({"figure.max_open_warning": 0})
from ipywidgets import interact
import ipywidgets as widgets

from p_tqdm import p_map

from scipy.ndimage import gaussian_filter

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
nx = 10

# Set the coefficient of heat transfer for each grid cell.
# Using trick from page 15 of "An Introduction to the Numerics of Flow in Porous Media using Matlab".
# It's a nice way to generate realistic-looking parameter fields.
# In real life we use third-party tools to generate good (whatever that means) prior parameter fields.
alpha_t = np.exp(
    5
    * gaussian_filter(gaussian_filter(rng.random(size=(nx, nx)), sigma=2.0), sigma=1.0)
)


dx = 1
dt = dx**2 / (4 * np.max(alpha_t))

# True initial temperature field.
u_top = 100.0
u_init = np.empty((k_end, nx, nx))
u_init.fill(0.0)

# Set the boundary conditions
u_init[:, 1 : (nx - 1), 0] = u_top

# How much noise to add to heat equation, also called model noise.
# scale = 0.1
scale = None

u_t = pde.heat_equation(u_init, alpha_t, dx, dt, k_start, k_end, rng=rng, scale=scale)


# %% [markdown]
# ## Interactive plot of true temperature field

# %%
def interactive_truth(k):
    fig, ax = plt.subplots()
    fig.suptitle("True temperature field")
    p = ax.pcolormesh(u_t[k].T, cmap=plt.cm.jet)
    ax.set_title(f"k = {k}")
    ax.invert_yaxis()
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
# ## Define placement of sensors and generate synthetic observations based on the true temperature field

# %%
# placement of sensors, i.e, where the observations are done
pad = 1
coords = np.array([(x, y) for x in range(pad, nx - pad) for y in range(pad, nx - pad)])
ncoords = coords.shape[0]
nmeas = 40
coords_idx = np.random.choice(np.arange(ncoords), size=nmeas, replace=False)
obs_coordinates = [utils.Coordinate(xc, yc) for xc, yc in coords[coords_idx]]

# At which times observations are taken
obs_times = np.linspace(5, k_end, 50, endpoint=False, dtype=int)

d = utils.observations(obs_coordinates, obs_times, u_t, lambda value: abs(0.05 * value))
# number of measurements
m = d.shape[0]
print("Number of observations: ", m)

k_levels = d.index.get_level_values("k").to_list()
x_levels = d.index.get_level_values("x").to_list()
y_levels = d.index.get_level_values("y").to_list()

# Plot temperature field and show placement of sensors.
obs_coordinates_from_index = set(zip(x_levels, y_levels))
x, y = zip(*obs_coordinates_from_index)

fig, ax = plt.subplots()
p = ax.pcolormesh(u_t[0].T, cmap=plt.cm.jet)
ax.invert_yaxis()
ax.set_title("True temperature field with sensor placement")
utils.colorbar(p)
ax.plot([i + 0.5 for i in x], [j + 0.5 for j in y], "s", color="white", markersize=5)


# %%
def gen_field():
    u = np.empty((k_end, nx, nx))

    u_top = 100.0
    u_init = np.empty((k_end, nx, nx))
    u_init.fill(0.0)

    # Set the boundary conditions
    u_init[:, 1 : (nx - 1), 0] = u_top + rng.normal()

    return u


fields = [gen_field() for _ in range(N)]


# %%
def matrix_from_fields(fields, k):
    nx = fields[0][0].shape[0]
    A = np.zeros(shape=(nx * nx, N))
    for f in range(len(fields)):
        A[:, f] = fields[f][k].ravel()
    return A


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
        [alpha_t] * N,
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
        Y[:, i] = A[:, i].reshape(nx, nx)[
            d_k.index.get_level_values("x").to_list(),
            d_k.index.get_level_values("y").to_list(),
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
                (state_idx[0] - obs_coordinates[0].x) ** 2
                + (state_idx[1] - obs_coordinates[0].y) ** 2
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
    axes[0].invert_yaxis()
    p0 = axes[0].pcolormesh(
        A_with_update.mean(axis=1).reshape(nx, nx).T,
        cmap=plt.cm.viridis,
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title("Truth")
    axes[1].invert_yaxis()
    p1 = axes[1].pcolormesh(u_t[k].T, cmap=plt.cm.viridis, vmin=vmin, vmax=vmax)

    utils.colorbar(p1)

    fig.tight_layout()


interact(
    updated_vs_truth,
    k=widgets.IntSlider(
        min=0, max=d.index.get_level_values("k").max(), step=1, value=0
    ),
)

# %%
