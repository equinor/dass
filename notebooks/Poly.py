# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Poly case

# %%
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
rng = np.random.default_rng()

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams.update({"font.size": 10})
from ipywidgets import interact
import ipywidgets as widgets

from p_tqdm import p_map

from scipy.ndimage import gaussian_filter

import iterative_ensemble_smoother as ies

# %%
# %load_ext autoreload
# %autoreload 2
from dass import pde, utils, analysis, taper

# %%
N = 200


# %%
def poly(a, b, c, x):
    return a * x**2 + b * x + c


# %%
import numpy as np

rng = np.random.default_rng(12345)

a_t = 0.5
b_t = 1.0
c_t = 3.0


def p(x):
    return a_t * x**2 + b_t * x + c_t


x_observations = [0, 2, 4, 6, 8]
observations = [
    (p(x) + rng.normal(loc=0, scale=0.2 * p(x)), 0.2 * p(x), x) for x in x_observations
]

d = pd.DataFrame(observations, columns=["value", "sd", "x"])

d = d.set_index("x")

m = d.shape[0]

# %%
fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")
ax.grid()

# %%
# Assume diagonal ensemble covariance matrix for the measurement perturbations.
Cdd = np.diag(d.sd.values**2)

# 9.4 Ensemble representation for measurements
E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=N).T
E = E - E.mean(axis=1, keepdims=True)
assert E.shape == (m, N)

D = np.ones((m, N)) * d.value.values.reshape(-1, 1) + E

# %%
coeff_a = rng.normal(0, 1, size=N)
coeff_b = rng.normal(0, 1, size=N)
coeff_c = rng.normal(0, 1, size=N)

# %%
A = np.concatenate(
    (coeff_a.reshape(-1, 1), coeff_b.reshape(-1, 1), coeff_c.reshape(-1, 1)), axis=1
).T

# %%
fwd_runs = p_map(
    poly,
    coeff_a,
    coeff_b,
    coeff_c,
    [np.arange(max(x_observations) + 1)] * N,
    desc=f"Running forward model.",
)

# %%
Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    m,
    N,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"

# %%
X = analysis.ES(Y, D, Cdd)
A_ES = A @ X

# %%
fig, ax = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(10, 10)

ax[0, 0].set_title("a - prior")
ax[0, 0].hist(A[0, :])

ax[0, 1].set_title("a - posterior")
ax[0, 1].hist(A_ES[0, :])
ax[0, 1].axvline(a_t, color="k", linestyle="--", label="truth")
ax[0, 1].legend()

ax[1, 0].set_title("b - prior")
ax[1, 0].hist(A[1, :])

ax[1, 1].set_title("b - posterior")
ax[1, 1].hist(A_ES[1, :])
ax[1, 1].axvline(b_t, color="k", linestyle="--", label="truth")
ax[1, 1].legend()

ax[2, 0].set_title("c - prior")
ax[2, 0].hist(A[2, :])

ax[2, 1].set_title("c - posterior")
ax[2, 1].hist(A_ES[2, :])
ax[2, 1].axvline(c_t, color="k", linestyle="--", label="truth")
ax[2, 1].legend()

fig.tight_layout()

# %%
