# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Snake_oil data assimilation with dass
#
# First we clone the snake_oil dataset from ert, and run an ensemble smoother experiment with ert. Here shown with git sparse checkout (available in newer git):
#

# Sparse checkout of snake_oil from ert:
# !git clone --depth 1 --filter=blob:none --sparse https://github.com/equinor/ert
# %cd ert
# !git sparse-checkout set test-data/local/snake_oil
# %cd ..

# Run ERT with ensemble smoother (~3 minutes)
# %cd ert/test-data/local/snake_oil
# !ert ensemble_smoother snake_oil.ert --target-case default
% cd ../../../..

from pathlib import Path
from dass import analysis
import fmu.ensemble  # NB: We need bugfix in PR 221 merged
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subscript
from subscript.fmuobs.parsers import ertobs2df
from subscript.fmuobs.writers import df2obsdict

# At this point, we assert there is a runpath for the prior available
runpath = "ert/test-data/local/snake_oil/storage/snake_oil/runpath/"


prior = fmu.ensemble.ScratchEnsemble(
    "snaky", paths=(runpath + "realization-*/iter-0")
)
prior.find_files("*.UNSMRY")
prior.load_smry()

# Load the updated parameters as ERT did it (for comparison):
ert_posterior = fmu.ensemble.ScratchEnsemble(
    "snaky", paths=(runpath + "realization-*/iter-1")
)

obs_df: pd.DataFrame = ertobs2df(
    Path("ert/test-data/local/snake_oil/observations/observations.txt").read_text(encoding="utf-8")
)
obs_df

# Slice to the kind of observations that fmu.ensemble supports:
obs_df = obs_df[obs_df["CLASS"] == "SUMMARY_OBSERVATION"].sort_values("LABEL")
obs_df

obs_dict = df2obsdict(obs_df)
ens_obs = fmu.ensemble.Observations(obs_dict)
print(ens_obs)

misfit = ens_obs.mismatch(prior)
print(misfit)

# The prior (as in its parameters) in Evensen's formulation:
A = prior.parameters.set_index("REAL").sort_index().values.T

Cdd = np.diag(obs_df["ERROR"].values ** 2)

# Ensemble representation for measurements (aka observations):
rng = np.random.default_rng()
E = rng.multivariate_normal(mean=np.zeros(len(Cdd)), cov=Cdd, size=len(prior)).T
E = E - E.mean(axis=1, keepdims=True)
D = np.ones((len(obs_df), len(prior))) * obs_df["VALUE"].values.reshape(-1, 1) + E

# Extract simulated values and align in matrix
Y = (
    misfit.sort_values(["REAL", "LABEL"])[["REAL", "SIMVALUE", "LABEL"]]
    .pivot(index="LABEL", columns="REAL", values="SIMVALUE")
    .values
)

assert Y.shape == (len(obs_df), len(prior))

# Perform ES update
X = analysis.ES(Y, D, Cdd)

A_ES = A @ X

print(prior.parameters.set_index("REAL").columns.values)
np.set_printoptions(suppress=True)
print(A.mean(axis=1))
print(A_ES.mean(axis=1))

# %matplotlib inline
bins=10
for p_idx, parametername in enumerate(prior.parameters.set_index("REAL")):
        plt.figure()
        plt.hist(prior.parameters[parametername], bins, alpha=0.5, label='prior')
        plt.hist(ert_posterior.parameters[parametername], bins, alpha=0.5, label='ert-post')
        plt.hist(A_ES[p_idx], bins, alpha=0.5, label="dass")
        plt.legend(loc='upper right')
        plt.title(parametername)
        









