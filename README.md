# Data Assimilation

Implementations of Ensemble Smoother as given in [1] and Iterative Ensemble Smoother (IES) as given in [2],
see [dass/analysis.py](dass/analysis.py).

Examples / Tutorials:

- [notebooks/ES_2D_Heat_Equation.ipynb](notebooks/ES_2D_Heat_Equation.ipynb) for parameter estimation using ES and the Heat Equation.
- [notebooks/EnKF_2D_Heat_Equation.ipynb](notebooks/EnKF_2D_Heat_Equation.ipynb) for sequential data assimilation or filtering using EnKF and the Heat Equation.
- [notebooks/IES_Burgers.ipynb](notebooks/IES_Burgers.ipynb) for parameter estimation using IES and Burgers' equation.

## Installation

```bash
git clone https://github.com/equinor/dass.git
cd dass
# dass supports Python 3.8 and above.
python3.9 -m venv .venvdass
source .venvdass/bin/activate
# Add -e if you want to make changes.
pip install -e .
# Install additional requirements for developers.
pip install -r dev-requirements.txt
# Start jupyter notebook
jupyter notebook
# To make sure everything works, run Smoothers.ipynb in the notebooks folder
```

## On notation

The implementation of ES is based on section 9.5 of [1], while the implementation of IES is based on [2].
The notation used in the two papers differ slightly, so we have made a few tweaks to make them more similar.

- $A$ is used for the prior ensemble. (It's $X$ in [2])
- $E$ is not divided by $\sqrt{N-1}$ as is done in [2], which means that we do not multiply $E$ by $\sqrt{N-1}$ in the definition of $E$.
- We do not use $EE^T / (N-1)$ to estimate the parameter covariance matrix, because we assume a diagonal observation error covariance matrix $C_{dd}$.
We instead scale matrices used in the analysis step such that $C_{dd}$ becomes the identity matrix.
This is what is known as exact inversion.
- $Y$ is used to hold measured responses, which are predictions made by the dynamical model at points in time and space for which we have observations.

## References

[1] - [Data Assimilation
The Ensemble Kalman Filter](https://link.springer.com/book/10.1007/978-3-642-03711-5)

[2] - [Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching](https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full)
