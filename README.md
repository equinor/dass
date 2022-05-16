# Data Assimilation

Implementations of Ensemble Smoother as given in [1] and Iterative Ensemble Smoother (IES) as given in [2],
see [dass/analysis.py](dass/analysis.py).

The only forward model currently available is a 2D stochastic heat equation, see [dass/pde.py](dass/pde.py).

For complete examples see

- [notebooks/Smoothers.ipynb](notebooks/Smoothers.ipynb) for pure parameter estimation using smoothers.
- [notebooks/EnKF.ipynb](notebooks/EnKF.ipynb) for sequential data assimilation or filtering.

## References

[1] - [Data Assimilation
The Ensemble Kalman Filter](https://link.springer.com/book/10.1007/978-3-642-03711-5)

[2] - [Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching](https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full)
