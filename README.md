# Data Assimilation

`dass` is tool for learning about data assimilation / history matching created by the developers of [ERT](https://github.com/equinor/ert).
It is inspired by [DAPPER](https://github.com/nansencenter/DAPPER) and [HistoryMatching](https://github.com/patnr/HistoryMatching).

It includes implementations of Ensemble Smoother (ES) as given in [1], see [dass/analysis.py](dass/analysis.py).
The implementation of ES can easily be extended to the Ensemble Smoother with Multiple Data Assimilation (ES-MDA) as described in [2].

For notebooks with examples and tutorials see the `notebooks/` folder.

## Installation

```bash
git clone https://github.com/equinor/dass.git
cd dass
# Install dass and dependencies using uv.
uv sync
# Start jupyter notebook
uv run jupyter notebook
# To make sure everything works, run on the of the notebooks in the notebooks/ folder.
```

## References

[1] - [Data Assimilation
The Ensemble Kalman Filter](https://link.springer.com/book/10.1007/978-3-642-03711-5)

[2] - [Ensemble smoother with multiple data assimilation](https://www.sciencedirect.com/science/article/pii/S0098300412000994)
