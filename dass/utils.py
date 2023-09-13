from collections import namedtuple
from typing import Callable, List

import numpy as np
import numpy.typing as npt

rng = np.random.default_rng()
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

Coordinate = namedtuple("Coordinate", ["x", "y"])


def observations(
    coordinates: List[Coordinate],
    times: npt.NDArray[np.int_],
    field: npt.NDArray[np.float_],
    error: Callable,
) -> pd.DataFrame:
    """Generate synthetic observations by adding noise to true field-values.

    Parameters
    ----------
    error: Callable
        Function that takes a single argument (the true field value) and returns
        a value to be used as the standard deviation of the noise.
    """
    # Create dataframe with observations and necessary meta data.
    observations = []
    for coordinate in coordinates:
        for k in times:
            value = field[k, coordinate.x, coordinate.y]
            sd = error(value)
            observations.append(
                pd.DataFrame(
                    {
                        "k": [k],
                        "x": [coordinate.x],
                        "y": [coordinate.y],
                        "value": [value + rng.normal(loc=0.0, scale=sd)],
                        "sd": [sd],
                    }
                )
            )
    return pd.concat(observations).set_index(["k", "x", "y"], verify_integrity=True)


def colorbar(mappable):
    # https://joseph-long.com/writing/colorbars/
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
