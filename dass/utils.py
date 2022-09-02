from collections import namedtuple
from typing import List

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
) -> pd.DataFrame:
    """Generate synthetic observations by adding noise to true field-values."""
    d = pd.DataFrame(
        {
            "k": pd.Series(dtype=int),
            "x": pd.Series(dtype=int),
            "y": pd.Series(dtype=int),
            "value": pd.Series(dtype=float),
            "sd": pd.Series(dtype=float),
        }
    )

    # Create dataframe with observations and necessary meta data.
    for coordinate in coordinates:
        for k in times:
            # The reason for u[k, y, x] instead of the perhaps more natural u[k, x, y],
            # is due to a convention followed by matplotlib's `pcolormesh`
            # See documentation for details.
            value = field[k, coordinate.y, coordinate.x]
            sd = max(0.10 * value, 1)
            _df = pd.DataFrame(
                {
                    "k": [k],
                    "x": [coordinate.x],
                    "y": [coordinate.y],
                    "value": [value + rng.normal(loc=0.0, scale=sd)],
                    "sd": [sd],
                }
            )
            d = pd.concat([d, _df])
    d = d.set_index(["k", "x", "y"], verify_integrity=True)

    return d


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
