"""Functions to make particle dataset for training superresolution models."""

import numpy as np

from typing import Tuple


def particle_distribution(
    size: Tuple[int, int, int],
    distribution_fn: callable = None,
    **kwargs,
):
  """Simulates normalized scatterer distribution.

  This function populates a grid with random values which represent the
  normalized fraction of scatters in that voxel.

  `distribution` sets the function used to generate the random values in each
  voxel.

  Values are clipped outside the range [0, 1).

  Args:
    dimensions: Number of pixels.
    distribution_fn: Function to generate random values. Defaults to
    `numpy.random.rand`.
    kwargs: Arguments passed to `distribution_fn`.

  Returns:
    np.ndarray of shape `size`.

  Raises:
    ValueError: If arguments are invalid.
  """
  if len(size) != 3:
    raise ValueError("`size` must be a tuple of 3 `int`."
                     "Got {}".format(size))
  if distribution_fn is None:
    distribution_fn = np.random.random

  distribution = distribution_fn(size=size, **kwargs)

  return np.clip(distribution, 0, 1.)