"""Utilities for constructing psf filters."""

import numpy as np

from typing import List

_FROM_SAME = "FROM_SAME"
_FROM_SINGLE = "FROM_SINGLE"


def to_filter(
    psfs: List[np.ndarray],
    mode: str,
) -> np.ndarray:
  """Constructs a single filter from a list of psfs.
  
  This function builds a filter of shape `[length, channels_in, channels_out]`.

  In mode `FROM_SINGLE` this constructs a filter with one input channel and
  `len(psfs)` output channels.

  In mode `FROM_SAME`, the number of input and output channels are the same and
  each psf represents a map from one channel to the same channel in the output.
   In other words,
    F_{l, i, j} = {
        if i == j: psf[l]
        if i != j: [0]
      }

  This is useful when applying multiple filters (for example, when trying
  to simulate multiple observation frequencies) in an observation. The output
  of

  Args:
    psfs: List of 1D arrays representing the PSF of an imaging device.
    mode: String representing type of filter. Must be one of `FROM_SINGLE`,
    `FROM_SAME`.

  Returns:
    Array of shape `[length, channels_in, channels_out]`.

  Raises:
    ValueError: If all psf's are not the same length. Or mode arg is invalid.
    """
  if any(len(psf.shape) > 1 for psf in psfs):
    raise ValueError("All PSF's must be 1D, got {}.".format(
      [psf.shape for psf in psfs]))
  if any(psf.shape != psfs[0].shape for psf in psfs):
    raise ValueError("All PSF's must have same shape, got {}.".format(
      [psf.shape for psf in psfs]
    ))

  # Stack filters and add `in_channel` dimension.
  filter = np.stack(psfs, -1)[..., np.newaxis, :]

  if mode == _FROM_SINGLE:
    return filter
  if mode == _FROM_SAME:
    # Tile `filter` so `in_channels` has same dimension as `out_channels`.
    filter = np.tile(filter, [1, len(psfs), 1])
    return filter * np.eye(len(psfs))[np.newaxis, ...]
  else:
    raise ValueError("`mode` must be one of {} but got `{}`".format(
      [_FROM_SAME, _FROM_SINGLE], mode))
