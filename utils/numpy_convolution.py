"""Fast numpy convolution operations."""

import math

import numpy as np

from scipy import signal

_VALID = "VALID"
_SAME = "SAME"


def convolve_2d(
    tensor: np.ndarray,
    filter: np.ndarray,
    padding: str,
) -> np.ndarray:
  """Computes 2D convolution in style of `tf.nn.convolve2d`.

  Args:
    tensor: Array with shape `[batch, height, width, channel]`.
    filter: Array with shape
      `[filter_height, filter_width, in_channel, out_channels]`.
    padding: One of `SAME`, `VALID`.

  Returns:
    Result of convolving `tensor` with `filter`.

  Raises:
      `ValueError` if `filter` or `tensor` have incorrect shape.
  """
  if tensor.dtype != filter.dtype:
    raise ValueError("`tensor` and `filter` must have same dtype got `{}`"
                     "".format([tensor.dtype, filter.dtype]))
  if len(filter.shape) != 4:
    raise ValueError("`filter` must have shape "
                     "[height, width, channel_in, channel_out]`")
  if filter.shape[-2] != tensor.shape[-1]:
    raise ValueError("`filter` and `tensor` must have same number of "
                     "`channel_in`")

  filter_lengths = filter.shape[:2]

  # Pad spatial axes given desired output.
  spatial_pads = [
    _pads(axis_length, padding) for axis_length in filter_lengths]

  # We do not want to pad either `batch` or `filter` axes.
  pads = [[0, 0]] + spatial_pads + [[0, 0]]

  tensor = np.pad(tensor, pads, mode="constant")

  # Add `batch` dimension to `filter`.
  filter = filter[np.newaxis, ...]

  # We iterate over `channel_out` axis to make a stack with elements corresponding
  # to individual `channel_out` filters.
  conv_list = [
    signal.correlate(tensor, filter_slice, mode="valid", method="auto")
    for filter_slice
    in [filter[..., out_channel] for out_channel in range(filter.shape[-1])]
  ]

  return np.concatenate(conv_list, -1)


# TODO(noah): implement other stride lengths
def _pads(
    filter_length: int,
    mode:str,
):
  """Computes zero-padding given filter length and mode."""
  if mode==_VALID:
    return [0, 0]
  if mode==_SAME:
    total_pad = float(filter_length - 1)
    pad_top = math.floor(total_pad / 2.)
    pad_bottom = math.ceil(total_pad / 2.)
    return [pad_top, pad_bottom]

