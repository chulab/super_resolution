"""Contains functions to perform complex convolutions."""

import tensorflow as tf


def convolve_complex_1d(
    tensor: tf.Tensor,
    filter: tf.Tensor,
    stride: int = 1,
    padding: str = "VALID",
):
  """Performs a convolution in 1D with a complex kernel.

  Convolution is performed along last dimension of `tensor`.

  Args:
    tensor: `tf.Tensor` with shape
      `batch_dimensions + [length, in_channels]`.
    filter: `tf.Tensor` with shape `[filter_length, in_channels, out_channels]`.
    padding: Type of padding. May be `VALID` or `SAME`.
    stride: Stride size to use between convolutions.

  Returns:
    `tf.Tensor` resulting from convolving `padding` with `tensor`. Has shape
    `batch_dimensions + convolution_dimensions + [out_channels]`

  Raises:
    ValueError: If `tensor` and `filter` do not have compatible dtype or if
      `filter` has incorrect shape.
  """
  if tensor.dtype != filter.dtype:
    raise ValueError("`tensor` and `filter` must have same dtype got `{}`"
                     "".format([tensor.dtype, filter.dtype]))
  filter.shape.assert_is_compatible_with([None, None, None])

  filter_length = filter.shape[0]

  if padding == "VALID":
    pass
  elif padding == "SAME":
    if (tensor.shape[-2] % stride == 0):
      pad_along_height = max(filter_length - stride, 0)
    else:
      pad_along_height = max(filter_length - (tensor.shape[-2] % stride), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    tensor = tf.pad(tensor, [[0, 0]] * tensor.shape[:-2].ndims + [
      [pad_top, pad_bottom]] + [[0, 0]])
  else:
    raise ValueError("`padding` must be one of `VALID` or `SAME` but got `{}`"
                     "".format(padding))

  # Slice `tensor`.
  tensor_slices = [tensor[..., start_slice:start_slice + filter_length, :] for
                   start_slice in
                   range(0, tensor.shape[-2] - filter_length + 1,
                         stride)]

  # Add batch dimensions to filter.
  filters = tf.reshape(filter,
                       [1] * tensor.shape[:-1].ndims + filter.shape.as_list())

  # Stack slices. `tensor` now has shape
  # `batch_dimensions + [output_dimension, filter_length, in_channels]`.
  tensor = tf.stack(tensor_slices, -3)

  # Expand last dimension of `tensor` to account for `filter_count`. `tensor`
  # now has shape
  # `batch_dimensions + [output_dimension, filter_length, in_channels, 1]`.
  tensor = tensor[..., tf.newaxis]

  # Mupltiply tensor and filters.
  tensor = tensor * filters

  # Sum along `filter_length` and `in_channels` dimensions.
  return tf.reduce_sum(tensor, [-3, -2])

