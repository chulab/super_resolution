"""Utilities for defining loss."""

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops


def quantize_tensor(
    tensor: float,
    bit_depth: int,
    min_val: float,
    max_val: float,
    separate_channels: bool=True,
):
  """Quantizes image.

  Args:
    image: tf.Tensor of shape `[batch, height, width]`.
    quantizations: Quantization (bit depth) of intensity values in output image.
    Equivalently, number of channels in output image.
    separate_channels: If `True` then the different quantizations are treated as
    different classes.

  Returns:
    segmented_image: tf.Tensor of shape `[batch, height, width, quantizations]`
  """
  quantization_count = 2 ** bit_depth
  clipped = tf.clip_by_value(tensor, min_val, max_val)
  scaled = clipped / max_val
  quantizations = list(np.linspace(-1.e-5, 1., quantization_count))
  quantized = math_ops._bucketize(scaled, quantizations)
  # Reset indices to valid indices into `quantization_count` lenth tensor.
  # The bins go from [low_value, high_value) so `0` -> `1` and
  # `1.` -> `quantiztion_count + 1`
  quantized = quantized - 1
  if not separate_channels:
    return quantized
  else:
    return tf.one_hot(quantized, quantization_count)