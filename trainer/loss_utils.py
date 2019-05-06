"""Utilities for defining loss."""
import logging
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops


def quantize_tensor(
    tensor: float,
    quantization_count: int,
    min_val: float,
    max_val: float,
):
  """Quantizes image.

  Args:
    image: tf.Tensor of shape `[batch, height, width]`.
    quantizations: Quantization (bit depth) of intensity values in test_output image.
    Equivalently, number of channels in test_output image.
    separate_channels: If `True` then the different quantizations are treated as
    different classes.

  Returns:
    class: tf.Tensor of shape `[batch, height, width]`
    logits: tf.Tensor of shape `[batch, height, width, quantizations]`
  """
  clipped = tf.clip_by_value(tensor, min_val, max_val)
  quantizations = list(np.linspace(min_val, max_val, quantization_count + 1))[:-1]
  logging.info("using quantizations {}".format(quantizations))
  quantized = math_ops._bucketize(clipped, quantizations)
  # Reset indices to valid indices into `quantization_count` lenth tensor.
  # The bins go from [low_value, high_value) so `0` -> `1` and
  # `1.` -> `quantiztion_count + 1`
  quantized = quantized - 1
  quantized = tf.cast(quantized, tf.int32)
  return quantized, tf.one_hot(quantized, quantization_count)


def _logit_to_class(logit):
  """Class prediction from 1-hot logit.

  This is used to generate the numerical class labels from a one-hot encoded
  logit. For example:

  # logits = [0, 0, 1], [0, 1, 0]]
  output = _logit_to_class(logits)
  # output = [2, 1]

  Args:
    Tensor of shape `batch_dimensions + [class_count]`

  Returns:
    Tensor of shape `batch_dimensions`.
  """
  return tf.cast(tf.argmax(logit, -1), tf.int32)


def inverse_class_weight(
  logits: tf.Tensor,
  epsilon: float = 10.,
):
  """Computes weights that are the inverse of frequency of class in `logits`.

  Args:
    logits: tf.Tensor of shape `[B, H, W, C]` where `c` is the class dimension.
    epsilon: Regularizing term to smooth proportions.

  Returns:
    `tf.Tensor` of same shape as `logits` containing weights.
  """
  real_proportion = (tf.reduce_sum(
    logits,
    axis=[0, 1, 2],
    keepdims=True,
  ) + epsilon) / (tf.cast(tf.size(logits), tf.float32) + epsilon)
  return tf.reduce_sum((1 / real_proportion) * logits, axis=-1)


def downsample_target(target, downsample):
  """Downsample tensor of shape `[batch, height, width]`."""
  target = target[..., tf.newaxis]
  target = tf.keras.layers.AveragePooling2D(downsample, padding='same')(target)
  return target[..., 0]
