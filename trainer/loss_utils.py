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
  with tf.name_scope("inverse_propotional_weight"):
    real_proportion = (tf.reduce_sum(
      logits,
      axis=[0, 1, 2],
      keepdims=True,
    ) + epsilon) / (tf.cast(tf.size(logits), tf.float32) + epsilon)
    return tf.reduce_sum((1 / real_proportion) * logits, axis=-1)

def inverse_class_weight_laplace(
  logits: tf.Tensor,
):
  """Computes weights that are the inverse of frequency of class in `logits`
  with Laplace smoothing.

  Args:
    logits: tf.Tensor of shape `[B, H, W, C]` where `c` is the class dimension.

  Returns:
    `tf.Tensor` of same shape as `logits` containing weights.
  """
  with tf.name_scope("inverse_propotional_weight"):
    channels = int(logits.shape[-1])
    real_proportion = (tf.reduce_sum(
      logits,
      axis=[0, 1, 2],
      keepdims=True,
    ) + 1) / (tf.cast(tf.size(logits), tf.float32) + channels)
    return tf.reduce_sum((1 / real_proportion) * logits, axis=-1)


def downsample_target(target, downsample):
  """Downsample tensor of shape `[batch, height, width]`."""
  target = target[..., tf.newaxis]
  target = tf.keras.layers.AveragePooling2D(downsample, padding='same')(target)
  return target[..., 0]

def bets_and_rewards_weight(distributions_quantized, distribution_class,
  prediction_class, params):
  '''Computes weights that are a product of a function of class frequencies in
  distributions (referred to as rewards) and a function of class frequencies in
  predictions (referred to as bets). Supported functions are None, 1/n, -logn,
  1/sqrtn where n refers to class frequency.

  It has been found empirically that setting bets=-logn and rewards=1/sqrtn
  works well in general.

  Args:
    distributions_quantized: tf.Tensor of shape `[B, H, W, C]` where `C` is
      the class dimension.
    distribution_class: tf.Tensor of shape `[B, H, W]` with true class label.
    prediction_class: tf.Tensor of shape `[B, H, W]` with predicted class label.
    params: HParams with
     `bit_depth`: int such that 2 ** bit_depth = num_classes.
     `scale_steps`: Number of training steps to apply bets-rewards scaling.
     `diff_scale`: if `abs` or `square`, scales proportional weights by absolute
       and squared difference in prediction and distribution classes
       respectively.

  Returns:
    `tf.Tensor` of same shape as `distributions_quantized` containing weights.
  '''
  proportion = (tf.reduce_sum(
      distributions_quantized,
      axis=[0, 1, 2],
      keepdims=True,
      ) + 1) / (tf.reduce_sum(distributions_quantized)
        + 2 ** params.bit_depth)
  inv_proportion = 1 / proportion

  ones_like = tf.cast(tf.ones_like(prediction_class), tf.float32)

  def bets_and_rewards_fn(params):
    if params.rewards == "1/n":
      rewards = tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1)
    elif params.rewards == "-logn":
      rewards = - 1 * tf.reduce_sum(tf.math.log(proportion) *
        distributions_quantized, axis=-1)
    elif params.rewards == "1/sqrtn":
      rewards = tf.math.sqrt(tf.reduce_sum(inv_proportion *
        distributions_quantized, axis=-1))
    else:
      rewards = ones_like

    one_hot_predictions = tf.one_hot(prediction_class, 2 ** params.bit_depth)

    if params.bets == "1/n":
      bets= tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1)
    elif params.bets== "-logn":
      bets = - 1 * tf.reduce_sum(tf.math.log(proportion) * one_hot_predictions,
        axis=-1)
    elif params.bets == "1/sqrtn":
      bets = tf.math.sqrt(tf.reduce_sum(inv_proportion *
        one_hot_predictions, axis=-1))
    else:
      bets = ones_like

    return bets, rewards

  less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
  bets, rewards = tf.cond(less_equal, lambda: bets_and_rewards_fn(params),
    lambda: (ones_like, ones_like))

  proportional_weights = bets * rewards
  EPSILON = 0.01
  if params.diff_scale == "abs":
    proportional_weights *= tf.cast(tf.math.abs(distribution_class -
      prediction_class), tf.float32) + EPSILON
  elif params.diff_scale == "square":
    proportional_weights *= tf.cast(tf.math.square(distribution_class -
      prediction_class), tf.float32) + EPSILON

  # normalize each batch
  norm = tf.math.reduce_sum(proportional_weights, axis=[-2, -1], keepdims=True)
  proportional_weights = tf.div_no_nan(proportional_weights, norm) \
    * int(distribution_class.shape[1]) * int(distribution_class.shape[2])

  return proportional_weights
