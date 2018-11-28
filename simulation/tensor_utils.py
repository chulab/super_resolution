"""Tensor Utilities."""

import tensorflow as tf
import numpy as np

from typing import List


def rotate_tensor(
    tensor: tf.Tensor,
    angles: tf.Tensor,
    rotation_axis: int,
) -> tf.Tensor:
  """Rotates a `tf.Tensor` along a given batch dimension.

  This function broadcasts a 2D rotation on a `tf.Tensor` with arbitrary
  batch dimensions. The rotation is applied to each image or batch of images
  where the batching dimension is set by `rotation_axis`. The rotation is
  given in radians applied in a counterclockwise manner.

  Explicitly:
    tensor = ... # Tensor of shape `[B_1, B_2, ... , H, W, C]`
    rotate_tensor(tensor, [R_1, R_2 ... ], rotation_axis = 1)
    # Returns tensor where:
    #   [:, 0, ..., H, W, C]  `H` and `W` dimensions are rotated by `R_1`.
    #   [:, 1, ..., H, W, C]  `H` and `W` dimensions are rotated by `R_2`.
    #   Etc.

  This function is used when there is a separate the rotation axis along with
  one or more batch dimensions.

  Args:
    tensor: `tf.Tensor` of shape
      `batch_dimensions + [height, width, channels]`.
    angles: `tf.Tensor` of shape `[rotation_axis_dimension]` describing the
      rotations to be applied to each batch along the rotation axis in radians.
    rotation_axis: Int indicating rotation axis.

  Returns:
    `tf.Tensor` of same shape as `tensor` with rotation applied.

  Raises:
    ValueError: If input parameters are invalid.
  """
  tensor_shape = tensor.shape.as_list()
  if len(tensor_shape) < 4:
    raise ValueError(
      "`tensor` must have rank at least 4, got {}.".format(len(tensor_shape)))
  if rotation_axis < 0:
    raise ValueError(
      "`rotation_axis` must be positive got {}".format(rotation_axis))
  if len(tensor_shape) - rotation_axis < 4:
    raise ValueError(
      "`rotation_axis` must be a batch dimension (last 3 axes, are reserved "
      "for `[height, width, channel]`)."
    )
  if angles.shape.ndims != 1:
    raise ValueError(
      "`angles` must be a 1D list. Got {}.".format(angles.shape))
  if angles.shape.as_list()[0] != tensor_shape[rotation_axis]:
    raise ValueError("`angles` length must equal `rotation_axis` shape."
                     "Got {} and {}.".format(
      angles.shape.as_list(), tensor_shape[rotation_axis]))

  axes = [(axis, shape) for axis, shape in enumerate(tensor_shape)]

  # Keep `rotation_axis` as batch dimension.
  transpose = [axes.pop(rotation_axis)]

  # Append last 3 dimensions [`height`, `width`, `channels`].
  transpose += [axes.pop(axis) for axis in range(-3, 0)]

  # Finally append other dimensions.
  transpose += axes

  # Transpose.
  tensor = tf.transpose(tensor, [axis for axis, _ in transpose])

  # Reshape to put all of the batch dimensions into channels.
  tensor = tf.reshape(tensor, [shape for _, shape in transpose[:3]] + [-1])

  # Perform rotation.
  tensor = tf.contrib.image.rotate(tensor, angles, "BILINEAR")

  # Retrieve dimensions compressed into `channels`.
  tensor = tf.reshape(tensor, [shape for _, shape in transpose])

  inverse_transpose = _reverse_transpose_sequence(
    [axis for axis, _ in transpose])

  # Transpose to return axes to original positions.
  return tf.transpose(tensor, inverse_transpose)


def combine_batch_into_channels(
    tensor: tf.Tensor,
    exclude_dimension: int,
):
  """Merges excess batch dimensions into channel dimension.

  Reshapes `tensor` and merges all batch dimensions except `exclude_dimension`
  into the channel dimension.

  Args:
    tensor: `tf.Tensor` of shape `batch_dimensions + [height, width, channels]`.
    exclude_dimension: Int describing batch dimension to preserve.

  Returns:
    `tf.Tensor` of shape
      `[preserved_batch_dimension, width, height, new_channels]` where the size
      of `new_channels` depends on the number and size of `batch_dimensions` in
      `tensor`. Also returns the transposition sequence required to recover
      the original axis order if
  """
  tensor_shape = tensor.shape.as_list()
  if exclude_dimension < 0:
    raise ValueError(
      "`exclude_dimension` must be positive got {}".format(exclude_dimension))
  if len(tensor_shape) - exclude_dimension < 4:
    raise ValueError(
      "`exclude_dimension` must be a batch dimension (last 3 axes, are reserved "
      "for `[height, width, channel]`)."
    )

  axes = [(axis, shape) for axis, shape in enumerate(tensor_shape)]

  # Keep `rotation_axis` as batch dimension.
  transpose = [axes.pop(exclude_dimension)]

  # Append last 3 dimensions [`height`, `width`, `channels`].
  transpose += [axes.pop(axis) for axis in range(-3, 0)]

  # Finally append other dimensions.
  transpose += axes

  # Transpose.
  tensor = tf.transpose(tensor, [axis for axis, _ in transpose])

  # Reshape to put all of the batch dimensions into channels.
  return tf.reshape(tensor, [shape for _, shape in transpose[:3]] + [-1])


def _reverse_transpose_sequence(transpose_sequence: List):
  """Generates sequence to reverse a transpose given the original sequence."""
  return [new_position for new_position, _ in
          sorted(enumerate(transpose_sequence), key=lambda x: x[1])]
