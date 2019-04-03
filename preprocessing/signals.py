"""Signal processing functions."""

import tensorflow as tf

import numpy as np

def swapaxes(
    tensor,
    axis_1,
    axis_2
):
  axes = list(range(tensor.shape.ndims))
  axes[axis_1], axes[axis_2] = axes[axis_2], axes[axis_1]

  print("swap combo {}".format(axes))
  return tf.transpose(tensor, axes)


def hilbert(
    tensor: tf.Tensor,
    axis: int,
):
  """Computes the Hilbert transform of an input tensor along `axis`.

  See documentation for `scipy.signal.hilbert`.

  Args:
    tensor: Tf.Tensor on which to apply hilbert transform.
    axis: Axis of `tensor` on which to apply transform.

  Returns:
    `tf.Tensor` of same shape as `tensor` and complex dtype.
  """
  tensor = tf.cast(tensor, tf.complex64)
  tensor_shape = tensor.shape.as_list()
  N = tensor_shape[axis]
  h = np.zeros(N)

  last_axis = len(tensor_shape) - 1

  if axis != last_axis:
    print("swapping axes.")
    tensor = swapaxes(tensor, axis, last_axis)

  Xf = tf.fft(tensor)

  if N % 2 == 0:
    h[0] = h[N // 2] = 1
    h[1:N // 2] = 2
  else:
    h[0] = 1
    h[1:(N + 1) // 2] = 2

  if len(tensor_shape) > 1:
    ind = [np.newaxis] * len(tensor_shape)
    ind[-1] = slice(None)
    h = h[tuple(ind)]

  tensor = tf.ifft(Xf * h)

  if axis != last_axis:
    print("swapping axes.")
    tensor = swapaxes(tensor, axis, last_axis)

  return tensor