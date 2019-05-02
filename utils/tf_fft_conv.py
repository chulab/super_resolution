"""Compute convolution in tensorflow using fft."""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import manip_ops

from online_simulation import online_dataset_utils
from simulation import response_functions


def convert_to_complex(tensor):
  """Makes imaginary tensor by adding 0's in /hat{i}."""
  return tf.complex(tensor, tf.zeros_like(tensor))


def pad_to_size(tensor, shape):
  tensor_shape = tensor.shape.as_list()
  if len(shape) != len(tensor_shape):
    raise ValueError('len of `shape` must be equal to dimensions of `tensor`')
  pads = [[0, s - tensor_shape[i]] for i, s in enumerate(shape)]
  return tf.pad(tensor, pads, mode="constant")


def fft_conv(tensor_a, tensor_b, mode):
  """Calculates convolution of `tensor_a` and `tensor_b`.

  Size of `tensor_b` must be less than or equal to `tensor_a`.
  """
  tensor_size_a = tensor_a.shape.as_list()
  tensor_size_b = tensor_b.shape.as_list()
  dft_size = [size * 2 - 1 for size in tensor_size_a]
  out_shape = [s1 + s2 - 1 for s1, s2 in zip(tensor_size_a, tensor_size_b)]
  fslice = tuple([slice(sz) for sz in out_shape])
  tensor_a, tensor_b = pad_to_size(tensor_a, dft_size), pad_to_size(tensor_b,
                                                                    dft_size)
  tensor_a, tensor_b = convert_to_complex(tensor_a), convert_to_complex(
    tensor_b)
  tensor_out = tf.real(tf.ifft2d(tf.fft2d(tensor_a) * tf.fft2d(tensor_b)))[
    fslice]
  if mode == 'full':
    return tensor_out
  if mode == 'same':
    return _centered(tensor_out, tensor_size_a)


def fft_correlate(tensor_a, tensor_b, mode):
  """Computes correlation of `tensor_a` and `tensor_b`"""
  return fft_conv(tensor_a, _reverse_and_conj(tensor_b), mode)


def _centered(tensor, newsize):
  """Returns tensor cropped to `size` at center."""
  currshape = tensor.shape.as_list()
  start_ind = [(cur - new) // 2 for cur, new in zip(currshape, newsize)]
  end_ind = [start + new for start, new in zip(start_ind, newsize)]
  myslice = [slice(start, end) for start, end in zip(start_ind, end_ind)]
  return tensor[tuple(myslice)]


def _reverse_and_conj(x):
  """Reverse array `x` in all dimensions and perform the complex conjugate"""
  return tf.reverse(x, axis=list(range(x.shape.ndims)))


def fftshift(x, axes=None, name=None):
  """Shift the zero-frequency component to the center of the spectrum.

   This function swaps half-spaces for all axes listed (defaults to all).
  Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
   @compatibility(numpy)
  Equivalent to numpy.fft.fftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html
  @end_compatibility
   For example:
   ```python
  x = tf.signal.fftshift([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
  x.numpy() # array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
  ```
   Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple`, optional
            Axes over which to shift.  Default is None, which shifts all axes.
    name: An optional name for the operation.
   Returns:
    A `Tensor`, The shifted tensor.
  """
  with tf.name_scope(name, "fftshift") as name:
    x = tf.convert_to_tensor(x)
  if axes is None:
    axes = tuple(range(x.shape.ndims))
    shift = [int(dim // 2) for dim in x.shape]
  elif isinstance(axes, int):
    shift = int(x.shape[axes] // 2)
  else:
    shift = [int((x.shape[ax]) // 2) for ax in axes]

  return manip_ops.roll(x, shift, axes)


def ifftshift(x, axes=None, name=None):
  """The inverse of fftshift.
   Although identical for even-length x,
  the functions differ by one sample for odd-length x.
   @compatibility(numpy)
  Equivalent to numpy.fft.ifftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifftshift.html
  @end_compatibility
   For example:
   ```python
  x = tf.signal.ifftshift([[ 0.,  1.,  2.],[ 3.,  4., -4.],[-3., -2., -1.]])
  x.numpy() # array([[ 4., -4.,  3.],[-2., -1., -3.],[ 1.,  2.,  0.]])
  ```
   Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple` Axes over which to calculate.
          Defaults to None, which shifts all axes.
    name: An optional name for the operation.
   Returns:
    A `Tensor`, The shifted tensor.
  """
  with tf.name_scope(name, "ifftshift") as name:
    x = tf.convert_to_tensor(x)
    if axes is None:
      axes = tuple(range(x.shape.ndims))
      shift = [-int(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
      shift = -int(x.shape[axes] // 2)
    else:
      shift = [-int(x.shape[ax] // 2) for ax in axes]

    return manip_ops.roll(x, shift, axes)


def signal_and_envelope(tensor_a, tensor_b, mode, sampling_rate, frequency, angle, freq_sigma):
  """Calculates analytic signal of convolving `tensor_a` and `tensor_b`.

  Size of `tensor_b` must be less than or equal to `tensor_a`.

  This function is used to calculate the envelope containing speckle resulting
  from convolving a scatterer distribution (`tensor_a`) with a psf (`tensor_b`).

  args:
    tensor_a: `tf.Tensor` of shape `[H, W]`
    tensor_b: `tf.Tensor` of shape `[H2, W2]`
    mode: See documentation for `fft_conv`.
    sampling_rate: Sampling rate in Hz of `tensor_a`.
    frequency: Center frequency to filter.
    angle: Incident angle of impulse in radians.
    frequency_sigma: Approximate sigma in frequency of pulse.

  Returns: `tf.Tensor`
  """
  tensor_size_a = tensor_a.shape.as_list()
  tensor_size_b = tensor_b.shape.as_list()
  dft_size = [size * 2 - 1 for size in tensor_size_a]
  out_shape = [s1 + s2 - 1 for s1, s2 in zip(tensor_size_a, tensor_size_b)]
  fslice = tuple([slice(sz) for sz in out_shape])
  tensor_a, tensor_b = pad_to_size(tensor_a, dft_size), pad_to_size(tensor_b,
                                                                    dft_size)
  tensor_a, tensor_b = convert_to_complex(tensor_a), convert_to_complex(
    tensor_b)

  ft_tensor = tf.fft2d(tensor_a) * tf.fft2d(tensor_b)

  ft_grid_unit = sampling_rate / dft_size[0]
  length_ft = [(s - 1) * ft_grid_unit for s in dft_size]
  ft_coordinates = tf.stack(response_functions.coordinate_grid(
      length_ft, [ft_grid_unit] * len(length_ft), center=True, mode="TF"), -1)

  fftshift_tensor = fftshift(ft_tensor)

  pulse_filter = centered_filter(ft_coordinates, frequency * 2, angle, freq_sigma * 4) * 2
  pulse_filter = convert_to_complex(pulse_filter)
  filtered_tensor = fftshift_tensor * pulse_filter
  filtered_tensor = ifftshift(filtered_tensor)

  envelope_tensor_out = tf.abs(tf.ifft2d(filtered_tensor))[fslice]
  signal_tensor_out = tf.real(tf.ifft2d(ft_tensor))[fslice]

  if mode == 'full':
    return envelope_tensor_out, signal_tensor_out
  if mode == 'same':
    return (_centered(envelope_tensor_out, tensor_size_a),
            _centered(signal_tensor_out, tensor_size_a))


def centered_filter(coordinates, radius, theta, size):
  """Creates filter that is a hard circle located at `(radius, theta)`."""
  center = tf.stack([tf.cos(theta) * radius, -1 * tf.sin(theta) * radius])
  return online_dataset_utils._circle(coordinates, center, size)
