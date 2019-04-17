"""Compute convolution in tensorflow using fft."""
import tensorflow as tf


def convert_to_complex(tensor):
  return tf.complex(tensor, tf.zeros_like(tensor))


def pad_to_size(tensor, shape):
  tensor_shape=tensor.shape.as_list()
  if len(shape) != len(tensor_shape):
    raise ValueError('len of `shape` must be equal to dimensions of `tensor`')
  pads = [[0, s - tensor_shape[i]]  for i, s in enumerate(shape)]
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
