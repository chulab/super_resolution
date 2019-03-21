"""Preprocessing functions for US simulation data."""

import math
import logging

import tensorflow as tf


def _gaussian_kernel(
    size: int,
    mean: float,
    std: float,
  ):
  """Makes 2D gaussian Kernel for convolution."""

  d = tf.distributions.Normal(mean, std)

  vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

  gauss_kernel = tf.einsum('i,j->ij',
                           vals,
                           vals)

  return gauss_kernel / tf.reduce_sum(gauss_kernel)


class imageBlur(object):
  """Blurs images using a gaussian kernel.

  This function only works on images with a single channel.

  Example usage:
    blur = imageBlur( ... )

    dataset = ...

    dataset_blur = dataset.map()
  """

  def __init__(
      self,
      grid_pitch: float,
      sigma_blur: float,
      kernel_size: float,
  ):
    """Initializes `imageBlur`.

    Args:
      grid_pitch: grid dimension in physical units.
      sigma_blur: sigma of gaussian kernel in physical units.
      kernel_size: size of kernel in physical units.
    """
    if grid_pitch <= 0:
      raise ValueError("`grid_pitch` must be a positive float.")
    if sigma_blur <= 0:
      raise ValueError("`sigma_blur` must be a positive float.")
    if kernel_size <= 0:
      raise ValueError("`kernel_size` must be a positive float.")

    self._kernel = self._gaussian_kernel(grid_pitch, sigma_blur, kernel_size)

  def _gaussian_kernel(self, grid_pitch, sigma_blur, kernel_size):
    """Constructs gaussian kernel."""
    # Find `kernel_size` in grid units.
    kernel_size_grid = int(math.ceil(kernel_size / grid_pitch))
    logging.debug("kernel_size_grid set to {}".format(kernel_size_grid))

    # Find `sigma_blur` in grid units.
    sigma_blur_grid = sigma_blur / grid_pitch
    logging.debug("grid sigma set to {}".format(sigma_blur_grid))

    # Make Gaussian Kernel with desired specs.
    gauss_kernel = _gaussian_kernel(kernel_size_grid, 0., sigma_blur_grid)

    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    return gauss_kernel[:, :, tf.newaxis, tf.newaxis]

  def blur(self, tensor):
    """Blurs input tensor."""
    return tf.nn.conv2d(
      tensor, self._kernel, strides=[1, 1, 1, 1], padding="SAME")