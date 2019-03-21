"""Preprocessing functions for US simulation data."""

import math
import logging

import numpy as np
from scipy import signal
import tensorflow as tf


from simulation import psf_utils


def _gaussian_kernel(size: int,
                        std: float,
                        ):
  """Makes 2D gaussian Kernel for convolution."""

  vals = signal.gaussian(size * 2 + 1, std)

  gauss_kernel = np.einsum('i,j->ij',
                           vals,
                           vals)

  return gauss_kernel / np.sum(gauss_kernel)


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
      blur_channels: int=1,
  ):
    """Initializes `imageBlur`.

    Args:
      grid_pitch: grid dimension in physical units.
      sigma_blur: sigma of gaussian kernel in physical units.
      kernel_size: size of kernel in physical units.
      blur_channels: Number of channels in image to be blurred. Each channel is
        treated independently.
    """
    if grid_pitch <= 0:
      raise ValueError("`grid_pitch` must be a positive float.")
    if sigma_blur <= 0:
      raise ValueError("`sigma_blur` must be a positive float.")
    if kernel_size <= 0:
      raise ValueError("`kernel_size` must be a positive float.")

    self._kernel = self._gaussian_kernel(
      grid_pitch, sigma_blur, kernel_size, blur_channels)

  def _gaussian_kernel(
      self, grid_pitch, sigma_blur, kernel_size, blur_channels):
    """Constructs gaussian kernel."""
    # Find `kernel_size` in grid units.
    kernel_size_grid = int(math.ceil(kernel_size / grid_pitch))
    logging.debug("kernel_size_grid set to {}".format(kernel_size_grid))

    # Find `sigma_blur` in grid units.
    sigma_blur_grid = sigma_blur / grid_pitch
    logging.debug("grid sigma set to {}".format(sigma_blur_grid))

    # Make Gaussian Kernel with desired specs.
    gauss_kernel = _gaussian_kernel(kernel_size_grid, sigma_blur_grid)

    # List of `gauss_kernel` of same length as number of channels to be blurred.
    kernel_channels = [gauss_kernel for _ in range(blur_channels)]

    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    return psf_utils.to_filter(kernel_channels, psf_utils._FROM_SAME)

  def blur(self, tensor):
    """Blurs input tensor."""
    return tf.nn.conv2d(
      tensor, self._kernel, strides=[1, 1, 1, 1], padding="SAME")