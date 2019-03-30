"""Preprocessing functions for US simulation data."""

import math
import logging

import numpy as np
from scipy import signal
import tensorflow as tf

from simulation import psf_utils
from simulation import tensor_utils
from training_data import record_utils

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


def blur(
    observation_spec,
    distribution_blur_sigma,
    observation_blur_sigma
):
  _distribution_blur = imageBlur(
    grid_pitch=observation_spec.grid_dimension,
    sigma_blur=distribution_blur_sigma,
    kernel_size=2 * distribution_blur_sigma,
    blur_channels=1,
  )
  logging.debug("Initialized `_distribution_blur`.")

  _observation_blur = imageBlur(
    grid_pitch=observation_spec.grid_dimension,
    sigma_blur=observation_blur_sigma,
    kernel_size=2 * observation_blur_sigma,
    blur_channels=len(observation_spec.psf_descriptions) * len(
      observation_spec.angles)
  )
  logging.debug("Initialized `_observation_blur`.")

  def blur_(distribution, observation):
    distribution = distribution[tf.newaxis, ..., tf.newaxis]
    observation, transpose, reverse_transpose = \
      tensor_utils.combine_batch_into_channels(observation[tf.newaxis], 0)


    # Apply blur to distribution.
    distribution = _distribution_blur.blur(distribution)
    observation = _observation_blur.blur(observation)

    # Observation has shape `[angle_count, height, width, psfs]`.
    observation = tf.transpose(
      tf.reshape(observation, [shape for _, shape in transpose]),
      reverse_transpose)[0]

    print("OBSERVATION SHAPE {}".format(observation))

    return distribution[0, ..., 0], observation

  return blur_


def parse():
  return record_utils._parse_example


def set_shape(distribution_shape, observation_shape):

  def set_shape(distribution, observation):
    distribution.set_shape(distribution_shape)
    observation.set_shape(observation_shape)
    return distribution, observation

  return set_shape


def check_for_nan(distribution, observation):
  """Returns array of 0's if any value in array is a Nan."""
  return (tf.cond(tf.math.reduce_any(tf.is_nan(distribution)), lambda: tf.zeros_like(distribution), lambda: distribution),
         tf.cond(tf.math.reduce_any(tf.is_nan(observation)), lambda: tf.zeros_like(observation), lambda: observation))


def rotate_observation(angles):

  def rotate_observation_(distribution, observation):
    observation = tensor_utils.rotate_tensor(
      observation,
      tf.convert_to_tensor(
        [-1 * angle for angle in angles]),
      0
    )

    observation = tensor_utils.combine_batch_into_channels(
      observation[tf.newaxis], 0)[0][0]
    return distribution, observation

  return rotate_observation_


def downsample(distribution_downsample_size, observation_downsample_size):

  def downsample_(distribution, observation):
    distribution = tf.image.resize_images(
      distribution, distribution_downsample_size)

    observation = tf.image.resize_images(
      observation, observation_downsample_size)
    return distribution, observation

  return downsample_

def swap(distribution, observation):
  return observation, distribution