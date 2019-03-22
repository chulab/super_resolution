"""Defines Parser Class which is used to parse and preprocess simulation data."""

import logging
from typing import Tuple

import tensorflow as tf

from simulation import defs
from simulation import tensor_utils
from preprocessing import preprocess
from training_data import record_utils


class Parser(object):
  """Parses tf.Proto examples and applies preprocessing."""

  def __init__(
      self,
      observation_spec: defs.ObservationSpec,
      reverse_rotation: bool,
      distribution_blur_sigma: float,
      observation_blur_sigma: float,
      distribution_downsample_size: Tuple[int],
      observation_downsample_size: Tuple[int],
      example_size: Tuple[int]
  ):
    """Initializes `Parser`."""
    assert isinstance(observation_spec, defs.ObservationSpec)
    self._observation_spec = observation_spec
    self._example_size = example_size

    self._reverse_rotation = reverse_rotation

    self._distribution_blur = preprocess.imageBlur(
      grid_pitch=observation_spec.grid_dimension,
      sigma_blur=distribution_blur_sigma,
      kernel_size=2 * distribution_blur_sigma,
      blur_channels=1,
    )
    self._observation_blur = preprocess.imageBlur(
      grid_pitch=observation_spec.grid_dimension,
      sigma_blur=observation_blur_sigma,
      kernel_size=2 * observation_blur_sigma,
      blur_channels=len(observation_spec.psf_descriptions) * len(
        observation_spec.angles)
    )
    self._distribution_downsample_size = distribution_downsample_size
    self._observation_downsample_size = observation_downsample_size

  @property
  def distribution_blur(self):
    return self._distribution_blur

  @property
  def observation_blur(self):
    return self._observation_blur

  @property
  def observation_downsample_size(self):
    return self._observation_downsample_size

  @property
  def distribution_downsample_size(self):
    return self._distribution_downsample_size

  def set_shape(self, distribution, observation):
    height, width = (self._example_size[0], self._example_size[1])
    observation.set_shape(
      [len(self._observation_spec.angles), height, width,
       len(self._observation_spec.psf_descriptions)])
    distribution.set_shape([height, width])

  def parse(self, tensor_serialized: tf.Tensor):
    return self._parse(tensor_serialized)

  def _parse(self, tensor_serialized):
    # `distribution` has shape `[Height, Width]`.
    # `observation` has shape `[angle_count, height, width, psfs]`.
    distribution, observation = record_utils._parse_example(tensor_serialized)

    # Add shapes.
    self.set_shape(distribution, observation)

    distribution = distribution[tf.newaxis, ..., tf.newaxis]
    observation, transpose, reverse_transpose = \
      tensor_utils.combine_batch_into_channels(observation[tf.newaxis], 0)

    # Apply blur to distribution.
    distribution = self._distribution_blur.blur(distribution)
    observation = self._observation_blur.blur(observation)

    # Observation has shape `[angle_count, height, width, psfs]`.
    observation = tf.transpose(
      tf.reshape(observation, [shape for _, shape in transpose]),
      reverse_transpose)[0]

    # Apply downsample.
    distribution = tf.image.resize_images(
      distribution, self._distribution_downsample_size)

    observation = tf.image.resize_images(
      observation, self._observation_downsample_size)

    # Optionally rotate back.
    if self._reverse_rotation:
      observation = tensor_utils.rotate_tensor(
        observation,
        tf.convert_to_tensor([-1 * angle for angle in self._observation_spec.angles]),
        0
      )

    logging.debug("observation {}".format(observation))
    logging.debug("distribution {}".format(distribution))

    return distribution[0, ..., 0], observation