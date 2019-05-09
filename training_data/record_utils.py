"""Functions to generate datsets and save."""

import numpy as np
import tensorflow as tf
from simulation import defs

from typing import List


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return _float_list_feature([value])


def _float_list_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _construct_example(
    distribution: np.ndarray,
    observation: np.ndarray,
):
  """Constructs `tf.train.Example` for scatterer distribution and simulation.

  This function converts a pair of `distribution` and `observation` and
  associated `ObservationSpec` to a tensorflow `Example` which can be written
  and read as a tf protobuf. This example can be decoded by using
  `_parse_example`.

  Args:
    distribution: np.array representing distribution.
    observation: Same as `distribution` but representing observation.
  Returns:
    `tf.train.Example`.

  Raises:
    ValueError: if `distribution` or `observation` have bad Dtype.
  """
  if distribution.dtype != np.float32:
    raise ValueError("`distribution` must have dtype `float32` got {}"
                     "".format(distribution.dtype))
  if observation.dtype != np.float32:
    raise ValueError("`observation` must have dtype `float32` got {}"
                     "".format(observation.dtype))

  distribution_proto = tf.make_tensor_proto(distribution)
  observation_proto = tf.make_tensor_proto(observation)

  return tf.train.Example(features=tf.train.Features(feature={
    'distribution': _bytes_feature(distribution_proto.SerializeToString()),
    'observation': _bytes_feature(observation_proto.SerializeToString()),
    'info/distribution/height': _int64_feature(distribution.shape[0]),
    'info/distribution/width': _int64_feature(distribution.shape[1]),
    'info/observation/height': _int64_feature(observation.shape[1]),
    'info/observation/width': _int64_feature(observation.shape[2]),
  }))


def _parse_example(
    example_serialized: tf.Tensor,
    includes_shape=False
):
  """Parses a `tf.train.Example` proto containing distribution and observation.

  This function parses an example produced by `_construct_example`.

  Args:
    example_serialized: `tf.Tensor` containing a serialized `Example` protocol
      buffer.
    includes_shape: Bool. Determines if there is `shape` information stored
    with tensors. For backward comatibility with an earlier version of
    `_construct_example` which did not store shape.

  Returns:
    distribution: See `_construct_example`.
    observation: See `_construct_example`.
  """
  feature_map = {
    'distribution': tf.FixedLenFeature([], tf.string),
    'observation': tf.FixedLenFeature([], tf.string)
  }
  if includes_shape:
    feature_map.update({
        'info/distribution/height': tf.FixedLenFeature([], tf.int64),
        'info/distribution/width': tf.FixedLenFeature([], tf.int64),
        'info/observation/height': tf.FixedLenFeature([], tf.int64),
        'info/observation/width': tf.FixedLenFeature([], tf.int64)
      })

  features = tf.parse_single_example(example_serialized, feature_map)

  distribution = tf.io.parse_tensor(features['distribution'], tf.float32)
  observation = tf.io.parse_tensor(features['observation'], tf.float32)

  if includes_shape:
    distribution_shape = (features['info/distribution/height'],
                          features['info/distribution/width'])
    observation_shape = (features['info/observation/height'],
                          features['info/observation/width'])
    return distribution, observation, distribution_shape, observation_shape

  return distribution, observation
