"""Functions to generate datsets and save."""

import numpy as np
import tensorflow as tf
from simulation import defs

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
    observation_params: defs.ObservationSpec,
):
  """Constructs `tf.train.Example` for scatterer distribution and simulation.

  This function converts a pair of `distribution` and `observation` and
  associated `ObservationSpec` to a tensorflow `Example` which can be written
  and read as a tf protobuf. This example can be decoded by using
  `_parse_example`.

  Args:
    distribution: np.array representing distribution.
    observation: Same as `distribution` but representing observation.
    observation_params: `ObservationSpec` describing observation conditions
      used when generation `observation`.

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

  distribution = tf.make_tensor_proto(distribution)
  observation = tf.make_tensor_proto(observation)

  return tf.train.Example(features=tf.train.Features(feature={
    'distribution': _bytes_feature(distribution.SerializeToString()),
    'observation': _bytes_feature(observation.SerializeToString()),
    'observation_params/angles': _bytes_feature(tf.make_tensor_proto(
      observation_params.angles).SerializeToString()),
    'observation_params/frequencies': _bytes_feature(tf.make_tensor_proto(
      observation_params.frequencies).SerializeToString()),
    'observation_params/modes': _bytes_feature(tf.make_tensor_proto(
      observation_params.modes).SerializeToString()),
    'observation_params/grid_dimension': _float_feature(
      observation_params.grid_dimension),
    'observation_params/transducer_bandwidth': _float_feature(
      observation_params.transducer_bandwidth),
    'observation_params/numerical_aperture': _float_feature(
      observation_params.numerical_aperture),
  }))


def _parse_example(example_serialized: tf.Tensor):
  """Parses a `tf.train.Example` proto containing distribution and observation.

  This function parses an example produced by `_construct_example`.

  Args:
    example_serialized: `tf.Tensor` containing a serialized `Example` protocol
      buffer.

  Returns:
    distribution: See `_construct_example`.
    observation: See `_construct_example`.
    observation_params: See `_construct_example`.
  """
  feature_map = {
    'distribution': tf.FixedLenFeature([], tf.string),
    'observation': tf.FixedLenFeature([], tf.string),
    'observation_params/angles': tf.FixedLenFeature([], tf.string),
    'observation_params/frequencies': tf.FixedLenFeature([], tf.string),
    'observation_params/modes': tf.FixedLenFeature([], tf.string),
    'observation_params/grid_dimension': tf.FixedLenFeature([], tf.float32),
    'observation_params/transducer_bandwidth': tf.FixedLenFeature([],
                                                                  tf.float32),
    'observation_params/numerical_aperture': tf.FixedLenFeature([],
                                                                tf.float32),
  }

  features = tf.parse_single_example(example_serialized, feature_map)

  distribution = tf.io.parse_tensor(features['distribution'], tf.float32)
  observation = tf.io.parse_tensor(features['observation'], tf.float32)

  # Return `ObservationSpec` object.
  observation_params = {
    "angles": tf.io.parse_tensor(features['observation_params/angles'], tf.float32),
    "frequencies": tf.io.parse_tensor(features['observation_params/frequencies'], tf.float32),
    "modes": tf.io.parse_tensor(features['observation_params/modes'], tf.int32),
    "grid_dimension": features['observation_params/grid_dimension'],
    "transducer_bandwidth": features[
      'observation_params/transducer_bandwidth'],
    "numerical_aperture": features['observation_params/numerical_aperture'],
  }

  return distribution, observation, observation_params