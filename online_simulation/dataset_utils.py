"""Tfrecords ecample utilities."""
import numpy as np

import sys
import os
# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from training_data import record_utils


def convert_to_example(
  probability_distribution: np.ndarray,
  scatterer_distribution: np.ndarray,
):
  """Construct `tf.train.Example` for scatterer probability and distribution.

  This function converts a pair of `probability` and `distribution` to a
  tensorflow `Example` which can be written and read as a tf protobuf. This
  example can be decoded by using `_parse_example`.

  Args:
    distribution: np.array representing distribution.
    observation: Same as `distribution` but representing observation.
  Returns:
    `tf.train.Example`.

  Raises:
    ValueError: if `distribution` or `observation` have bad Dtype.
  """
  if probability_distribution.dtype != np.float32:
    raise ValueError("`probability_distribution` must have dtype `float32` got"
                     " {}".format(probability_distribution.dtype))
  if scatterer_distribution.dtype != np.float32:
    raise ValueError("`scatterer_distribution` must have dtype `float32` got {}"
                     "".format(scatterer_distribution.dtype))

  probability_proto = tf.make_tensor_proto(probability_distribution)
  distribution_proto = tf.make_tensor_proto(scatterer_distribution)

  return tf.train.Example(features=tf.train.Features(feature={
    'probability_distribution': record_utils._bytes_feature(
      probability_proto.SerializeToString()),
    'scatterer_distribution': record_utils._bytes_feature(
      distribution_proto.SerializeToString()),
    'info/height': record_utils._int64_feature(probability_distribution.shape[0]),
    'info/width': record_utils._int64_feature(probability_distribution.shape[1]),
  }))


def _parse_example(example_serialized):
  """Parse tf.train.example written by `convert_to_example`."""
  feature_map = {
    'probability_distribution': tf.FixedLenFeature([], tf.string),
    'scatterer_distribution': tf.FixedLenFeature([], tf.string),
    'info/height': tf.FixedLenFeature([], tf.int64),
    'info/width': tf.FixedLenFeature([], tf.int64),
  }

  features = tf.parse_single_example(example_serialized, feature_map)

  probability_distribution = tf.io.parse_tensor(
    features['probability_distribution'], tf.float32)
  scatterer_distribution = tf.io.parse_tensor(
    features['scatterer_distribution'], tf.float32)

  shape = (features['info/height'], features['info/width'])

  probability_distribution = tf.reshape(probability_distribution, shape)
  scatterer_distribution = tf.reshape(scatterer_distribution, shape)

  return {
    "probability_distribution": probability_distribution,
    "scatterer_distribution": scatterer_distribution
  }
