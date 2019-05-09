"""Uses `online_dataset_utils` to write a dataset to disk.

This is useful for making a reusable dataset for evaluation or testing."""
import argparse

import numpy as np

import sys
import os
# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
tf.enable_eager_execution()

from online_simulation import online_dataset_utils

from training_data import record_writer

from training_data import record_utils


def create_iterator(params):
  """Create `dataset` iterator."""
  dataset = online_dataset_utils.random_circles_dataset(
    physical_dimension=params.physical_dimension,
    grid_unit=params.grid_dimension,
    min_radius=params.min_radius,
    max_radius=params.max_radius,
    max_count=params.max_count,
    scatterer_density=params.scatterer_density,
    db=params.db
  )

  return dataset.make_one_shot_iterator()


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

  return probability_distribution, scatterer_distribution


def make_and_save_dataset(
  output_directory,
  example_count,
  examples_per_shard,
  dataset_params,
  dataset_name,
):
  """Generate and save dataset of scatterer distributions/probabilities."""
  iterator = create_iterator(dataset_params)

  print("made iterator")

  writer = record_writer.RecordWriter(
    directory=output_directory,
    dataset_name=dataset_name,
    examples_per_shard=examples_per_shard,
  )

  for i in range(example_count):
    next = iterator.get_next()
    example = convert_to_example(
      probability_distribution=next['probability_distribution'].numpy(),
      scatterer_distribution=next['scatterer_distribution'].numpy(),
    )
    print("save example")
    writer.savev2(example)

  writer.close()


def parse_args():
  """Parse arguments."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--dataset_params',
    type=str,
    help='Comma separated list of "name=value" pairs.',
    default=''
  )

  parser.add_argument(
    '--output_directory',
    type=str,
    help='Output path.',
    default=''
  )

  parser.add_argument(
    '--examples_per_shard',
    type=int,
    help='Number of examples per shard.',
    default=''
  )

  parser.add_argument(
    '--example_count',
    type=int,
    help='Total number of examples.',
    default=''
  )

  parser.add_argument(
    '--name',
    type=str,
    help='Dataset name.',
    default='test'
  )

  return parser.parse_args()


def main():
  args = parse_args()

  params = online_dataset_utils.dataset_params()
  params.parse(args.dataset_params)

  directory = args.output_directory

  if not os.path.isdir(directory):
    directory = os.path.dirname(directory)
  if not os.path.exists(directory):
    os.makedirs(directory)

  make_and_save_dataset(
    output_directory=directory,
    example_count=args.example_count,
    examples_per_shard=args.examples_per_shard,
    dataset_params=params,
    dataset_name=args.name,
  )


if __name__ == "__main__":
  main()
