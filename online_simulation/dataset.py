"""Uses `online_dataset_utils` to write a dataset to disk.

This is useful for making a reusable dataset for evaluation or testing."""

import argparse
import numpy as np

import sys
import os

import tensorflow as tf
tf.enable_eager_execution()

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_simulation import online_dataset_utils

from training_data import record_writer
from online_simulation import record_utils


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


def make_and_save_dataset(
  output_directory,
  example_count,
  examples_per_shard,
  dataset_params,
  dataset_name,
):
  """Generate and save dataset of scatterer distributions/probabilities."""
  iterator = create_iterator(dataset_params)


  writer = record_writer.RecordWriter(
    directory=output_directory,
    dataset_name=dataset_name,
    examples_per_shard=examples_per_shard,
  )

  for i in range(example_count):
    next = iterator.get_next()
    example = record_utils.convert_to_example(
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
