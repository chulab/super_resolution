"""Main file to generate and save distribution/observation dataset."""

from datetime import datetime
import math
import os
import sys
from typing import Generator

import tensorflow as tf

from training_data import record_utils
from simulation import defs

def _dataset_from_generator(
  generator: Generator[dict, None, None],
  observation_spec: defs.ObservationSpec,
  output_directory: str,
  dataset_name: str,
  num_examples_in_dataset: int,
  examples_per_shard: int,
):
  """Saves simulation examples produced by generator to sharded files."""

  num_shards = math.ceil(num_examples_in_dataset / examples_per_shard)

  counter = 0
  for shard in range(num_shards):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    output_filename = '%s-%.5d-of-%.5d' % (dataset_name, shard + 1, num_shards)
    output_file = os.path.join(output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    shard_counter = 0

    for _ in range(examples_per_shard):
      # Get next output. Typically this will be from an estimator which is running
      # in the background.
      try:
        output = next(generator)
      except StopIteration:
        print("`StopIteration` raised by generator. End of examples. Closing "
              "file.")
        break

      distribution = output["distribution"]
      observation = output["observation"]

      example = record_utils._construct_example(
        distribution, observation, observation_spec
      )

      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 500:
        print(
          '%s : Processed %d of %d images in batch.' %
          (datetime.now(), counter, num_examples_in_dataset))
        sys.stdout.flush()

    writer.close()
    print('%s : Wrote %d images to %s' %
          (datetime.now(), shard_counter, output_file))
    sys.stdout.flush()

  print('%s : Wrote %d images to %d shards.' %
        (datetime.now(), counter, num_shards))
  sys.stdout.flush()


def create(
    dataset,
    dataset_name,
    output_directory,
    examples_per_shard,
    simulation_params,
):
  """Creates dataset by simulating observation of scatterer distribution."""

  # Load distribution.

  # Make Estimator.

  # Get predictions.

  _dataset_from_generator(
    simulation_outputs, observation_spec, output_directory, dataset_name)

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset_path', dest='dataset_path',
                      help='Path to the scatterer distribution dataset (np.Array).',
                      type=str,
                      required=True)

  parser.add_argument('--simulation_param_path', dest='simulation_param_path',
                      help='Path to the simulation param JSON file.', type=str,
                      required=True)

  parser.add_argument('--prefix', dest='dataset_name',
                      help='Prefix for the tfrecords (e.g. `train`, `test`, `val`).',
                      type=str,
                      required=True)

  parser.add_argument('--output_dir', dest='output_dir',
                      help='Directory for the tfrecords.', type=str,
                      required=True)

  parser.add_argument('--shards', dest='num_shards',
                      help='Number of shards to make.', type=int,
                      required=True)

  parsed_args = parser.parse_args()

  return parsed_args


def main():
  args = parse_args()

  with open(args.dataset_path) as f:
    dataset = np.load(f)

  with open(args.simulation_param_path) as f:
    simulation_params = json.load(f)

  create(
    dataset=dataset,
    dataset_name=args.dataset_name,
    output_directory=args.output_dir,
    num_shards=args.num_shards,
    simulation_params=args.simulation_params
  )