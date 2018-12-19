"""Main file to generate and save distribution/observation dataset."""
import argparse
from datetime import datetime
import math
import os
import sys
from typing import Generator

import numpy as np
import tensorflow as tf

from training_data import record_utils
from simulation import defs
from simulation import estimator


def reduce_split(array: np.ndarray, axis: int):
  """Splits n-dimensional array along `axis` into `n-1` dimensional chunks."""
  return [np.squeeze(a) for a in np.split(array, array.shape[axis], axis)]


def _dataset_from_generator(
  generator: Generator[dict, None, None],
  observation_spec: defs.ObservationSpec,
  output_directory: str,
  dataset_name: str,
  num_examples_in_dataset: int,
  examples_per_shard: int,
):
  """Saves simulation examples produced by generator to sharded files.

  Args:
    generator: Generator which returns a dictonary containing `distribution`
      and `observation` elements. Both should have shape
      `[batch, height, width]`.
  """

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

      distribution_bloc = output["distribution"]
      observation_bloc = output["observation"]
      assert distribution_bloc.shape[0] == observation_bloc.shape[0]

      for distribution, observation in zip(
          reduce_split(distribution_bloc, 0),
          reduce_split(observation_bloc, 0)):

        example = record_utils._construct_example(
          distribution, observation, observation_spec)

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
    distributions: np.ndarray,
    observation_spec: defs.ObservationSpec,
    axial_psf_length: int,
    lateral_psf_length: int,
    dataset_name: str,
    output_directory: str,
    examples_per_shard: int,
):
  """Creates dataset by simulating observation of scatterer distribution.

  This function builds a dataset of simulated ultrasound images. First, it
  builds a `SimulationEstimator` which generates batches of simulated images
  based on input `distributions`. It then saves this dataset to disk as a
  `tfrecords` example.

  Each example contains:
    * Distribution - Array of `[Height, Width]`
    * Observation - Array of `[Height', Width']` (may be different from
      `Distribution`.
    * ObservationSpec - Information on simulation.
    For further information see `record_utils._construct_example`.

  Examples are stored in `output_directory` in a set of files. Each file will
  contain `examples_per_shard` examples.

  Args:
    distributions: `np.ndarray` of shape `[batch, height, width]` describing
      distribution of scatterers.
    observation_spec: `ObservationSpec` parameterizing simulation.
    datset_name: Name describing dataset (e.g. `train`, `eval`).
    output_directory: Path to directory for dataset.
    examples_per_shard: `int` of number of examples per shard (file) in dataset.

  Raises:
    ValueError: If `distributions` does not have shape
      `[num_examples, height, width]`.
  """
  if distributions.ndim != 3:
    raise ValueError("`distributions` must have shape "
                     "`[batch, height, width]` but got {}".format(
      distributions.shape))
  if distributions.dtype != np.float32:
    raise ValueError("`distributions` must have dtype `float32` but got "
                     "{}".format(distributions.dtype))

  # Load distribution.
  def _input_fn():
    return tf.data.Dataset.from_tensor_slices(distributions).batch(1)

  num_examples_in_dataset = distributions.shape[0]

  # Make Estimator.
  simulation_estimator = estimator.SimulationEstimator(
    observation_spec, lateral_psf_length, axial_psf_length)

  # Get predictions.
  simulation_output = simulation_estimator.predict(
    _input_fn, yield_single_examples=False)

  _dataset_from_generator(
    simulation_output,
    observation_spec,
    output_directory,
    dataset_name,
    num_examples_in_dataset,
    examples_per_shard
  )


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--observation_dataset_path', dest='observation_dataset_path',
                      help='Path to the scatterer distribution dataset (np.ndarray).',
                      type=str,
                      required=True)

  parser.add_argument('--simulation_param_path',
                      dest='simulation_param_path',
                      help='Path to the simulation param JSON file.',
                      type=str,
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

  parser.add_argument('--lateral_psf_length', dest='lateral_psf_length',
                      help='Number of shards to make.', type=int,
                      required=True)

  parsed_args = parser.parse_args()

  return parsed_args


def main():
  args = parse_args()

  with open(args.observation_dataset_path) as f:
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