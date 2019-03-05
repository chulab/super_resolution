"""Functions for saving datasets consisting of numpy arrays."""

import math
import os

import numpy as np


def save_dataset(
  generator,
  count,
  shard_size,
  save_directory,
  file_prefix,
):
  """Saves datset of elements from generator.

  This function builds and saves a dataset of scatterer distributions given
  a generator which produces np.ndarrays contining scatterer distributions.

  Each shard contains a single np.ndarrays with shape
  `[shard_size] + generator_output_shape`. This implies that the generator must
  return a consistent array size.

  Args:
    generator: Generator which returns np.ndarray objects.
    count: Total number of distributions in dataset.
    shard_size: Number of distributions per shard.
    save_directory: Path to folder where data will be saved.
    file_prefix: Optional prefix to add to filenames.

  Raises:
    ValueError if `save_directory` is not valid.
  """
  if not os.path.isdir(save_directory):
    raise ValueError("`save_directory` must be a valid directory.")

  file_count = int(math.ceil(float(count) / shard_size))
  for shard in range(1, file_count + 1):
    output = np.stack([next(generator) for _ in range(shard_size)], 0)
    filename = "%s_%d_of_%d.npy" % (file_prefix, shard, file_count)
    output_file = os.path.join(save_directory, filename)
    np.save(output_file, output)
