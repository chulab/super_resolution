"""Defines `RecordWriter` class which writes TFRecords."""

import os

import tensorflow as tf

from simulation import defs
from training_data import record_utils

class RecordWriter(object):
  """Saves ultrasound simulation data as `tf.TFRecords` examples.

  Example usage:
  # record_writer = RecordWriter()
  # observation = ...
  # simulation = ...
  # record_writer.write(observation, simulation)
  # ...
  # record_writer.close()

  It is the responsibility of the user to close the final file using
  `RecordWriter.close()`.

  Attributes:
    observation_spec: ObservationSpec object describing all elements to be saved.
    directory: Directory to save records.
    dataset_name: Optional string appended to dataset files. E.g. `train`,
      `eval`, `test`.
    examples_per_shard: Number of examples to save per `TFRecords` file.
  """

  def __init__(
      self,
      directory: str,
      dataset_name: str,
      examples_per_shard: int,
  ):
    self.directory = directory
    self._dataset_name = dataset_name
    self.examples_per_shard = examples_per_shard

    # The `_current_shard` is always the shard to which examples are written.
    self._current_shard = 0
    # The `_current_file` is the file to which examples are written.
    self._current_file = None
    # The `_currente_example_in_shard` is the number of the NEXT example to be
    # written. I.e. upon the next call to `save`.
    self._current_example_in_shard = 0

    self._initialize_file()

  @property
  def directory(self):
    return self._directory

  @directory.setter
  def directory(self, directory):
    if not os.path.isdir(directory):
      raise ValueError("Directory is not valid.")
    self._directory = directory

  @property
  def dataset_name(self):
    return self._dataset_name

  @property
  def examples_per_shard(self):
    return self._examples_per_shard

  @examples_per_shard.setter
  def examples_per_shard(self, examples_per_shard):
    if examples_per_shard <= 0:
      raise ValueError("`examples_per_shard` must be positive integer")
    self._examples_per_shard = examples_per_shard

  def save(self, distribution, observation):
    """Saves an example to the current file.

    Args:
      distribution: `np.ndarray` of shape `[height, width]`
        representing scatterer distribution.
      observation: `np.ndarray` of shape `[angle_count, height, width, channels]`
        representing simulation on scatterer distribution.
    """
    example = record_utils._construct_example(distribution, observation)
    self._writer.write(example.SerializeToString())
    self._current_example_in_shard += 1
    # If we have written `_examples_per_shard` then open a new file.
    self._maybe_close()

  def close(self):
    """Closes current file."""
    self._writer.close()

  def _maybe_close(self):
    """Checks number of examples written and opens new file if necessary."""
    if self._current_example_in_shard == self._examples_per_shard:
      self._close_current_file_and_initialize()

  def _close_current_file_and_initialize(self):
    self._writer.close()
    self._current_shard += 1
    self._initialize_file()

  def _initialize_file(self):
    filename = "{}_{}".format(self.dataset_name, self._current_shard)
    output_file = os.path.join(self.directory, filename)
    self._current_file = output_file
    self._initialize_writer(output_file)
    self._current_example_in_shard = 0

  def _initialize_writer(self, file):
    self._writer = tf.python_io.TFRecordWriter(file)