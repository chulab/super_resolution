"""Defines `RecordWriter` class which writes TFRecords."""

import os

import numpy as np
import tensorflow as tf

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
    self._current_shard = -1
    # The `_current_file` is the file to which examples are written.
    self._current_file = None
    # The `_currente_example_in_shard` is the number of the NEXT example to be
    # written. I.e. upon the next call to `save`.
    self._current_example_in_shard = 0

    self._writer = None

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

  def check_not_nan(self, array):
    return not np.isnan(array).any()

  def save(self, distribution, observation):
    """Saves an example to the current file.

    Args:
      distribution: `np.ndarray` of shape `[height, width]`
        representing scatterer distribution.
      observation: `np.ndarray` of shape `[angle_count, height, width, channels]`
        representing simulation on scatterer distribution.
    """
    if self.check_not_nan(distribution) and self.check_not_nan(observation):
      # If we have written `_examples_per_shard` then open a new file.
      self._maybe_reinitialize()
      example = record_utils._construct_example(distribution, observation)
      self._writer.write(example.SerializeToString())
      self._writer.flush()
      self._current_example_in_shard += 1

  def close(self):
    """Closes current file."""
    self._writer.close()

  def _maybe_reinitialize(self):
    """Checks number of examples written and opens new file if necessary."""
    if self._current_example_in_shard == self._examples_per_shard:
      self._close_current_file_and_initialize()
    # If this is the first file.
    if self._current_shard == -1:
      self._current_shard +=1
      self._initialize_file()

  def _close_current_file_and_initialize(self):
    self._writer.close()
    self._current_shard += 1
    self._initialize_file()

  def _initialize_file(self):
    filename = "{name}_{shard:07}".format(
      name=self.dataset_name, shard=self._current_shard)
    output_file = os.path.join(self.directory, filename)
    self._current_file = output_file
    self._initialize_writer(output_file)
    self._current_example_in_shard = 0

  def _initialize_writer(self, file):
    self._writer = tf.python_io.TFRecordWriter(file)