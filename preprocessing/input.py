"""Constructs input function for training."""

import logging

from typing import Callable

import tensorflow as tf


def input_fn(
    dataset_directory: str,
    parse_fn: Callable,
    interleave_cycle_length: int = 1,
    shuffle_buffer_size: int=1,
    batch_size: int=1,
    num_parallel_reads: int=1,
) -> tf.data.Dataset:
  """Returns a dataset for train and eval.

  This function first loads tfrecords from `dataset_directory`. These records
  are parsed using `parse_fn` which should also include any preprocessing.

  Args:
    dataset_directory: Directory containing `TFRecords` US examples.
    parse_fn: Function for parsing examples. Includes and preprocessing. Must
      have output of a Tuple: `(features, labels)`.
    interleave_cycle_length: Interleaving of loaded files.
    shuffle_buffer_size: Number of examples to shuffle over.
    batch_size: Number of examples per batch.
    num_parallel_reads: Number of examples to process in parallel (using
    `parse_fn`).

  Returns:
    tf.data.Datset with output `features_labels.`
  """
  file_pattern = dataset_directory + "*.tfrecord"
  logging.info("Looking for files with glob {}".format(file_pattern))

  # Makes `Dataset` of file names.
  files = tf.data.Dataset.list_files(file_pattern, shuffle=False)

  # Generates `Dataset` from each file and interleaves.
  dataset = files.interleave(
    tf.data.TFRecordDataset, cycle_length=interleave_cycle_length)

  # Shuffle examples.
  dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

  # Extract data.
  dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_reads)

  # Batch.
  dataset = dataset.batch(batch_size=batch_size)

  dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

  return dataset