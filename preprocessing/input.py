"""Constructs input function for training."""

import argparse
import logging
import os

from typing import Callable, List

import tensorflow as tf


def input_fn(
    dataset_directory: str,
    parse_fns: List[Callable],
    parallel_calls: List[int],
    interleave_cycle_length: int = 1,
    shuffle_buffer_size: int=1,
    batch_size: int=1,
    prefetch: int=None,
    file_signature: str="*.tfrecord"
) -> tf.data.Dataset:
  """Returns a dataset for train and eval.

  This function first loads tfrecords from `dataset_directory`. These records
  are parsed using `parse_fns` which should also include any preprocessing.

  Args:
    dataset_directory: Directory containing `TFRecords` US examples.
    parse_fns: Function for parsing examples. Includes and preprocessing. Must
      have output of a Tuple: `(features, labels)`.
    parallel_calls: List of same length as `parse_fns`. Number of examples to
      process in parallel.
    interleave_cycle_length: Interleaving of loaded files.
    shuffle_buffer_size: Number of examples to shuffle over.
    batch_size: Number of examples per batch.
    file_signature: Glob signature of files in `dataset_directory`

  Returns:
    tf.data.Datset with output `features_labels.`
  """
  if prefetch is None:
    prefetch = tf.contrib.data.AUTOTUNE

  with tf.name_scope("input"):
    file_pattern = os.path.join(dataset_directory, file_signature)
    logging.info("Looking for files with glob {}".format(file_pattern))

    # Makes `Dataset` of file names.
    files = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    # Repeat and shuffle.
    files = files.apply(tf.data.experimental.shuffle_and_repeat(100))

    # Generates `Dataset` from each file and interleaves.
    dataset = files.interleave(
      tf.data.TFRecordDataset, cycle_length=interleave_cycle_length)

    # Shuffle examples.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Extract data.
    for parse_fn, parallel_calls in zip(parse_fns, parallel_calls):
      print(parse_fn)
      dataset = dataset.map(parse_fn, num_parallel_calls=parallel_calls)

    # Batch.
    dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.prefetch(prefetch)
    return dataset


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--parallel_calls',
    type=int,
    default=1,
  )

  parser.add_argument(
    '--interleave_cycle_length',
    type=int,
    default=1,
  )

  parser.add_argument(
    '--shuffle_buffer_size',
    type=int,
    default=1,
  )

  parser.add_argument(
    '--batch_size',
    dest='batch_size',
    type=int,
    default=1,
  )

  parser.add_argument(
    '--prefetch',
    type=int,
    default=1,
  )

  parser.add_argument(
    '--file_signature',
    type=str,
    default="*.tfrecord",
  )

  args, _ = parser.parse_known_args()

  return args


def make_input_fn(
    dataset_directory,
    parse_fns,
  ):

  args = parse_args()

  return input_fn(
    dataset_directory=dataset_directory,
    parse_fns=parse_fns,
    parallel_calls=[args.parallel_calls] * len(parse_fns),
    interleave_cycle_length=args.interleave_cycle_length,
    shuffle_buffer_size=args.shuffle_buffer_size,
    batch_size=args.batch_size,
    prefetch=args.prefetch,
    file_signature=args.file_signature,
  )