"""Loads dataset of simulations and converts to envelopes."""
import argparse
import logging
import os
from typing import List
import multiprocessing as mp
import glob
import sys

import tensorflow as tf

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_data import record_utils
from preprocessing import signals

def rf_to_envelope(
    tensor,
    axis
):
  """Applies a hilber transform along `axis` of `tensor`."""
  return signals.hilbert(tensor, axis)


def tfrecord_to_example(
    tfrecord_file
):
  """Loads tfrecord file and returns list of Examples contained."""
  iterator = tf.python_io.tf_record_iterator(tfrecord_file)
  examples = []
  while True:
    try:
      example_str = iterator.next()
      examples.append(record_utils._parse_example(example_str))
    except tf.errors.OutOfRangeError:
      logging.info("Parsed {} examples.".format(len(examples)))
      break
  return examples


def envelope_example(
    example
):
  """Parses example and applies envelope function."""
  distribution, observation = example
  observation = rf_to_envelope(observation)
  with tf.Session() as sess:
    distribution, observation = sess.run(
      [distribution, observation])
  return record_utils._construct_example(
    distribution=distribution, observation=observation)


def save_tfrecord(
    examples: List,
    file_name: str,
):
  writer = tf.python_io.TFRecordWriter(file_name)
  logging.info("writing examples using writer {}".format(writer))
  for example  in examples:
    writer.write(example.SerializeToString())
  writer.close()


def parse_and_save(
    file_path,
    directory,
):
  name = os.path.basename(file_path)
  examples = tfrecord_to_example(file_path)
  processed_examples = [
    envelope_example(e) for e in examples
  ]
  file_name = os.path.join(directory, name)
  save_tfrecord(processed_examples, file_name)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--num_workers", type=int, default=mp.cpu_count()
)
  parser.add_argument("--source_directory", type=str)
  parser.add_argument("--output_directory", type=str)

  return parser.parse_args()


def main():
  logging.basicConfig(filename='convert_to_envelope.log', level=logging.DEBUG)

  args = parse_args()
  logging.info(args)

  glob_str = "{}/*.tfrecord".format(args.source_directory)
  logging.info("using glob_str {}".format(glob_str))
  files = glob.iglob(glob_str)

  pool = mp.Pool(processes=args.num_workers)

  pool.starmap(
    parse_and_save,
    [(file, args.output_directory) for file in files]
  )

  logging.info("terminating pool.")
  pool.terminate()

if __name__ == "__main__":
  main()
