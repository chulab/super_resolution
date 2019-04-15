"""Loads dataset of simulations and converts to envelopes."""
import argparse
import logging
import os
from typing import List
import multiprocessing as mp
import glob
import sys
from scipy import signal
import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_data import record_utils


def rf_to_envelope(
    tensor,
    axis
):
  """Applies a hilbert transform along `axis` of `tensor`."""
  return np.abs(signal.hilbert(tensor, axis=axis)).astype(np.float32)


def tfrecord_to_example(
    tfrecord_file
):
  """Loads tfrecord file and returns list of Examples contained."""
  dataset = tf.data.TFRecordDataset(tfrecord_file)
  iterator =  dataset.make_one_shot_iterator()
  examples = []
  while True:
    try:
      example_str = iterator.get_next()
      example = record_utils._parse_example(example_str)
      example = [i.numpy() for i in example]
      examples.append(example)
    except tf.errors.OutOfRangeError:
      logging.info("Parsed {} examples.".format(len(examples)))
      break
  return examples


def envelope_example(
    example
):
  """Parses example and applies envelope function."""
  distribution, observation = example
  observation = rf_to_envelope(observation, 1)
  return record_utils._construct_example(
    distribution=distribution, observation=observation)


def save_tfrecord(
    examples: List,
    file_name: str,
):
  writer = tf.python_io.TFRecordWriter(file_name)
  logging.info("writing examples using writer {}".format(writer))
  i = 0
  for example  in examples:
    writer.write(example.SerializeToString())
    i+=1
  logging.info("Wrote {} examples to {}".format(i, file_name))
  writer.close()


def parse_and_save(
    file_path,
    out_directory,
):
  name = os.path.basename(file_path)
  examples = tfrecord_to_example(file_path)
  processed_examples = [
    envelope_example(e) for e in examples
  ]
  file_name = os.path.join(out_directory, name)
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
