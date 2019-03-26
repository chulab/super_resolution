"""Verification for input pipeline by saving some data."""

import logging
import time
import os

import matplotlib.pyplot as plt

import tensorflow as tf

def save_input(
    dataset: tf.data.Dataset,
    dataset_name: str,
    output_dir: str,
    iteration_count: int = 11,
):
  """Saves output from dataset.

  Args:
    dataset: tf.data.Dataset.
    iteration_count: Number of iterations to average over to get output speed.
    dataset_name: Name of `dataset`. Also appended to output images.
    output_dir: Directory to save output images.
  """
  iterator = dataset.make_one_shot_iterator()
  output = iterator.get_next()

  with tf.Session () as sess:
    time_start = time.time()
    for _ in range(iteration_count):
      sess.run(output)
      logging.debug("Called dataset.")
    rate = iteration_count / (time.time() - time_start)
    logging.info("{name} Dataset produces output at rate {rate} examples/sec"
                 "".format(name=dataset_name, rate=rate))

    test_observation, test_distribution = sess.run(output)

    logging.info("{name} Dataset `observation` has shape {shape}".format(
      name=dataset_name, shape=test_observation.shape))
    logging.info("{name} Dataset `distribution` has shape {shape}".format(
      name=dataset_name, shape=test_distribution.shape))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(test_observation[0, -1, ..., 0])
    ax[0].set_title("Observation")
    ax[1].imshow(test_distribution[0])
    ax[1].set_title("Distribution")
    fig.suptitle("Output dataset {name}".format(name=dataset_name))
    output_file = os.path.join(output_dir, dataset_name)
    fig.savefig(output_file)

    del fig