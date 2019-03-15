"""Utility to plot sample from simulation.

Saves plots in same location as simulation data.

Example usage:

  # python plot_simulations.py \
  # -f $FILE_PATH$ \
  # -gd .1e-3 .1e-3 \
"""
import argparse
import os
import sys

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib_scalebar import scalebar
import numpy as np
from scipy import signal
import tensorflow as tf


from simulation import create_observation_spec
from training_data import record_utils
from training_data import utils

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-f',
    '--file',
    dest='file',
    help='Path to simulation TFRecords file.',
    type=str,
    required=True,
    )

  parser.add_argument(
    '-os',
    '--observation_spec',
    dest='observation_spec_path',
    help='Path to `observation_spec` parametrizing simulation.',
    required=True
    )

  return parser.parse_args()

def main():
  args = parse_args()
  file_directory = os.path.dirname(args.file)

  # Load `observation_spec`.
  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path
  )

  test_dataset = tf.data.TFRecordDataset([args.file])
  test_dataset = test_dataset.map(record_utils._parse_example)
  iterator = test_dataset.make_one_shot_iterator()
  next_dist, next_obs = iterator.get_next()

  with tf.Session() as sess:
    distribution, observation = sess.run(
      [next_dist, next_obs])

  print(distribution.shape)
  print(observation.shape)

  # Save distribution.
  distribution_filename = os.path.join(file_directory, "distribution.png")
  fig = plt.figure()
  im = plt.imshow(distribution, interpolation=None)
  fig.colorbar(im)
  fig.suptitle("Sample distribution with {} pixels.".format(
    str(distribution.shape)), fontsize=20,)
  sb = scalebar.ScaleBar(observation_spec.grid_dimension)
  plt.gca().add_artist(sb)
  plt.savefig(distribution_filename)
  plt.close(fig)

  # Save observations.
  observation_images = utils.extract_angles_and_frequencies(
    observation[0], observation_spec)

  for c_i, image in enumerate(observation_images):
    observation_file_name = os.path.join(
      file_directory, "observation_{}.png".format(c_i))
    fig, ax = plt.subplots(1, 2)
    rf = ax[0].imshow(image.image, interpolation=None)
    env = ax[1].imshow(
      np.abs(signal.hilbert(image.image, axis=0)), interpolation=None)

    plt.colorbar(rf, ax=ax[0])
    plt.colorbar(env, ax=ax[1])

    fig.suptitle(
      "Angle {angle} Frequency {freq} Mode {mode} with {pix} pixels.".format(
      angle = image.angle, freq=image.psf_description.frequency,
        mode=image.psf_description.mode, pix=str(image.image.shape)
      ))
    for ax_ in ax:
      sb = scalebar.ScaleBar(observation_spec.grid_dimension)
      ax_.add_artist(sb)

    plt.savefig(observation_file_name)
    plt.close(fig)


if __name__ == "__main__":
  main()