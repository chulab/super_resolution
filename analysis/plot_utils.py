"""Utility functions for plotting."""

import os

from matplotlib_scalebar import scalebar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from scipy import signal

from training_data import utils


def colorbar(mappable):
  ax = mappable.axes
  fig = ax.figure
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  return fig.colorbar(mappable, cax=cax)

def plot_observation_prediction__distribution(
    observation: np.ndarray,
    distribution: np.ndarray,
    prediction: np.ndarray,
    grid_dimension: float,
    output_dir: str,
):
  fig, ax = plt.subplots(1, 3)

  o = ax[0].imshow(np.abs(signal.hilbert(observation, axis=0)), interpolation=None)
  ax[0].set_title("Observation")
  sb = scalebar.ScaleBar(grid_dimension)
  ax[0].add_artist(sb)
  colorbar(o)

  d = ax[1].imshow(distribution, interpolation=None)
  ax[1].set_title("Distribution")
  sb = scalebar.ScaleBar(grid_dimension)
  ax[1].add_artist(sb)
  colorbar(d)

  p = ax[2].imshow(prediction, interpolation=None)
  ax[2].set_title("Prediction")
  sb = scalebar.ScaleBar(grid_dimension)
  ax[2].add_artist(sb)
  colorbar(p)

  output_file = os.path.join(output_dir, 'prediction_and_true_distribution')
  plt.savefig(output_file)

  plt.tight_layout(h_pad=1)

  del fig


def plot_observation_example(
    observation,
    observation_spec,
    output_dir,
    update_grid_dimension:None,
):
  # Save observations.
  observation_images = utils.extract_angles_and_frequencies(
    observation, observation_spec)

  for c_i, image in enumerate(observation_images):
    observation_file_name = os.path.join(
      output_dir, "observation_{}.png".format(c_i))
    fig, ax = plt.subplots(1, 2)
    rf = ax[0].imshow(image.image, interpolation=None)
    env = ax[1].imshow(
      np.abs(signal.hilbert(image.image, axis=0)), interpolation=None)

    colorbar(rf)
    colorbar(env)

    fig.suptitle(
      "Angle {angle} Frequency {freq} Mode {mode} with {pix} pixels.".format(
      angle = image.angle, freq=image.psf_description.frequency,
        mode=image.psf_description.mode, pix=str(image.image.shape)
      ))
    for ax_ in ax:
      if update_grid_dimension:
        # If a `grid_dimension` is provided, use it. This corresponds to the
        # situation that the grid dimension has be modified by the input
        # pipeline.
        grid_dimension=update_grid_dimension
      else:
        grid_dimension=observation_spec.grid_dimension
      sb = scalebar.ScaleBar(grid_dimension)
      ax_.add_artist(sb)

    plt.savefig(observation_file_name)
    plt.close(fig)