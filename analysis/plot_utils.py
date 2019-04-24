"""Utility functions for plotting."""

import os
import math
from matplotlib_scalebar import scalebar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from scipy import signal
from scipy import stats

from training_data import utils


def colorbar(mappable):
  ax = mappable.axes
  fig = ax.figure
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  return fig.colorbar(mappable, cax=cax)

def plot_observation_prediction_distribution(
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

def find_nearest_square_divisor(x: int):
  n = math.floor(math.sqrt(x))
  while True:
    if x % n == 0:
      return (n, int(x / n))
    n -= 1

def plot_grid(images, file_name=None, **kwargs):
  """Plots a list of images in a grid.

  Either saves or reuturns matplotlib figure.

  Args:
    images: List of 2D np.ndarrays.
    file_name: Optional string representing path to save figure.
    **kwargs: kwargs passed to `plot_with_colorbar_and_scalebar`.

  Returns:
    If no `file_name` is provided, then returns figure containing plots.
    """
  r, c = find_nearest_square_divisor(len(images))
  fig, ax = plt.subplots(r, c, figsize=(c * 3, r * 3))
  if isinstance(ax, np.ndarray):
    ax = ax.flatten()
  else:
    ax = [ax]
  for a, p in zip(ax, images):
    plot_with_colorbar_and_scalebar(a, p, **kwargs)

  if file_name is not None:
    plt.savefig(file_name)
    plt.close(fig)
  else:
    return fig


def plot_with_colorbar_and_scalebar(ax, array, scale=None, **kwargs):
  im = ax.imshow(array, **kwargs)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im, cax=cax)

  if scale is not None:
    sb = scalebar.ScaleBar(scale)
    ax.add_artist(sb)

def scatter_with_error_bar(
    x,
    y,
    ax,
):
  # Bin locations for quantized data.
  bin_locations = np.unique(x)
  nbins = bin_locations.shape[0]

  # Bin and compute error bars.
  n, _ = np.histogram(x, bins=nbins)
  sy, _ = np.histogram(x, bins=nbins, weights=y)
  sy2, _ = np.histogram(x, bins=nbins, weights=[i ** 2 for i in y])
  mean = sy / n
  std = np.sqrt(sy2 / n - mean * mean)

  # Plot binned data with error bars.
  ax.errorbar(bin_locations, mean, yerr=std, capsize=10, markeredgewidth=2)

  x = np.array(x)
  x = x + np.random.randn(x.shape[0]) * .1
  y = np.array(y)
  y = y + np.random.randn(y.shape[0]) * .1

  # Scatter plot of data.
  ax.scatter(x, y, s=2)