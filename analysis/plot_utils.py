"""Utility functions for plotting."""

import os
import math
from matplotlib_scalebar import scalebar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import OrderedDict
from online_simulation import online_simulation_utils

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

  plt.tight_layout(h_pad=1)

  if output_dir is not None:
    output_file = os.path.join(output_dir, 'prediction_and_true_distribution')
    plt.savefig(output_file)
    del fig
  else:
    return fig

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

def plot_grid(images, file_name=None, titles=None, **kwargs):
  """Plots a list of images in a grid.

  Either saves or returns matplotlib figure.

  Args:
    images: List of 2D np.ndarrays.
    file_name: Optional string representing path to save figure.
    titles: List of strings, denoting title for each image in grid.
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
  for i, (a, p) in enumerate(zip(ax, images)):
    if titles is not None:
      a.set_title(titles[i])
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


def plot_same_grid(images, file_name=None, titles=None, **kwargs):
  '''
  Plots in a grid with a single common scale and color bar.

  Either saves or returns matplotlib figure.

  Args:
    images: List of 2D np.ndarrays.
    file_name: Optional string representing path to save figure.
    titles: List of strings, denoting title for each image in grid.
    **kwargs: if `scale` is in **kwargs, plots scalebar according to scale.

  Returns:
    If no `file_name` is provided, then returns figure containing plots.
  '''
  r, c = find_nearest_square_divisor(len(images))
  fig = plt.figure()
  grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(r,c),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
  stacked = np.stack(images, axis=0)
  min = np.amin(stacked)
  max = np.amax(stacked)

  for i, (p, ax) in enumerate(zip(images, grid)):
    im = ax.imshow(p, vmin=min, vmax=max)
    if titles is not None:
      ax.set_title(titles[i])

  ax.cax.colorbar(im)
  ax.cax.toggle_label(True)

  if 'scale' in kwargs is not None:
    sb = scalebar.ScaleBar(kwargs['scale'])
    ax.add_artist(sb)

  # proportion horizontally between grid and colorbar
  rect_width = (3 * c) / (3 * c + 1)
  rect=[0, 0, rect_width, 1]
  plt.tight_layout(rect=rect)

  if file_name is not None:
    plt.savefig(file_name)
    plt.close(fig)
  else:
    return fig


def get_tensors_from_tensorboard(
  tb_dir,
  tensor_tags,
  step,
  squeeze_last = True,
):
  '''
  Retrieves all tensors given by their tags from a TFEvent file at a given step.

  Args:
    tb_dir: Directory of TFEvent file
    tensor_tags: Name (with namescope) of tensors used when calling tf.summary.
    step: Step to retrieve tensors.
    squeeze_last: Whether last dimension of tensors should be squeezed.

  Returns:
    List of retrieved tensors.
    If the ith tensor has shape [tensor_shape], the ith element of the list will
    be of the shape [times, tensor_shape] where times is the number of times the
    ith tensor appears during the given step.
  '''

  acc = EventAccumulator(tb_dir)
  acc.Reload()

  collated = []

  with tf.Graph().as_default():
    for tag in tensor_tags:
      tag_storage = []
      for tensor in acc.Tensors(tag):
        if tensor.step == step:
          # tag_array has shape (batch, tensor_base_shape) or
          # (batch, tensor_base_shape, 1)
          tag_array = tf.make_ndarray(tensor.tensor_proto)
          if squeeze_last:
            tag_array = np.squeeze(tag_array, axis=-1)
          tag_storage.append(tag_array)
      concated = np.concatenate(tag_storage, axis=0)
      collated.append(concated)

  return collated


def get_scalars_from_tensorboard(
  tb_dir,
  scalar_tags,
  step,
):
  '''
  Retrieves all scalars given by their tags from a TFEvent file at a given step.

  Args:
    tb_dir: Directory of TFEvent file
    scalar_tags: Name (with namescope) of scalars used when calling tf.summary.
    step: Step to retrieve tensors.

  Returns:
    List of retrieved scalars.
  '''

  acc = EventAccumulator(tb_dir)
  acc.Reload()
  collated = []

  with tf.Graph().as_default():
    for tag in scalar_tags:
      for scalar in acc.Scalars(tag):
        if scalar.step == step:
          collated.append(scalar.value)

  return collated

def get_all_tensor_from_tensorboard(
  tb_dir,
  tensor_tag
):
  '''
  Retrieves all instances of a tensor with a given tag across all steps in a
  TFEvent.

  Args:
    tb_dir: Directory of TFEvent file
    tensor_tag: Name (with namescope) of tensor used when calling tf.summary.

  Returns:
    List of steps and List of tensor instances.
    Because a tensor may appear multiple times in one step, tensor instances
    belonging to the same step are stacked along the 0th dimension.
  '''


  acc = EventAccumulator(tb_dir)
  acc.Reload()

  step_to_record = OrderedDict()

  with tf.Graph().as_default():
    for tensor in acc.Tensors(tensor_tag):
      array = np.squeeze(tf.make_ndarray(tensor.tensor_proto), -1)
      if tensor.step not in step_to_record:
        step_to_record[tensor.step] = []
      step_to_record[tensor.step].append(array)

  steps = step_to_record.keys()
  record = step_to_record.values()
  record = [np.concatenate(tensor_list, 0) for tensor_list in record]

  return steps, record

def get_all_scalar_from_tensorboard(
  tb_dir,
  scalar_tag
):
  '''
  Retrieves all instances of a scalar with a given tag across all steps in a
  TFEvent.

  Args:
    tb_dir: Directory of TFEvent file
    scalar_tag: Name (with namescope) of scalar used when calling tf.summary.

  Returns:
    List of steps and List of scalar instances.
  '''

  acc = EventAccumulator(tb_dir)
  acc.Reload()
  scalar_record = []
  steps = []

  with tf.Graph().as_default():
    for scalar in acc.Scalars(scalar_tag):
      scalar_record.append(scalar.value)
      steps.append(scalar.step)

  return steps, scalar_record


def plot_grid_from_tensorboard(
  tb_dir,
  tensor_tags,
  step,
  titles = None,
  same = True,
  limit = None,
  **kwargs
):
  '''
  Searches event files in tensorboard directory for image tensors labelled by
  tags at a certain step. Plots a grid figure from these image tensors
  (as a set) for each time they appear.

  Args:
    tb_dir: directory to search.
    tensor_tags: array of names/tags used when calling tf.summary.
    step: desired step.
    titles: List of strings, denoting titles of the tensors.
    same: Whether a common colorbar (and/or scalebar) should be used.
    **kwargs: refer to plot_grid.

  Returns:
    List of figures depicting each set of image tensors.
  '''

  if len(tensor_tags) == 0:
    return

  collated = get_tensors_from_tensorboard(tb_dir, tensor_tags, step)
  collated = np.stack(collated, -1)

  if limit is not None:
    collated = collated[:limit, ...]

  if same:
    plot_fn = plot_same_grid
  else:
    plot_fn = plot_grid

  # split along batch dim (0) to obtain each set of images before splitting
  # along last dim (-1) to generate a list of each set
  split_images = [[np.squeeze(im, -1) for im in
    np.split(np.squeeze(images, 0), images.shape[-1], -1)] for images in
    np.split(collated, collated.shape[0], 0)]
  figures = [plot_fn(images, titles=titles, **kwargs) for images in
    split_images]

  return figures

def plot_observation_from_distribution(
  distribution,
  psfs,
  grid_dimension,
  limit = None,
  **kwargs
):
  '''
  Simulates US signals from scatterers given psfs and plots them in a grid
  figure.

  Args:
    distribution: np.ndarray of shape [B, H, W].
    psfs: List of defs.PsfDescription.
    grid_dimension: float representing grid size in metres.
    limit: maximum number of sets along batch dimension to plot.
    **kwargs: refer to plot_grid.

  Returns:
    List of figures.
  '''
  distribution = tf.convert_to_tensor(distribution)
  if limit is not None:
    distribution = distribution[:limit, ...]

  sim = online_simulation_utils.USSimulator(
    psfs=psfs,
    image_grid_size=distribution.shape.as_list()[1:],
    grid_dimension=grid_dimension,
  )

  simulated = sim.observation_from_distribution(distribution)
  simulated = simulated.eval()

  titles = ["A {} F {} M {}".format('{:.3g}'.format(p.angle), '{:.3g}'.format(
    p.psf_description.frequency), p.psf_description.mode) for p in psfs]

  split_images = [[np.squeeze(im, -1) for im in
    np.split(np.squeeze(images, 0), images.shape[-1], -1)] for images in
    np.split(simulated, simulated.shape[0], 0)]
  figures = [plot_grid(images, titles=titles, **kwargs) for images in
    split_images]

  return figures
