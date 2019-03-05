"""Functions to generate dataset (in shards) for certain geometric shapes"""

import numpy as np
import math
import random
from skimage.draw import line_aa

from simulation import response_functions

from typing import List

VALID_TYPES = ['CIRCLE', 'LINE']


def _circle(
  coordinates: np.ndarray,
  origin: List[float],
  radius: float,
):
  """Returns `np.ndarray` with a circle of `radius` located at `origin`.

  Set cells are `1` while empty cells are `0`.

  Args:
    coordinates: Array containing coordinates in n-d space.
    origin: Circle center location in physical n-d space.
    radius: Radius of circle in physical units.

  Returns:
    np.ndarray with total dimensions num_dim with each dimension having
    size `array_size`.

  Raises:
    ValueError: If `coordinates` and `origin` are not compatible.
  """
  if len(origin) != coordinates.shape[-1]:
    raise ValueError("`origin` and `coordinates` must have same number of"
                     "dimensions but got {} and {}".format(
      len(origin), coordinates.shape[-1]))

  distance_to_origin_squared = np.sum((coordinates - origin) ** 2, -1)

  return distance_to_origin_squared <= radius ** 2


def random_circles(
  dimensions: List[float],
  grid_dimensions: List[float],
  min_radius: float,
  max_radius: float,
  min_intensity:float,
  max_intensity:float,
  max_count: int,
):
  """Produces an array of balls in arbitrary dimensions.

  In the 2D mode this function produces an array with circles centered at
  random locations within the array. These circles will have a radius between
  `min_radius` and `max_radius`.

  Args:
    dimensions: Dimensions of box in metres.
    grid_dimensions: Grid size in metres.
    min_rad: Minimum circle radius in metres.
    max_rad: Maximum circle radius in metres.
    max_count: Maximum number of circles to place in array.

  Returns:
    `np.ndarray` with total dimensions num_dim with each dimension having
    size `physical_dim / grid_size`.
  """
  coordinates = np.stack(
    response_functions.coordinate_grid(dimensions, grid_dimensions, False),
    -1
  )

  # `box` will store circles.
  box = np.zeros(coordinates.shape[:-1])

  # Array will contain at least one ball, and up to `max_count`.
  count = np.random.randint(1, max_count + 1)

  for _ in range(count):
    # Get random circle origin.
    origin = [np.random.uniform(0, length) for length in dimensions]

    # Random radius.
    radius = np.random.uniform(min_radius, max_radius)

    # Intensity.
    intensity = np.random.uniform(min_intensity, max_intensity)

    # Add circle.
    box += _circle(coordinates, origin, radius) * intensity

  return np.clip(box, a_max=max_intensity)


def line_2d_endpoints(array_size: int, origin, grad: float):
  """Returns endpoints of the 2d line passing through an origin with a given
  gradient in an array.

  Args:
      array_size: size of each dimension of array.
      origin: Tuple(int, int) indices of a point on the line.
      grad: gradient at origin.

  Returns:
      Tuple(int, int, int, int) representing coordinates of left and right
      endpoints in the form of (x_left, y_left, x_right, y_right).
  """

  x_left = y_left = x_right = y_right = 0

  if grad == 0:
      return 0, origin[1], array_size, origin[1]
  elif grad > 0:
      x_diff_left = max(-1 * origin[0] , -1 * float(origin[1]) / grad)
      x_diff_right = min(array_size - 1 - origin[0], \
          float(array_size - 1 - origin[1]) / grad)
  else:
      x_diff_left = max(-1 * origin[0], \
          -1 * float(origin[1] - array_size + 1) / grad)
      x_diff_right = min(array_size - 1 - origin[0], \
          - 1 * float(origin[1]) / grad)

  x_left = floor(origin[0] + x_diff_left)
  y_left = ceil(origin[1] + x_diff_left * grad)
  x_right = floor(origin[0] + x_diff_right)
  y_right = ceil(origin[1] + x_diff_right * grad)

  return x_left, y_left, x_right, y_right

def line_fn(
  physical_dim: float,
  grid_size: float,
  max_count: int,
  sharpness: int = 100,
  num_dim: int = 2,
):
  """Produces an array of lines.

  Set cells are `1` while empty cells are `0`.

  Increasing sharpness causes line to be thinner and more salient.

  Args:
    physical_dim: length of box in metres.
    grid_size: grid size in metres.
    max_count: maximum number of lines generated.
    sharpness: anti-aliasing threshold for cell to be set.
    num_dim: number of dimensions of box (only 2 for now).

  Returns:
    np.ndarray with total dimensions num_dim with each dimension having
    size `physical_dim / grid_size`.
  """

  array_size = ceil(physical_dim / grid_size)
  count = np.random.randint(1, max_count + 1)
  box = np.zeros([array_size] * num_dim)

  for _ in range(count):
      origin = np.random.randint(0, array_size, num_dim)
      grad = tan(random.uniform(-pi / 2, pi / 2))

      x_left, y_left, x_right, y_right = line_2d_endpoints(array_size, origin, grad)

      rr, cc, val = line_aa(x_left, y_left, x_right, y_right)
      box[rr, cc] += val * 255

  box = (box > sharpness).astype(int)

  return box


def make_dataset(
  type: str,
  count: int,
  shard_size: int = 1000,
  file_prefix: str,
  directory: str,
  *args,
):
  """Writes data of a given type in a directory as .npy shards.

  Each shard file contains a np.ndarray of np.ndarray. Each np.ndarray element
  of the parent np.ndarray is a dataset of randomly generated shapes of `type`
  where set cells are '1' and empty cells are '0'.

  Args:
    type: shape to generate in data.
    count: total number of datasets.
    shard_size: number of datasets in a shard.
    file_prefix: prefix to be appended before shard index.
    directory: location to save.
    args: parameters to pass into the corresponding type_fn (e.g. circle_fn
          if type == 'circle' and line_fn if type == 'line').
  """

  shape_fn = None
  if (type == 'circle'):
      shape_fn = circle_fn
  elif (type == 'line'):
      shape_fn = line_fn
  else:
      raise ValueError("type must be in %s" % str(VALID_TYPES))

  num_files = ceil(float(count) / shard_size)
  for i in range(num_files):
      output = [shape_fn(*args) for _ in range(shard_size)]
      file_path = "%s/%s_%d_of_%d.npy" % (directory, file_prefix, i+1, num_files)
      np.save(file_path, output)
