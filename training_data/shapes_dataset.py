"""Functions to generate dataset (in shards) for certain geometric shapes"""

import math
import os

import numpy as np

from simulation import response_functions

from typing import List


_CIRCLE = 'CIRCLE'
VALID_TYPES = [_CIRCLE]


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


def shape_generator(
  type: str,
  *args,
  **kwargs,
):
  """Generator for shapes of a given type.

  Args:
    type: Shape to generate in data.
    args: Parameters to pass into the corresponding shape function.
    kwargs: Same as `args`.
  """
  if (type == _CIRCLE):
      shape_fn = random_circles
  else:
      raise ValueError("type must be in %s" % str(VALID_TYPES))
  while True:
    yield shape_fn(*args, **kwargs)

def save_dataset(
  generator,
  count,
  shard_size,
  save_directory,
  file_prefix,
):
  """Saves datset of elements from generator.

  This function builds and saves a dataset of scatterer distributions given
  a generator which produces np.ndarrays contining scatterer distributions.

  Each shard contains a single np.ndarrays with shape
  `[shard_size] + generator_output_shape`. This implies that the generator must
  return a consistent array size.

  Args:
    generator: Generator which returns np.ndarray objects.
    count: Total number of distributions in dataset.
    shard_size: Number of distributions per shard.
    save_directory: Path to folder where data will be saved.
    file_prefix: Optional prefix to add to filenames.

  Raises:
    ValueError if `save_directory` is not valid.
  """
  if not os.path.isdir(save_directory):
    raise ValueError("`save_directory` must be a valid directory.")

  file_count = int(math.ceil(float(count) / shard_size))
  for shard in range(1, file_count + 1):
    output = np.stack([next(generator) for _ in range(shard_size)], 0)
    filename = "%s_%d_of_%d.npy" % (file_prefix, shard, file_count)
    output_file = os.path.join(save_directory, filename)
    np.save(output_file, output)
