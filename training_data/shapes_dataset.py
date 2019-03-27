"""Functions to generate dataset (in shards) for certain geometric shapes"""

import argparse

import numpy as np

from simulation import response_functions

from typing import List

from utils.array_utils import normalize_along_axis


_CIRCLE = 'CIRCLE'
_LINE = 'LINE'
VALID_TYPES = [_CIRCLE, _LINE]


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
    np.ndarray with same shape as `coordinates[:-1]`, i.e. `coordinates`
    minus the last axis.

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
  background_noise: float=0.,
):
  """Produces an array of balls in arbitrary dimensions.

  In the 2D mode this function produces an array with circles centered at
  random locations within the array. These circles will have a radius between
  `min_radius` and `max_radius`.

  Args:
    dimensions: Dimensions of box in metres.
    grid_dimensions: Grid size in metres.
    min_radius: Minimum circle radius in metres.
    max_radius: Maximum circle radius in metres.
    min_intensity: Min intensity of a single circle.
    max_intensity: Max intensity of a single circle and overall clipping max.
    max_count: Maximum number of circles to place in array.
    background_noise: Fraction (between 0 and 1.) representing background noise.

  Returns:
    `np.ndarray` with total dimensions num_dim with each dimension having
    size `physical_dim / grid_size`.
  """
  coordinates = np.stack(
    response_functions.coordinate_grid(dimensions, grid_dimensions, False),
    -1
  )

  # `box` will store circles.
  box = np.ones(coordinates.shape[:-1]) * background_noise

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

  return np.clip(box, a_min=0., a_max=max_intensity)


def _line(
    coordinates: np.ndarray,
    origin: List[float],
    gradient: List[float],
    radius: float,
):
    """Returns `np.ndarray` with a line of `gradient` through `origin`
    with thickness given by `radius`.

    Set cells are `1` while empty cells are `0`.

    Args:
      coordinates: Array containing coordinates in n-d space.
      origin: Point on line in physical n-d space.
      gradient: Gradient of line expressed as a n-d vector.
      radius: Radius of line in physical units.

    Returns:
        np.ndarray with same shape as `coordinates[:-1]`, i.e. `coordinates`
        minus the last axis.

    Raises:
        ValueError: If `coordinates` and `origin` are not compatible.
    """
    if len(origin) != coordinates.shape[-1]:
      raise ValueError("`origin` and `coordinates` must have same number of"
                       "dimensions but got {} and {}".format(
        len(origin), coordinates.shape[-1]))

    # compute relative position vectors
    relative_positions = coordinates - origin

    # store normalized gradient for efficiency
    normalized_grad = normalize_along_axis(gradient)

    # find projection of position vectors orthogonal to line defined by gradient
    ortho_projector = lambda v: v - np.dot(v, normalized_grad) * normalized_grad
    ortho_proj = np.apply_along_axis(ortho_projector, -1, relative_positions)
    squared_ortho_distance = np.sum(ortho_proj ** 2, -1)

    return squared_ortho_distance <= radius ** 2


def random_lines(
  dimensions: List[float],
  grid_dimensions: List[float],
  min_radius: float,
  max_radius: float,
  min_intensity:float,
  max_intensity:float,
  max_count: int,
  background_noise: float=0.,
):
    """Produces an array of lines in arbitrary dimensions.

    In the 2D mode this function produces an array with lines passing through
    random locations with random gradients within the array. These lines will
    have a radius between `min_radius` and `max_radius`.

    Args:
      dimensions: Dimensions of box in metres.
      grid_dimensions: Grid size in metres.
      min_radius: Minimum line radius in metres.
      max_radius: Maximum line radius in metres.
      min_intensity: Min intensity of a single line.
      max_intensity: Max intensity of a single line and overall clipping max.
      max_count: Maximum number of lines to place in array.
      background_noise: Fraction (between 0 and 1.) representing background noise.

    Returns:
      `np.ndarray` with total dimensions num_dim with each dimension having
      size `physical_dim / grid_size`.
    """
    coordinates = np.stack(
      response_functions.coordinate_grid(dimensions, grid_dimensions, False),
      -1
    )

    # `box` will store lines.
    box = np.ones(coordinates.shape[:-1]) * background_noise

    # Array will contain at least one line, and up to `max_count`.
    count = np.random.randint(1, max_count + 1)

    for _ in range(count):
      # Get random point on line.
      origin = [np.random.uniform(0, length) for length in dimensions]

      # Random gradient.
      grad = [np.random.uniform(0, 1) for _ in dimensions]

      # Random radius.
      radius = np.random.uniform(min_radius, max_radius)

      # Random intensity.
      intensity = np.random.uniform(min_intensity, max_intensity)

      # Add line.
      box += _line(coordinates, origin, grad, radius) * intensity

    return np.clip(box, a_min=0., a_max=max_intensity)


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
  elif (type == _LINE):
      shape_fn = random_lines
  else:
      raise ValueError("type must be in %s" % str(VALID_TYPES))
  while True:
    yield shape_fn(*args, **kwargs)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--min_radius', dest='min_radius',
                      help='minimum radius.',
                      type=float, required=False)

  parser.add_argument('--max_radius', dest='max_radius',
                      help='Maximum radius.',
                      type=float, required=False)

  parser.add_argument('--max_count', dest='max_count',
                      help='maximum number of shapes', type=int, required=True)

  parser.add_argument('--background_noise', dest='background_noise',
                      help='Background noise fraction', type=float,
                      required=False)

  args, unknown = parser.parse_known_args()

  return args


def generate_shapes(
    type,
    physical_size,
    grid_dimensions,
):
  args = parse_args()

  return shape_generator(
    type=type,
    dimensions=physical_size,
    grid_dimensions=grid_dimensions,
    min_radius=args.min_radius,
    max_radius=args.max_radius,
    min_intensity=0.,
    max_intensity=1.,
    max_count=args.max_count,
    background_noise=args.background_noise,
  )
