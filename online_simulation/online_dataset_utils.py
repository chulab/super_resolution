"""Utilities for dataset for online model."""


import tensorflow as tf
from tensorflow import contrib
autograph = contrib.autograph

from simulation import response_functions


def dataset_params():
  return tf.contrib.training.HParams(
    dataset_type="CIRCLE",
    physical_dimension=0.001,
    grid_dimension=5e-6,
    min_radius=0,
    max_radius=1.5e-3,
    max_count=4,
    background_noise=0,
    scatterer_density=1.e9,
    db=10.,
  )


def _circle(
    coordinates: tf.Tensor,
    origin: tf.Tensor,
    radius: tf.Tensor,
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
  if origin.shape.as_list()[0] != coordinates.shape.as_list()[-1]:
    raise ValueError("`origin` and `coordinates` must have same number of"
                     "dimensions but got {} and {}".format(
      origin.shape.as_list()[0], coordinates.shape.as_list()[-1]))

  distance_to_origin_squared = tf.reduce_sum((coordinates - origin) ** 2, -1)
  return tf.where(distance_to_origin_squared <= radius ** 2,
                  tf.ones_like(distance_to_origin_squared),
                  tf.zeros_like(distance_to_origin_squared))


@autograph.convert()
def random_circles(
    coordinates,
    physical_dimensions,
    min_radius: float,
    max_radius: float,
    min_intensity: float,
    max_intensity: float,
    max_count: int,
    background_noise: float = 0.,
):
  """Produces an array of balls in arbitrary dimensions.

  In the 2D mode this function produces an array with circles centered at
  random locations within the array. These circles will have a radius between
  `min_radius` and `max_radius`.

  Args:
    dimensions: Dimensions of box in metres.
    grid_unit: Grid unit in metres.
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
  # `box` will store circles.
  box = tf.ones(coordinates.shape.as_list()[:2]) * background_noise

  # Array will contain at least one ball, and up to `max_count`.
  count = tf.random.uniform([1], 2, max_count + 1, dtype=tf.int32)[0]

  origins = tf.stack(
    [tf.random.uniform([count], 0, length, dtype=tf.float32) for length in
     physical_dimensions], 1)

  for i in tf.range(count):
    # Get random circle origin.
    origin = origins[i]

    # Random radius.
    radius = tf.random.uniform([1], min_radius, max_radius, dtype=tf.float32)

    # Intensity.
    intensity = tf.random.uniform([1], min_intensity, max_intensity,
                                  dtype=tf.float32)
    # Add circle.
    box += tf.cast(_circle(coordinates, origin, radius),
                   tf.float32) * intensity

  return box


def poisson_noise(
    tensor: tf.Tensor,
    lambda_multiplier: float,
):
  """See documentation for `training_data.poisson_noise`."""
  return tf.random.poisson(
    lam=tensor * lambda_multiplier,
    shape=[1]
  )[0]


def _poisson_dataset_map(
    probability_distribution,
    lambda_multiplier,
):
  """Returns dictionary of original probability and noisy distribution."""
  scatterer_distribution = poisson_noise(probability_distribution,
                                         lambda_multiplier)
  return {
    "probability_distribution": probability_distribution,
    "scatterer_distribution": scatterer_distribution,
  }


def density_to_lambda(
    density: float,
    grid_unit: float,
):
  """Converts a `density` with units `particles/m**2` to `lambda`.

  The `lambda` value produced by this function is used to parametrized the
  poissonian process of `poisson_noise`.

  Args:
    density: density in N(particles)/meter^2
    grid_unit: grid_unit in meters.

  Returns:
    lambda: float.
  """
  return density * (grid_unit ** 2)


def normalize(
    tensor,
):
  return tensor / tf.reduce_max(tensor)


def db_scale(
  tensor,
  db=10,
):
  """Scales `tensor` so that values are compressed into a 10db range.

  Example:
    with default 10db:
      # tensor == (0, .5, 1)
      tensor_db = db_scale(tensor)
      # tensor_db == (.1, .316, 1.)

  args:
    tensor: `tf.Tensor` with values to be scaled. It is assumed that values are
    in the range (0, 1).
    db: decibel range of output.

  Returns:
    `tf.Tensor` of same shape as `tensor` containing scaled values.
  """
  scaled = 10 ** (tensor * db / 10)
  return normalize(scaled)


def random_circles_dataset(
    physical_dimension,
    grid_unit,
    min_radius,
    max_radius,
    max_count,
    scatterer_density,
    db,
):
  physical_dimensions = [physical_dimension] * 2

  coordinates = tf.cast(tf.stack(response_functions.coordinate_grid(
      physical_dimensions, [grid_unit] * 2, False),
    -1), tf.float32)[:-1, :-1]

  dummy_dataset = tf.data.Dataset.from_tensors(0).repeat(None)

  # Dataset consists of probability distributions (representing structures).
  dataset = dummy_dataset.map(lambda _: random_circles(
    coordinates=coordinates,
    physical_dimensions=physical_dimensions,
    min_radius=min_radius,
    max_radius=max_radius,
    min_intensity=0.,
    max_intensity=1.,
    max_count=max_count,
    background_noise=0.,
  ), num_parallel_calls=-1)

  # Normalize probability distributions. This means that the different features
  # will have a constant relative probability (p1/p2) but the absolute values of
  # all probabilities will be scaled so the maximum probability is 1.
  dataset = dataset.map(normalize, num_parallel_calls=-1)

  dataset = dataset.map(lambda tensor: db_scale(tensor, db), num_parallel_calls=-1)

  lambda_multiplier = density_to_lambda(scatterer_density, grid_unit)

  # Add poisson noise.
  dataset = dataset.map(lambda tensor: _poisson_dataset_map(
    probability_distribution=tensor, lambda_multiplier=lambda_multiplier
  ), num_parallel_calls=-1)

  dataset = dataset.prefetch(1)

  return dataset
