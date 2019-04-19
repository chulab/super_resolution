"""Utilities for dataset for online model."""

import tensorflow as tf


from training_data import shapes_dataset
from training_data import particle_dataset
from training_data import generate_scatterer_dataset


def dataset_params():
  return tf.contrib.training.HParams(
    dataset_type="CIRCLE",
    physical_dimension=0.001,
    grid_dimension=5e-6,
    min_radius=0,
    max_radius=1.5e-3,
    max_count=4,
    background_noise=0,
    lambda_multiplier=0.01
  )


def distribution_dataset(
    dataset_type,
    physical_dimension,
    grid_dimension,
    min_radius,
    max_radius,
    max_count,
    background_noise,
    lambda_multiplier,
    normalize_output=False,
    dataset_size=None,
):
  physical_size = [physical_dimension] * 2
  grid_dimensions = [grid_dimension] * 2

  # Make distribution generator.
  distribution_generator = shapes_dataset.shape_generator(
    type=dataset_type,
    dimensions=physical_size,
    grid_dimensions=grid_dimensions,
    min_radius=min_radius,
    max_radius=max_radius,
    min_intensity=0.,
    max_intensity=1.,
    max_count=max_count,
    background_noise=background_noise,
  )

  # Add poissonian noise.
  poisson_generator = particle_dataset.poisson_generator(
    distribution_generator,
    lambda_multiplier=lambda_multiplier,
    normalize_output=normalize_output
  )

  test_output = next(poisson_generator)

  dataset = tf.data.Dataset.from_generator(
    lambda: poisson_generator,
    output_types=tf.int32,
    output_shapes=test_output.shape,
  )

  dataset = dataset.prefetch(5)

  return dataset