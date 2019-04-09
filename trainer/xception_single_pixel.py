"""Defines model used to learn reconstruction."""

import argparse
import logging
from typing import Tuple

import tensorflow as tf

from simulation import tensor_utils

from preprocessing import preprocess
from simulation import create_observation_spec
from utils import array_utils


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    learning_rate=0.001,
    observation_pool_downsample=None,
    distribution_pool_downsample=10,
    predict_pixel=[1,1],
    use_hilbert=True,
  )


def network(input_layer, params):

  # reduce number of channels.
  network = tf.keras.layers.SeparableConv2D(
    filters=3,
    kernel_size=5,
    strides=1
  ).apply(input_layer)

  with tf.name_scope('xception'):
    xception_model = tf.keras.applications.xception.Xception(
      include_top=False,
      weights=None,
      input_tensor=network,
      pooling='max',
    )
    network = xception_model.output

  print("xception_model output {}".format(network))

  network = tf.keras.layers.Flatten().apply(network)

  network = tf.keras.layers.Dense(
    20,
    activation=tf.nn.leaky_relu,
    use_bias=True,
  ).apply(network)

  network = tf.keras.layers.Dense(
    1,
    activation=None,
    use_bias=True,
  ).apply(network)


  return network


def model_fn(features, labels, mode, params):
  """Defines model graph for super resolution.

  Args:
    features: dict containing:
      `images`: a `tf.Tensor` with shape `[batch_size, height, width, channels]`
    labels: dict containing:
      `distribution`: a `tf.Tensor` with shape `[batch_size, height, width]`
    mode: str. must be one of `tf.estimator.ModeKeys`.
    params: `tf.contrib.training.HParams` object containing hyperparameters for
      model.

  Returns:
    `tf.Estimator.EstimatorSpec` object.
  """
  observations = features
  logging.info("`observations` tensor recieved in model is "
                "{}".format(observations))
  distributions = labels
  logging.info("`distributions` tensor recieved in model is "
                "{}".format(distributions))

  if params.use_hilbert:
    distributions, observations = preprocess.hilbert(hilbert_axis=1)(distributions, observations)

  distributions = distributions[ ..., tf.newaxis]

  distributions = tf.keras.layers.AveragePooling2D(
    params.distribution_pool_downsample).apply(distributions)

  if params.observation_pool_downsample:
    observations = tf.keras.layers.AveragePooling2D(
      params.observation_pool_downsample).apply(observations)
    logging.info("observations after pooling {}".format(observations))

  obs_shape = observations.shape

  # Split along `channels` axis.
  observations = tf.split(
    observations,
    axis=-1,
    num_or_size_splits=len(params.observation_spec.angles)
  )

  logging.info("observations after split {}".format(observations))

  observations = [
    tf.keras.layers.Lambda(
      tf.contrib.image.rotate,
      arguments={'angles': -1 * angle, 'interpolation': "BILINEAR"},
      name="rotate_{}".format(angle))(tensor)
    for tensor, angle in zip(observations, params.observation_spec.angles)]
  logging.info("observations after rotation {}".format(observations))

  observations = tf.keras.layers.Concatenate(axis=-1).apply(observations)

  # Have to set shape as it is lost in `rotation`.
  observations.set_shape(obs_shape)

  tf.summary.image("observation", observations[..., 0, tf.newaxis], 1)
  tf.summary.image("distributions", distributions, 1)

  # Run observations through CNN.
  predictions = network(observations, params)

  logging.info("params.predict_pixel {}".format(params.predict_pixel))

  # Select pixel to predict.
  distributions = distributions[:, params.predict_pixel[0], params.predict_pixel[1], :]

  with tf.variable_scope("predictions"):
    predict_output = {
      "predictions": predictions,
      "observations": observations,
    }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predict_output
    )

  # Loss. Compare output of nn to original images.
  with tf.variable_scope("loss"):
    difference_squared = (predictions - distributions) ** 2
    l2_loss = tf.reduce_sum(difference_squared)
    loss = l2_loss

  with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(
      learning_rate=tf.train.exponential_decay(
        learning_rate=params.learning_rate,
        global_step=tf.train.get_global_step(),
        decay_steps=500,
        decay_rate=0.8,
        staircase=True,
      )
    )
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())

  with tf.variable_scope("metrics"):
    rms = tf.metrics.root_mean_squared_error(
      labels=distributions, predictions=predictions)

    # Compare RMS for averaged image at `prediction_pixel`.
    averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)
    averaged_observation_pixel = averaged_observation[:, params.predict_pixel[0], params.predict_pixel[1], :]

    averaged_rms = tf.metrics.root_mean_squared_error(
      labels=distributions, predictions=averaged_observation_pixel)

    eval_metric_ops = {
      "rms": rms,
      "averaged_rms": averaged_rms,
    }    # Add image summaries.

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
  )


def input_fns_(
  example_shape: Tuple[int, int],
  observation_spec,
):
  """Input functions for training residual_frequency_first_model."""
  fns =[]

  # Parse.
  fns.append(preprocess.parse())

  # Add shape
  fns.append(preprocess.set_shape(
    distribution_shape=example_shape,
    observation_shape=[len(observation_spec.angles)] + list(example_shape) + [len(observation_spec.psf_descriptions)]))

  # Check for Nan.
  fns.append(preprocess.check_for_nan)

  # Take a single frequency.
  fns.append(preprocess.select_frequency(0))

  fns.append(preprocess.swap)

  return fns


def input_fns():
  args = parse_args()

  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path, args.cloud_train
  )

  return input_fns_(
    example_shape=args.example_shape,
    observation_spec=observation_spec,
  )


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--example_shape',
    type=lambda s: [int(size) for size in s.split(',')],
    required=True,
  )

  parser.add_argument(
    '--observation_spec_path',
    type=str,
    required=True,
  )

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

  args, _ = parser.parse_known_args()

  return args