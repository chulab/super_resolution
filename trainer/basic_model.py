"""Defines model used to learn reconstruction."""

import argparse
import logging
from typing import Tuple

import tensorflow as tf

from simulation import tensor_utils

from preprocessing import preprocess
from simulation import create_observation_spec

def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    learning_rate=0.001,
    observation_spec=None,
  )


def network(input_layer, angles, training):
  """Defines network.

  Args:
    `input_layer`: `tf.Tensor` node which outputs shapes `[b, h, w, c]`.
    These represent observations.
    training: Bool which sets whether network is in a training or evaluation/
      test mode. (Drop out is turned on during training but off during
      eval.)
  """
  input_layer.shape.assert_is_compatible_with(
    [None, len(angles), None, None, None])

  network = tensor_utils.rotate_tensor(
    input_layer,
    tf.convert_to_tensor([-1 * angle for angle in angles]),
    1
  )

  network = tensor_utils.combine_batch_into_channels(network, 0)[0]

  logging.info("Before feeding model {}".format(network))

  with tf.variable_scope("Model"):
    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[6, 6],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.DepthwiseConv2D(
      filters=32,
      kernel_size=[10, 10],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.DepthwiseConv2D(
      filters=32,
      kernel_size=[10, 10],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.DepthwiseConv2D(
      filters=32,
      kernel_size=[20, 20],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.Conv2D(
      filters=1,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
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
  logging.debug("`observations` tensor recieved in model is "
                "{}".format(observations))
  distributions = labels
  logging.debug("`distributions` tensor recieved in model is "
                "{}".format(distributions))

  # Downsample.
  observations = tf.keras.layers.AveragePooling2D(10).apply(observations)
  distributions = tf.keras.layers.AveragePooling2D(10).apply(distributions)

  if mode == tf.estimator.ModeKeys.TRAIN:
    training = True
  else:
    training = False

  # Run observations through CNN.
  predictions = network(
    observations, params['observation_spec'].angles, training)[..., 0]

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
    l2_loss = tf.reduce_sum((predictions - distributions) ** 2)
    loss = l2_loss

    # Add loss summary.
    tf.summary.scalar("loss", loss)

  with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(params["learning_rate"])
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())

  with tf.variable_scope("metrics"):
    rms_error = tf.metrics.root_mean_squared_error(
      labels=distributions, predictions=predictions)

    # Add eval summary.
    tf.summary.scalar("rms_error", rms_error[0])

    eval_metric_ops = {
      "rms_error": rms_error,
    }

    # Add image summaries.
    tf.summary.image("observation", observations[:, 0, ..., 0, tf.newaxis], 1)
    tf.summary.image("distributions", distributions[..., tf.newaxis], 1)
    tf.summary.image("predictions", predictions[..., tf.newaxis], 1)


  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops = eval_metric_ops
  )


def build_estimator(
    config,
    params,
):
  return tf.estimator.Estimator(
    model_fn=model_fn,
    config=config,
    params=params,
  )

def input_fns_(
  example_shape: Tuple[int, int],
  observation_spec,
  pool_downsample,
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

  fns.append(preprocess.hilbert(hilbert_axis=1))

  fns.append(preprocess.pool_downsample(
    distribution_pool_size=pool_downsample,
    observation_pool_size=pool_downsample,))

  fns.append(preprocess.swap)

  return fns


def input_fns():
  args = parse_args()

  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path, False
  )

  return input_fns_(
    example_shape=args.example_shape,
    observation_spec=observation_spec,
    pool_downsample=args.pool_downsample,
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

  parser.add_argument(
    '--pool_downsample',
    type=int,
    required=True,
  )

  args, _ = parser.parse_known_args()

  return args