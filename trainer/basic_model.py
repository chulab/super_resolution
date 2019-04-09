"""Defines model used to learn reconstruction."""

import argparse
import logging
from typing import Tuple
import numpy as np

import tensorflow as tf

from simulation import tensor_utils

from preprocessing import preprocess
from simulation import create_observation_spec
from trainer import metrics
from trainer import blocks
from utils import array_utils


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    learning_rate=0.001,
    observation_spec=None,
    conv_blocks=1,
    conv_block_kernel_size=5,
    conv_filters=72,
    spatial_blocks=1,
    spatial_scales=(1,),
    filters_per_scale=8,
    spatial_kernel_size=3,
    residual_blocks=1,
    residual_channels=32,
    residual_kernel_size=3,
    residual_scale=.1,
    pool_downsample=10,
  )


def network(input_layer, params):
  """Defines network.

  Args:
    `input_layer`: `tf.Tensor` node which outputs shapes `[b, h, w, c]`.
    These represent observations.
    training: Bool which sets whether network is in a training or evaluation/
      test mode. (Drop out is turned on during training but off during
      eval.)
  """
  logging.info("Before feeding model {}".format(input_layer))

  with tf.variable_scope("Model"):

    network = input_layer
    for _ in range(params.conv_blocks):
      network = tf.keras.layers.SeparableConv2D(
        filters=params.conv_filters,
        depth_multiplier=2,
        kernel_size=params.conv_block_kernel_size,
        padding="same",
        use_bias=True,
        activation=tf.nn.leaky_relu
      ).apply(network)

    for _ in range(params.spatial_blocks):
      network = blocks.spatial_block(
        network,
        scales=params.spatial_scales,
        filters_per_scale=params.filters_per_scale,
        kernel_size=params.spatial_kernel_size,
        activation=tf.nn.leaky_relu,
      )

    network = tf.keras.layers.SeparableConv2D(
      filters=params.residual_channels,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    for _ in range(params.residual_blocks):
      network = blocks.residual_block(
        network,
        channels=params.residual_channels,
        kernel_size=params.residual_kernel_size,
        residual_scale=params.residual_scale,
      )

    network = tf.keras.layers.Conv2D(
      filters=1,
      kernel_size=[1, 1],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu,
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

  distributions, observations = preprocess.hilbert(hilbert_axis=2)(distributions, observations)

  distributions = distributions[ ..., tf.newaxis]

  distributions = tf.keras.layers.AveragePooling2D(params.pool_downsample).apply(distributions)

  angles = params.observation_spec.angles

  observations = array_utils.reduce_split_tensor(observations, 1)
  observation_pooling_layer = tf.keras.layers.AveragePooling2D(params.pool_downsample)
  observations = [observation_pooling_layer.apply(o)[:, tf.newaxis]
                  for o in observations]

  observations = tf.keras.layers.Concatenate(axis=1).apply(observations)

  logging.info("observations after pooling {}".format(observations))

  observations = tensor_utils.rotate_tensor(
    observations,
    tf.convert_to_tensor([-1 * angle for angle in angles]),
    1
  )

  observations = tensor_utils.combine_batch_into_channels(observations, 0)[0]

  logging.info("observations after rotation {}".format(observations))

  # Run observations through CNN.
  predictions = network(observations, params)

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

    ssim = metrics.ssim(distributions, predictions, max_val=5)
    psnr = metrics.psnr(distributions, predictions, max_val=5)
    total_noise = metrics.total_variation(predictions)

    eval_metric_ops = {
      "rms": rms,
      "ssim": ssim,
      "psnr": psnr,
      "total_noise": total_noise,
    }

    # Average image along `channel` axis. This corresponds to previous SOA.
    averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

    # Add image summaries.
    tf.summary.image("observation", observations[..., 0, tf.newaxis], 1)
    tf.summary.image("averaged_observation", averaged_observation, 1)
    tf.summary.image("distributions", distributions, 1)
    tf.summary.image("predictions", predictions, 1)
    tf.summary.image("difference", difference_squared, 1)

  training_hooks = []

  # Report training failed if loss becomes Nan.
  training_hooks.append(tf.train.NanTensorHook(loss, fail_on_nan_loss=False))

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
    training_hooks=training_hooks,
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