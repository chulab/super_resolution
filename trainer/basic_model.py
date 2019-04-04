"""Defines model used to learn reconstruction."""

import argparse
import logging
from typing import Tuple

import tensorflow as tf

from simulation import tensor_utils

from preprocessing import preprocess
from simulation import create_observation_spec
from trainer import metrics
from utils import array_utils


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    learning_rate=0.001,
    observation_spec=None,
    spatial_blocks=1,
    spatial_scales=(1,),
    filters_per_scale=8,
    spatial_kernel_size=3,
    residual_blocks=1,
    residual_channels=32,
    residual_kernel_size=3,
    residual_scale=.1,
  )


def residual_block(inputs, channels, kernel_size, residual_scale):
  with tf.name_scope("residual_block"):
    res = tf.keras.layers.SeparableConv2D(
      channels,
      kernel_size,
      activation=None,
      use_bias=False,
      padding="same"
    ).apply(inputs)
    res = tf.keras.layers.ReLU().apply(res)
    res = tf.keras.layers.SeparableConv2D(
      channels,
      kernel_size,
      activation=None,
      use_bias=False,
      padding="same"
    ).apply(res)
    res = tf.keras.layers.Lambda(lambda x: x * residual_scale).apply(res)
    return tf.keras.layers.Add().apply([inputs, res])


def spatial_block(x, scales, filters_per_scale, kernel_size):
  convs = []
  for scale in scales:
    convs.append(
      tf.keras.layers.SeparableConv2D(
      filters=filters_per_scale,
      kernel_size=kernel_size,
      dilation_rate=(scale, scale),
      padding="same",
      activation=tf.nn.leaky_relu,
      ).apply(x))
  if len(scales) > 1:
    net = tf.keras.layers.Concatenate().apply(convs)
  else:
    net = convs[0]
  return net


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
    network = tf.keras.layers.Conv2D(
      filters=64,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(input_layer)

    for _ in range(params.spatial_blocks):
      network = spatial_block(
        network,
        scales=params.spatial_scales,
        filters_per_scale=params.filters_per_scale,
        kernel_size=params.spatial_kernel_size,
      )

    network = tf.keras.layers.SeparableConv2D(
      filters=params.residual_channels,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(input_layer)

    for _ in range(params.residual_blocks):
      network = residual_block(
        network,
        channels=params.residual_channels,
        kernel_size=params.residual_kernel_size,
        residual_scale=params.residual_scale,
      )

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
  logging.info("`observations` tensor recieved in model is "
                "{}".format(observations))
  distributions = labels
  logging.info("`distributions` tensor recieved in model is "
                "{}".format(distributions))

  distributions, observations = preprocess.hilbert(hilbert_axis=2)(distributions, observations)

  distributions = distributions[ ..., tf.newaxis]

  distributions = tf.keras.layers.AveragePooling2D(10).apply(distributions)

  angles = params.observation_spec.angles

  observations = array_utils.reduce_split_tensor(observations, 1)
  observation_pooling_layer = tf.keras.layers.AveragePooling2D(10)
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
    l2_loss = tf.reduce_sum((predictions - distributions) ** 2)
    loss = l2_loss

    # Add loss summary.
    tf.summary.scalar("loss", loss)

  with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
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

    # Add image summaries.
    tf.summary.image("observation", observations[..., 0, tf.newaxis], 1)
    tf.summary.image("distributions", distributions, 1)
    tf.summary.image("predictions", predictions, 1)


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

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

  args, _ = parser.parse_known_args()

  return args