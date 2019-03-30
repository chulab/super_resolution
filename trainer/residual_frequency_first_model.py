"""Defines model used to learn reconstruction."""

import argparse
import logging
from typing import Tuple

import tensorflow as tf
from utils import array_utils
from preprocessing import preprocess
from simulation import create_observation_spec

def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    learning_rate=0.1,
    observation_spec=None,
    frequency_scales=(1, 2, 4, 8),
    downsample_factor=3,
    angle_module_blocks=2,
    angle_module_kernel_size=(3,3),
  )


def downsampleBlock(x, depth_multiplier=2, kernel_size=(4, 4), stride=2):
  with tf.name_scope('downsample_block'):
    return tf.keras.layers.DepthwiseConv2D(
      kernel_size,
      strides=(stride, stride),
      padding="valid",
      use_bias=False,
      activation=None,
      depth_multiplier=depth_multiplier,
    ).apply(x)


def donwnsampleModule(input_shape, downsample_factor, **kwargs):
  """Defines downsampling module."""
  with tf.name_scope("downsample_module"):
    inputs = tf.keras.Input(shape=input_shape)

    net = inputs

    for _ in range(downsample_factor):
      net = downsampleBlock(net, **kwargs)

    net = tf.keras.layers.SeparableConv2D(
      72, kernel_size=(1, 1), padding="same").apply(net)

    return tf.keras.Model(inputs=inputs, outputs=net)


def upsampleBlock(x, filters=64, kernel_size=(4, 4), stride=2):
  """Upsamples spatial dimension of x by `stride`."""
  return tf.keras.layers.Conv2DTranspose(
    filters=filters,
    kernel_size=kernel_size,
    strides=(stride, stride),
    padding="valid",
    use_bias=False,
    activation=None,
  ).apply(x)


def spatial_block(x, scales, kernel_size=[3, 3]):
  convs = []
  for scale in scales:
    out = tf.keras.layers.SeparableConv2D(
      8,
      kernel_size=kernel_size,
      dilation_rate=(scale, scale),
      padding="same",
      activation=None,
    ).apply(x)
    convs.append(
      tf.keras.layers.LeakyReLU(alpha=.1).apply(out)
    )
  net = tf.keras.layers.Concatenate().apply(convs)
  return net


def resBlock(inputs, channels, kernel_size, scale=1.):
  res = tf.keras.layers.SeparableConv2D(
    channels,
    kernel_size,
    activation=None,
    padding="same"
  ).apply(inputs)
  res = tf.keras.layers.ReLU().apply(res)
  res = tf.keras.layers.SeparableConv2D(
    channels,
    kernel_size,
    activation=None,
    padding="same"
  ).apply(res)
  res = tf.keras.layers.Lambda(lambda x: x * scale).apply(res)
  return tf.keras.layers.Add().apply([inputs, res])


def frequencyModule(input_shape, scales, **kwargs):
  """Defines model for operating on frequencies."""
  with tf.name_scope("frequency_module"):
    inputs = tf.keras.Input(shape=input_shape)
    net = inputs
    net = spatial_block(net, scales, **kwargs)

    logging.info("net after spatial_block {}".format(net))

    net = tf.keras.Layers.SeparableConv2D(8, kernel_size=[3, 3]).apply(net)

    logging.info("net after channel reduction {}".format(net))

    for _ in range(3):
      net = resBlock(net, 8, [3,3], .1)

    return tf.keras.Model(inputs=inputs, outputs=net)


def angleModule(input_shape, channels, kernel_size, residual_blocks):
  with tf.name_scope("angle_module"):
    inputs = tf.keras.Input(shape=input_shape)

    net = inputs

    for i in range(residual_blocks):
      net = resBlock(net, channels=channels, kernel_size=kernel_size, scale=.1)
      print(net)

    return tf.keras.Model(inputs=inputs, outputs=net)


def network(inputs, params):
  # Split along `angle` axis.

  print(inputs)

  frequency_net = tf.keras.layers.Lambda(
    array_utils.reduce_split_tensor, arguments={'axis': 1}).apply(inputs)

  print("frequency_net {}".format(frequency_net))

  downsample_module = donwnsampleModule(
    input_shape=frequency_net[0].shape[1:],
    downsample_factor=params.downsample_factor)

  frequency_net = [downsample_module(input) for input in frequency_net]

  logging.info("frequency_net after downsampling {}".format(frequency_net))

  frequency_module = frequencyModule(
    input_shape=frequency_net[0].shape[1:],
    scales=params.frequency_scales,
  )

  # Frequency module
  frequency_net = [frequency_module(input) for input in frequency_net]

  print("frequency_net {}".format(frequency_net))

  # Rotate output so all features are co-registered.
  frequency_net = [
    tf.keras.layers.Lambda(tf.contrib.image.rotate, arguments={
      'angles': angle, 'interpolation': "BILINEAR"})(tensor)
    for tensor, angle in zip(frequency_net, params.observation_spec.angles)]

  print("frequency_net {}".format(frequency_net))

  net = tf.keras.layers.Concatenate().apply(frequency_net)

  logging.info('Net after concatenate freq modules {}'.format(net))

  # 64 Feature maps for residual layers.
  net = tf.keras.layers.SeparableConv2D(64, kernel_size=(1, 1),
                                        padding="same").apply(net)

  # Angle model.
  angle_module = angleModule(
    input_shape=net.shape[1:],
    channels=64,
    kernel_size=params.angle_module_kernel_size,
    residual_blocks=params.angle_module_blocks,
  )

  net = angle_module(net)

  logging.info("net after apply angle module {}".format(net))

  for i in range(params.downsample_factor):
    net = upsampleBlock(net, kernel_size=[4, 4], stride=2)

  logging.info("net after upsample {}".format(net))

  net = tf.keras.layers.SeparableConv2D(1, kernel_size=(1, 1),
                                        padding="same").apply(net)

  return net


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

  # Run observations through CNN.
  predictions = network(observations, params)[..., 0]

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
    eval_metric_ops = eval_metric_ops,
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
  example_shape: Tuple[int],
  observation_spec,
  distribution_blur_sigma,
  observation_blur_sigma,
):
  """Input functions for training residual_frequency_first_model."""
  fns =[]

  # Parse.
  fns.append(preprocess.parse())

  # Add shape
  fns.append(preprocess.set_shape(
    distribution_shape=example_shape,
    observation_shape=[len(observation_spec.angles)] + example_shape + [len(observation_spec.psf_descriptions)]))

  # Check for Nan.
  fns.append(preprocess.check_for_nan)

  # Blur
  fns.append(preprocess.blur(
    observation_spec, distribution_blur_sigma, observation_blur_sigma))

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
    distribution_blur_sigma=args.distribution_blur_sigma,
    observation_blur_sigma=args.observation_blur_sigma,
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
    '--distribution_blur_sigma',
    dest='distribution_blur_sigma',
    type=float,
    required=True,
  )

  parser.add_argument(
    '--observation_blur_sigma',
    dest='observation_blur_sigma',
    type=float,
    required=True,
  )

  args, _ = parser.parse_known_args()

  return args