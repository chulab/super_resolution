"""Defines model used to learn reconstruction."""

import argparse
import logging
from typing import Tuple
import math

import tensorflow as tf

from simulation import tensor_utils

from preprocessing import preprocess
from simulation import create_observation_spec
from trainer import blocks
from trainer import loss_utils
from trainer import imgs2img_continuous as imgs2img
from utils import array_utils
from tensor2tensor.layers import common_layers
from analysis import plot_utils
from preprocessing import input


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  # hparams = imgs2img.custom_img2img_transformer2d_tiny()
  hparams = imgs2img.custom_img2img_transformer2d_base()
  hparams.num_decoder_layers = 4
  # hparams.dropout = 0.0
  # hparams.dropout = 0.3
  # hparams.attention_dropout = 0.3
  # hparams.relu_dropout = 0.3
  hparams.add_hparam("bit_depth", 4)
  hparams.add_hparam("example_shape", 501)
  hparams.add_hparam("observation_pool_downsample", 1)
  hparams.add_hparam("distribution_pool_downsample", 1)
  hparams.add_hparam("bets", "None")
  hparams.add_hparam("rewards", "None")
  hparams.add_hparam("scale_steps", 5000)
  hparams.add_hparam("decay_step", 500)
  hparams.add_hparam("decay_rate", .9)
  return hparams


def gpu_preprocess(observations, distributions, params):

  distributions, observations = preprocess.hilbert(hilbert_axis=3)(distributions, observations)

  num_angles = len(params.observation_spec.angles)
  num_freqs = len(params.frequency_indices)

  distributions = distributions[ ..., tf.newaxis]
  distributions = tf.keras.layers.AveragePooling2D(
    params.distribution_pool_downsample).apply(distributions) * (
      params.distribution_pool_downsample ** 2)
  distributions = distributions[..., 0]

  angles = params.observation_spec.angles

  observation_pooling_layer = tf.keras.layers.AveragePooling2D(
    params.observation_pool_downsample)

  storage = []
  for freqs, ang in zip(tf.split(observations, observations.shape[1], 1), angles):
    pooled = observation_pooling_layer.apply(tf.squeeze(freqs, 1))
    height = int(pooled.shape[1])
    width = int(pooled.shape[2])
    rotated = tf.contrib.image.rotate(pooled, -1 * ang, interpolation='BILINEAR')
    storage.append(rotated)

  observations = tf.keras.layers.Concatenate(axis=-2).apply(storage)
  # observations = tf.reshape(observations, shape=[None, height,
  #   width, num_angles * num_freqs])
  # observations = tf.split(observations, observations.shape[-1], -1)
  #
  # observation_pooling_layer = tf.keras.layers.AveragePooling2D(
  #   params.observation_pool_downsample)
  # observations = [
  #   observation_pooling_layer.apply(o) for o in observations]
  # observations = tf.keras.layers.Concatenate(axis=-1).apply(observations)
  #
  # # length = int(params.example_shape / params.observation_pool_downsample)
  # observations = tf.reshape(observations, shape=[None, height, width, num_freqs, num_angles])
  #
  # observations = tf.split(observations, observations.shape[-1], -1)
  # observations = [
  #   tf.contrib.image.rotate(tf.squeeze(tensor, axis=-1), -1 * ang,
  #   interpolation='BILINEAR') for tensor, ang in zip(observations, angles)
  # ]
  #
  # observations = tf.keras.layers.Concatenate(axis=-1).apply(observations)
  observations.set_shape([None, height, width * num_angles, num_freqs])
  print("observations {}".format(observations))

  # distributions, observations = preprocess.hilbert(hilbert_axis=2)(distributions, observations)
  #
  # distributions = distributions[ ..., tf.newaxis]
  # distributions = tf.keras.layers.AveragePooling2D(
  #   params.distribution_pool_downsample).apply(distributions) * (
  #     params.distribution_pool_downsample ** 2)
  # distributions = distributions[..., 0]
  #
  # angles = params.observation_spec.angles
  #
  # observations = tf.split(observations, observations.shape[-1], -1)
  #
  # observation_pooling_layer = tf.keras.layers.AveragePooling2D(
  #   params.observation_pool_downsample)
  # observations = [
  #   observation_pooling_layer.apply(o) for o in observations]
  # print("observations {}".format(observations))
  #
  # height = observations[0].shape[1]
  # width = observations[0].shape[2]
  #
  # observations = [
  #   tf.contrib.image.rotate(tensor, -1 * ang, interpolation='BILINEAR')
  #   for tensor, ang in zip(observations, angles)
  # ]
  # observations = tf.keras.layers.Concatenate(axis=-1).apply(observations)
  # observations.set_shape([None, height, width, len(angles)])

  return observations, distributions


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
  hooks = []

  observations = features
  logging.info("`observations` tensor recieved in model is "
                "{}".format(observations))
  distributions = labels
  logging.info("`distributions` tensor recieved in model is "
                "{}".format(distributions))

  tf.summary.image("original_distribution", distributions[..., tf.newaxis], 1)

  observations, distributions = gpu_preprocess(observations, distributions, params)

  logging.info("`observations` tensor after gpu preprocess in model is "
                "{}".format(observations))
  logging.info("`distributions` tensor  after gpu preprocess in model is "
                "{}".format(distributions))

  # observations_quantized = loss_utils.quantize_tensor(
  #   observations, 64, 0., 25., False)
  # distributions_values = loss_utils.quantize_tensor(
  #   distributions, 2 ** params.bit_depth, 0., 2 ** params.bit_depth, False)
  # distributions_quantized = tf.one_hot(distributions_values, 2 ** params.bit_depth)

  distributions = tf.expand_dims(distributions, -1)

  observations_hook = tf.train.LoggingTensorHook(
      tensors={
        "obs_max": tf.math.reduce_max(observations),
        "obs_min": tf.math.reduce_min(observations),
        "dis_max": tf.math.reduce_max(distributions),
        "dis_min": tf.math.reduce_min(distributions)
      },
      every_n_iter=50,
  )
  hooks.append(observations_hook)

  # Average image along `channel` axis. This corresponds to previous SOA.
  averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

  # with tf.variable_scope("inputs"):
    # Add image summaries.
    # for i, angle in enumerate(params.observation_spec.angles):
    #   tf.summary.image("obs_angle_{}".format(angle), observations[..., i, tf.newaxis], 1)
    # tf.summary.image("averaged_observation", averaged_observation, 1)

  params.num_channels = common_layers.shape_list(observations)[-1]
  # params.num_channels = observations.shape[-1]
  network = imgs2img.Imgs2imgTransformer(params, mode)
  features_t = {
    "inputs": observations,
    "targets": distributions,
    "target_space_id": tf.constant(1, dtype=tf.float32),
  }

  predictions, _ = network.apply(features_t)

  # input_output_hook = tf.train.LoggingTensorHook(
  #     tensors={
  #       "inputs": observations_quantized,
  #       "targets": tf.expand_dims(distributions_values, -1),
  #       "predictions": predictions_quantized
  #     },
  #     every_n_iter=5,
  # )
  # hooks.append(input_output_hook)

  logging.info("predictions {}".format(predictions))
  logging.info("distributions {}".format(distributions))


  with tf.variable_scope("predictions"):
    image_hook = tf.train.LoggingTensorHook(
      tensors={"distribution": distributions[0, ..., 0],
               "prediction": predictions[0, ..., 0],},
      every_n_iter=50,
    )
    hooks.append(image_hook)

    tf.summary.image("distributions", distributions, 1)
    tf.summary.image("predictions", predictions, 1)
    tf.summary.image("difference", (distributions - predictions) ** 2, 1)

    # Visualize output of predictions as categories.
    dist_summary = tf.summary.tensor_summary("distribution_tensor", distributions)
    pred_summary = tf.summary.tensor_summary("predictions_tensor", predictions)
    diff_summary = tf.summary.tensor_summary("difference_tensor", distributions
      - predictions)

    predict_output = {
        "predictions": predictions
    }

  # Loss. Compare output of nn to original images.
  with tf.variable_scope("loss"):

    # proportion = (tf.reduce_sum(
    #     distributions_quantized,
    #     axis=[0, 1, 2],
    #     keepdims=True,
    #     ) + 2 ** params.bit_depth) / (tf.reduce_sum(distributions_quantized) + 2 ** params.bit_depth)

    # proportion = (tf.reduce_sum(
    #     distributions_quantized,
    #     axis=[0, 1, 2],
    #     keepdims=True,
    #     ) + 2 ** params.bit_depth)
    # inv_proportion = 1 / proportion
    #
    # ones_like = tf.cast(tf.ones_like(prediction_class), tf.float32)
    #
    # def bets_and_rewards_fn(params):
    #   if params.rewards == "1/n":
    #     rewards = tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1)
    #   elif params.rewards == "-logn":
    #     rewards = - 1 * tf.reduce_sum(tf.math.log(proportion) * distributions_quantized,
    #       axis=-1)
    #   elif params.rewards == "log1/n":
    #     rewards = tf.reduce_sum(tf.math.log(inv_proportion) * distributions_quantized,
    #       axis=-1)
    #   elif params.rewards == "1/sqrtn":
    #     rewards = tf.math.sqrt(tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1))
    #   else:
    #     rewards = ones_like
    #
    #   one_hot_predictions = tf.one_hot(prediction_class, 2 ** params.bit_depth)
    #
    #   if params.bets == "1/n":
    #     bets= tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1)
    #   elif params.bets== "-logn":
    #     bets = - 1 * tf.reduce_sum(tf.math.log(proportion) * one_hot_predictions,
    #       axis=-1)
    #   elif params.bets == "log1/n":
    #     bets = tf.reduce_sum(tf.math.log(inv_proportion) * one_hot_predictions,
    #       axis=-1)
    #   elif params.bets == "1/sqrtn":
    #     bets = tf.math.sqrt(tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1))
    #   else:
    #     bets = ones_like
    #
    #   return bets, rewards
    #
    #
    # less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
    # bets, rewards = tf.cond(less_equal, lambda: bets_and_rewards_fn(params),
    #   lambda: (ones_like, ones_like))
    # lr = tf.cond(less_equal, lambda: tf.constant(params.learning_rate),
    #   lambda: tf.constant(params.learning_rate / 1000))
    #
    # proportional_weights = bets * rewards
    #
    # # proportional_weights *= tf.cast(tf.math.abs(distribution_class - prediction_class), tf.float32)
    #
    # proportion_hook = tf.train.LoggingTensorHook(
    #   tensors={"proportional_weights": proportional_weights[0], "log_inv": tf.math.log(inv_proportion)},
    #   every_n_iter=50,
    # )
    # hooks.append(proportion_hook)
    #
    # softmax_loss = tf.losses.softmax_cross_entropy(
    #   onehot_labels=distributions_quantized,
    #   logits=predictions_quantized,
    #   weights=proportional_weights
    # )


    # tf.summary.scalar("softmax_loss", softmax_loss)

    loss = tf.losses.mean_squared_error(distributions, predictions)

  with tf.variable_scope("optimizer"):
    learning_rate = tf.train.exponential_decay(
      learning_rate=params.learning_rate,
      global_step=tf.train.get_global_step(),
      decay_steps=params.decay_step,
      decay_rate=params.decay_rate,
      staircase=False,
    )
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())


  with tf.variable_scope("metrics"):
  #
  #   batch_accuracy = tf.reduce_mean(
  #     tf.cast(tf.equal(distribution_class, prediction_class), tf.float32))
  #   tf.summary.scalar("batch_accuracy", batch_accuracy)
  #
  #
  #   accuracy_hook = tf.train.LoggingTensorHook(
  #     tensors={"batch_accuracy": batch_accuracy,},
  #     every_n_iter=50
  #   )
  #   hooks.append(accuracy_hook)
  #
  #   accuracy = tf.metrics.accuracy(
  #     labels=tf.argmax(distributions_quantized, -1),
  #     predictions=tf.argmax(predictions_quantized, -1)
  #   )
  #
    eval_metric_ops = {
      "mse": tf.metrics.mean_squared_error(distributions, predictions)
    }

  merged = tf.summary.merge([dist_summary, pred_summary, diff_summary])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
    training_hooks=hooks,
    evaluation_hooks=[tf.train.SummarySaverHook(save_steps=1, output_dir= params.job_dir + "/eval", summary_op = merged)]
  )


def input_fns_(
  example_shape: Tuple[int, int],
  observation_spec,
  frequency_indices,
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

  fns.append(preprocess.select_frequencies(frequency_indices))

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
    frequency_indices=args.frequency_indices
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
    '--frequency_indices',
    type=lambda s: [int(index) for index in s.split(',')],
    required=True,
    default='0'
  )

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

  args, _ = parser.parse_known_args()

  return args
