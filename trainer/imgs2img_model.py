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
from trainer import imgs2img_transformer as imgs2img
from utils import array_utils
from tensor2tensor.layers import common_layers


def make_hparams(bit_depth, example_shape) -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  hparams = imgs2img.custom_img2img_transformer2d_tiny()
  vocab_size = 2 ** bit_depth
  p_hparams = imgs2img.super_reso_problem_hparams(vocab_size, vocab_size, hparams)
  hparams.add_hparam("bit_depth", bit_depth)
  hparams.add_hparam("example_shape", example_shape)
  hparams.add_hparam("problem", p_hparams)
  hparams.add_hparam("observation_pool_downsample", 1)
  hparams.add_hparam("distribution_pool_downsample", 1)
  hparams.add_hparam("decay_step", 500)
  hparams.add_hparam("decay_rate", .9)
  return hparams


def gpu_preprocess(observations, distributions, params):

  distributions, observations = preprocess.hilbert(hilbert_axis=2)(distributions, observations)

  distributions = distributions[ ..., tf.newaxis]
  distributions = tf.keras.layers.AveragePooling2D(
    params.distribution_pool_downsample).apply(distributions) * (
      params.distribution_pool_downsample ** 2)
  distributions = distributions[..., 0]

  angles = params.observation_spec.angles

  observations = tf.split(observations, observations.shape[-1], -1)

  observation_pooling_layer = tf.keras.layers.AveragePooling2D(
    params.observation_pool_downsample)
  observations = [
    observation_pooling_layer.apply(o) for o in observations]
  print("observations {}".format(observations))
  observations = [
    tf.contrib.image.rotate(tensor, -1 * ang, interpolation='BILINEAR')
    for tensor, ang in zip(observations, angles)
  ]
  observations = tf.keras.layers.Concatenate(axis=-1).apply(observations)
  observations.set_shape([None, distributions.shape[1], distributions.shape[2],
    len(params.observation_spec.angles)])

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

  distributions_values = loss_utils.quantize_tensor(
    distributions, params.bit_depth, 0., 4., False)
  distributions_quantized = tf.one_hot(distributions_values, 2 ** params.bit_depth)

  # Average image along `channel` axis. This corresponds to previous SOA.
  averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

  with tf.variable_scope("inputs"):
    # Add image summaries.
    for i, angle in enumerate(params.observation_spec.angles):
      tf.summary.image("obs_angle_{}".format(angle), observations[..., i, tf.newaxis], 1)
    tf.summary.image("averaged_observation", averaged_observation, 1)

  params.num_channels = common_layers.shape_list(observations)[-1]
  # params.num_channels = observations.shape[-1]
  network = imgs2img.Imgs2imgTransformer(params, mode, params.problem)
  features = {
    "inputs": observations,
    "targets": tf.expand_dims(distributions_values, -1),
    "target_space_id": tf.constant(1, dtype=tf.float32),
  }
  predictions_quantized, _ = network.apply(features)

  logging.info("predictions_quantized {}".format(predictions_quantized))
  logging.info("distributions_quantized {}".format(distributions_quantized))


  with tf.variable_scope("predictions"):
    predict_output = {
      "observations": observations,
    }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predict_output
    )

  with tf.variable_scope("predictions"):
    def _logit_to_class(logit):
      return tf.argmax(logit, -1)
    distribution_class = _logit_to_class(distributions_quantized)
    prediction_class = _logit_to_class(predictions_quantized)

    # Visualize output of predictions as categories.
    tf.summary.tensor_summary("prediction_class", prediction_class)

    # Log fraction nonzero.
    predicted_nonzero_count = tf.cast(tf.count_nonzero(prediction_class), tf.float32)
    true_nonzero_count = tf.cast(tf.count_nonzero(distribution_class), tf.float32)
    true_nonzero_fraction = true_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    nonzero_fraction = predicted_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    tf.summary.scalar("nonzero_fraction", nonzero_fraction)
    nonzero_hook = tf.train.LoggingTensorHook(
      tensors={
        "predicted_nonzero_fraction": nonzero_fraction,
        "true_nonzero_fraction": true_nonzero_fraction,
      },
      every_n_iter=50,
    )
    hooks.append(nonzero_hook)


    def _class_to_image(category):
      return tf.cast(category, tf.float32)[..., tf.newaxis]
    dist_image = _class_to_image(distribution_class)
    pred_image = _class_to_image(prediction_class)

    image_hook = tf.train.LoggingTensorHook(
      tensors={"distribution": dist_image[0, ..., 0],
               "prediction": pred_image[0, ..., 0],},
      every_n_iter=50,
    )
    hooks.append(image_hook)

    tf.summary.image("distributions", dist_image, 1)
    tf.summary.image("predictions", pred_image, 1)
    tf.summary.image("difference", (dist_image - pred_image) ** 2, 1)

  # Loss. Compare output of nn to original images.
  with tf.variable_scope("loss"):
    real_proportion = (tf.reduce_sum(
        distributions_quantized,
        axis=[0, 1, 2],
        keepdims=True,
        ) + 10) / (tf.cast(tf.size(distributions_quantized), tf.float32) + 10)
    proportional_weights = 1 / (
      tf.reduce_sum(
        (1 / real_proportion) * distributions_quantized,
        axis=-1)
    )
    proportion_hook = tf.train.LoggingTensorHook(
      tensors={"proportional_weights": proportional_weights[0],},
      every_n_iter=50,
    )
    hooks.append(proportion_hook)

    softmax_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=distributions_quantized,
      logits=predictions_quantized,
      weights=proportional_weights,
    )
    tf.summary.scalar("softmax_loss", softmax_loss)

    loss = softmax_loss

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

    batch_accuracy = tf.reduce_mean(
      tf.cast(tf.equal(distribution_class, prediction_class), tf.float32))
    tf.summary.scalar("batch_accuracy", batch_accuracy)


    accuracy_hook = tf.train.LoggingTensorHook(
      tensors={"batch_accuracy": batch_accuracy,},
      every_n_iter=50
    )
    hooks.append(accuracy_hook)

    accuracy = tf.metrics.accuracy(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    eval_metric_ops = {
      "accuracy": accuracy,
    }

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
    training_hooks=hooks,
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
