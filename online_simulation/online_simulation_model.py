"""Model that performs simulation of US images online."""

import tensorflow as tf

import logging

from typing import List

from online_simulation import online_simulation_utils
from trainer import loss_utils
from preprocessing import preprocess
from preprocessing import signals
from utils import array_utils


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    bit_depth=2,
    learning_rate=0.001,
    decay_step=500,
    decay_rate=.9,
    psfs=None,
    log_steps=100,
  )


def entry_flow(input_layer):
  """Begining of CNN."""
  network = input_layer
  network = tf.keras.layers.Conv2D(
      filters=160,
      kernel_size=[3, 3],
      padding="same",
      use_bias=True,
      activation=tf.nn.leaky_relu
    ).apply(network)
  network = tf.keras.layers.BatchNormalization().apply(network)
  network = tf.keras.layers.Conv2D(
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      use_bias=True,
      activation=tf.nn.leaky_relu,
      strides=2,
    ).apply(network)
  network = tf.keras.layers.BatchNormalization().apply(network)
  return network


def _conv_block(
    input_layer,
    filters=64,
    kernel_size=[3, 3],
):
  network = input_layer
  network_res = tf.keras.layers.SeparableConv2D(
    filters=filters,
    depth_multiplier=1,
    kernel_size=kernel_size,
    padding="same",
    use_bias=True,
    activation=tf.nn.leaky_relu
  ).apply(network)
  network_res = tf.keras.layers.BatchNormalization().apply(network_res)
  network_res = tf.keras.layers.SeparableConv2D(
    filters=filters,
    depth_multiplier=1,
    kernel_size=kernel_size,
    padding="same",
    use_bias=True,
    activation=tf.nn.leaky_relu
  ).apply(network_res)
  network_res = tf.keras.layers.BatchNormalization().apply(network_res)
  network = tf.keras.layers.add([network, network_res])
  return network


def downsample_block(
    input_layer,
    depthwise_kernel_size=[3, 3],
    depth_multiplier=1,
):
  network = tf.keras.layers.DepthwiseConv2D(
    depth_multiplier=depth_multiplier,
    kernel_size=depthwise_kernel_size,
    dilation_rate=1,
    padding="same",
    strides=2,
  ).apply(input_layer)
  network = tf.keras.layers.BatchNormalization().apply(network)
  return network


def _upsample_block(
    input_layer,
    depthwise_kernel_size=[3, 3],
    filters=64,
):
  network = tf.keras.layers.Conv2DTranspose(
    filters=filters,
    kernel_size=depthwise_kernel_size,
    dilation_rate=2,
    padding="same",
    strides=1,
  ).apply(input_layer)
  network = tf.keras.layers.BatchNormalization().apply(network)
  return network


def network(input_layer, training):
  """Definition of network.

  Args:
    `input_layer`: `tf.Tensor` node which outputs shapes `[b, h, w, c]`.
    These represent observations.
    training: Bool which sets whether network is in a training or evaluation/
      test mode. (Drop out is turned on during training but off during
      eval.)
  """
  with tf.name_scope("model"):
    network = input_layer

    network = entry_flow(network)

    for i in range(4):
      network = _conv_block(network)
      network = downsample_block(network)

    network = _conv_block(network)

    for i in range(1):
        network = _upsample_block(network)

    network = tf.layers.dropout(network, training=training)

    return network


def downsample_by_pool(tensor, kernel_size):
  pool_layer = tf.keras.layers.AveragePooling2D(kernel_size, padding='same')
  return pool_layer.apply(tensor)


def average_observation(
    observations: List,
    angles: List,
):
  """Generates averaged observation by averaging hilbert transforms."""
  envelopes = []
  for o, a in zip(observations, angles):
    assert o.shape.ndims == 2
    o_temp = tf.contrib.image.rotate(o, -1 * a)
    o_temp = tf.cast(tf.abs(signals.hilbert(o_temp, 0)), tf.float32)
    o_temp = tf.contrib.image.rotate(o_temp, a)
    envelopes.append(o_temp)
  envelopes = tf.stack(envelopes, 0)
  return tf.reduce_mean(envelopes, 0)


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
  train_hooks = []
  eval_hooks = []

  with tf.name_scope("distributions"):
    full_resolution_scatterer = features['scatterer_distribution']
    full_resolution_probability = features['probability_distribution']

    def downsample_distribution(d, downsample):
      d = d[tf.newaxis, ..., tf.newaxis]
      d = downsample_by_pool(d, downsample)
      return d[..., 0]

    DOWNSAMPLE = 16
    downsample_scatterer = downsample_distribution(
      full_resolution_scatterer, DOWNSAMPLE)
    downsample_probability = downsample_distribution(
      full_resolution_probability, DOWNSAMPLE)

    # # NORMALIZE DISTRIBUTIONS.
    # distributions_normalized = preprocess.per_tensor_scale(distributions, 0., float(2 ** params.bit_depth))
    # logging.info("normalized distribution {}".format(distributions_normalized))

    # distributions_quantized = loss_utils.quantize_tensor(
    #   distributions_normalized, 2 ** params.bit_depth, 0., 2 ** params.bit_depth)
    # distributions_quantized = loss_utils.quantize_tensor(
    #   distributions, 2 ** params.bit_depth, 0., 2 ** params.bit_depth)
    probability_distribution_quantized = loss_utils.quantize_tensor(
      downsample_probability, 2 ** params.bit_depth, 0., 1.)

    # logging.info("distribution quantized {}".format(probability_distribution_quantized))

    logging.info("probability_distribution_quantized {}".format(probability_distribution_quantized))
    distribution_hook = tf.train.LoggingTensorHook(
      tensors={
        "full_resolution_scatterer": full_resolution_scatterer,
        "downsampled_scatterer": downsample_scatterer,
        "max_downsampled_scatterer": tf.math.reduce_max(downsample_scatterer),
        "full_resolution_prob": full_resolution_probability,
        "downsampled_prob": downsample_probability,
        "max_downsampled_prob": tf.math.reduce_max(downsample_probability)
      },
      every_n_iter=params.log_steps,
    )
    train_hooks.append(distribution_hook)

  with tf.variable_scope("observations"):
    # Use `Variable` nodes here because `constant` for some reason really slows
    # down the graph.
    tf_psfs = [tf.Variable(p.array, trainable=False) for p in params.psfs]
    simulator = online_simulation_utils.USsimulator(psfs=tf_psfs)
    observations = online_simulation_utils.observation_from_distribution(
      simulator, full_resolution_scatterer)
    observations_normalized = preprocess.per_tensor_scale(observations, -1., 1.)
    logging.info("observations {}".format(observations))

    observation_list = array_utils.reduce_split_tensor(observations[0], -1)
    angles = [p.angle for p in params.psfs]
    average_obs = average_observation(observation_list, angles=angles)

    observation_hook = tf.train.LoggingTensorHook(
      tensors={
        "observation_max": tf.math.reduce_max(observations, axis=[1, 2]),
        "observation_min": tf.math.reduce_min(observations, axis=[1, 2]),
        "observation_normalized_max": tf.math.reduce_max(
          observations_normalized, axis=[1, 2]),
        "observation_normalized_min": tf.math.reduce_min(
          observations_normalized, axis=[1, 2]),
      },
      every_n_iter=params.log_steps,
    )
    train_hooks.append(observation_hook)

    for i, psf in enumerate(tf_psfs):
      tf.summary.image("obs_{}".format(i), observations[..., i, tf.newaxis], 1)

    tf.summary.image(
      "average_observation", average_obs[tf.newaxis, ..., tf.newaxis], 1)

  if mode == tf.estimator.ModeKeys.TRAIN:
    training = True
  else:
    training = False

  features = network(observations, training)
  logging.info("features {}".format(features))

  # Get discretized predictions.
  predictions_quantized = tf.keras.layers.Conv2D(
    filters=2 ** params.bit_depth,
    kernel_size=[1, 1],
    dilation_rate=1,
    padding="same",
    activation=None,
    use_bias=False,
  ).apply(features)
  logging.info("predictions_quantized {}".format(predictions_quantized))

  with tf.variable_scope("predictions"):
    def _logit_to_class(logit):
      return tf.argmax(logit, -1)

    distribution_class = _logit_to_class(probability_distribution_quantized)
    prediction_class = _logit_to_class(predictions_quantized)

    # Log fraction nonzero.
    predicted_nonzero_count = tf.cast(tf.count_nonzero(prediction_class),
                                      tf.float32)
    true_nonzero_count = tf.cast(tf.count_nonzero(distribution_class),
                                 tf.float32)
    true_nonzero_fraction = true_nonzero_count / tf.cast(
      tf.size(prediction_class), tf.float32)
    nonzero_fraction = predicted_nonzero_count / tf.cast(
      tf.size(prediction_class), tf.float32)
    tf.summary.scalar("nonzero_fraction", nonzero_fraction)
    nonzero_hook = tf.train.LoggingTensorHook(
      tensors={
        "predicted_nonzero_fraction": nonzero_fraction,
        "true_nonzero_count": true_nonzero_count,
        "true_nonzero_fraction": true_nonzero_fraction,
      },
      every_n_iter=params.log_steps,
    )
    train_hooks.append(nonzero_hook)

    with tf.name_scope("images"):
      def _class_to_image(category):
        return tf.cast(category, tf.float32)[..., tf.newaxis]

      dist_image = _class_to_image(distribution_class)
      pred_image = _class_to_image(prediction_class)

      image_hook = tf.train.LoggingTensorHook(
        tensors={"distribution": dist_image[0, ..., 0],
                 "prediction": pred_image[0, ..., 0], },
        every_n_iter=params.log_steps,
      )
      eval_hooks.append(image_hook)

      dist_summary = tf.summary.image("full_resolution_scatterer", full_resolution_scatterer[0], 1)
      dist_summary = tf.summary.image("distributions", dist_image, 1)
      pred_summary = tf.summary.image("predictions", pred_image, 1)
      diff_summary = tf.summary.image("difference",
                                      tf.abs(dist_image - pred_image), 1)

      images_summaries = [dist_summary, pred_summary, diff_summary]

      images_summaries = tf.summary.merge(images_summaries)

      image_summary_hook = tf.train.SummarySaverHook(
        summary_op=images_summaries, save_secs=60)
      eval_hooks.append(image_summary_hook)

    with tf.name_scope("predictions"):
      predict_output = {
        "full_resolution_scatterer": full_resolution_scatterer,
        "full_resolution_probability": full_resolution_probability,
        "downsample_scatterer": downsample_scatterer,
        "downsample_probability": downsample_probability,
        "distribution_class": distribution_class,
        "normalized_observations": observations_normalized,
        "observations_unnormalized": observations,
        "prediction_class": prediction_class,
        "average_observation": average_obs,
      }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predict_output
      )

  with tf.name_scope("loss"):
    proportional_weights = loss_utils.inverse_class_weight(
      probability_distribution_quantized)
    proportion_hook = tf.train.LoggingTensorHook(
      tensors={"proportional_weights": proportional_weights[0],
               "min_weight": tf.reduce_min(proportional_weights)},
      every_n_iter=params.log_steps,
    )
    train_hooks.append(proportion_hook)

    softmax_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=probability_distribution_quantized,
      logits=predictions_quantized,
      weights=proportional_weights,
    )
    tf.summary.scalar("softmax_loss", softmax_loss)
    loss = softmax_loss

  with tf.name_scope("optimizer"):
    learning_rate = tf.train.exponential_decay(
      learning_rate=params.learning_rate,
      global_step=tf.train.get_global_step(),
      decay_steps=params.decay_step,
      decay_rate=params.decay_rate,
      staircase=True,
    )
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())

  with tf.name_scope("metrics"):
    batch_accuracy = tf.reduce_mean(
      tf.cast(tf.equal(distribution_class, prediction_class), tf.float32))
    tf.summary.scalar("batch_accuracy", batch_accuracy)

    accuracy_hook = tf.train.LoggingTensorHook(
      tensors={"batch_accuracy": batch_accuracy, },
      every_n_iter=params.log_steps
    )
    train_hooks.append(accuracy_hook)

    mean_squared_error = tf.metrics.mean_squared_error(
      labels=distribution_class,
      predictions=prediction_class,
    )

    accuracy_no_weight = tf.metrics.accuracy(
      labels=distribution_class,
      predictions=prediction_class,
    )

    accuracy_weight = tf.metrics.accuracy(
      labels=distribution_class,
      predictions=prediction_class,
      weights=proportional_weights,
    )

    eval_metric_ops = {
      "accuracy_weight": accuracy_weight,
      "accuracy_no_weight": accuracy_no_weight,
      "mean_squared_error": mean_squared_error,
    }

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
    training_hooks=train_hooks,
    evaluation_hooks=eval_hooks,
  )
