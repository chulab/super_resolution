"""Defines model used to learn reconstruction."""

import logging

import tensorflow as tf


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    learning_rate=0.1,
  )


def network(input_layer, training):
  """Defines network.

  Args:
    `input_layer`: `tf.Tensor` node which outputs shapes `[b, h, w, c]`.
    These represent observations.
    training: Bool which sets whether network is in a training or evaluation/
      test mode. (Drop out is turned on during training but off during
      eval.)
  """
  input_layer.shape.assert_is_compatible_with([None, None, None, None])

  with tf.variable_scope("Model"):
    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(input_layer)

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

    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[10, 10],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.Conv2D(
      filters=1,
      kernel_size=[10, 10],
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

  if mode == tf.estimator.ModeKeys.TRAIN:
    training = True
  else:
    training = False

  # Run observations through CNN.
  predictions = network(observations, training)[..., 0]

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

  with tf.variable_scope("predictions"):
    predict_output = {
      "predictions": predictions,
      "observations": observations,
      "distributions": distributions,
    }

    # Add image summaries.
    tf.summary.image("observation", observations[..., 0, tf.newaxis], 1)
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