"""Defines model used to learn reconstruction."""

import logging

import tensorflow as tf
from utils import array_utils


def make_hparams() -> dict:
  """Create a HParams object specifying model hyperparameters."""
  return {
    'learning_rate': 0.1,
    'observation_spec': None,
  }


def AngleModule(input_shape):
  """Defines angle module."""
  with tf.name_scope("angle_module"):
    inputs = tf.keras.Input(shape=input_shape)

    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[3, 3],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    )(inputs)

    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[6, 6],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    output = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[10, 10],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    return tf.keras.Model(inputs=inputs, outputs=output)


def model(input_shape, angles):
  """Defines model.

  Args:
    input_layer: `tf.Tensor` of shape `height, width, `
  """
  with tf.name_scope("angle_first_model"):
    inputs = tf.keras.Input(shape=input_shape)

    network = tf.keras.layers.Lambda(
      array_utils.reduce_split_tensor, arguments={'axis': 1}).apply(inputs)

    angle_module = AngleModule(input_shape=input_shape[1:])

    # First pipe each input through an `angle_module`.
    angle_output = [angle_module(input) for input in network]

    # Rotate output so all features are co-registered.
    angle_centered = [
      tf.keras.layers.Lambda(tf.contrib.image.rotate, arguments={
        'angles': angle, 'interpolation': "BILINEAR"})(tensor)
                      for tensor, angle in zip(angle_output, angles)]

    # Stack output of angle modules along `filter` dimension.
    network = tf.keras.layers.Concatenate()(angle_centered)

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
      dilation_rate=3,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[3, 3],
      dilation_rate=2,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[3, 3],
      dilation_rate=4,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    output = tf.keras.layers.Conv2D(
      filters=1,
      kernel_size=[5, 5],
      dilation_rate=1,
      padding="same",
      activation=tf.nn.leaky_relu
    ).apply(network)

    return tf.keras.Model(inputs=inputs, outputs=output)


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
  predictions = model(observations.shape[1:], params["observation_spec"].angles)(observations)[..., 0]

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

  with tf.variable_scope("predictions"):
    predict_output = {
      "predictions": predictions,
      "observations": observations,
      "distributions": distributions,
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