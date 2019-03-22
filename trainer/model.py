"""Defines model used to learn reconstruction."""

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
  distributions = labels

  if mode == tf.estimator.ModeKeys.TRAIN:
    training = True
  else:
    training = False

  # Run observations through CNN.
  predictions = network(observations, training)[..., 0]

  # Loss. Compare output of nn to original images.
  l2_loss = tf.reduce_sum((predictions - distributions) ** 2)
  loss = l2_loss

  rms_errror = tf.metrics.root_mean_squared_error(
    labels=distributions, predictions=predictions)

  eval_metric_ops = {
    "rms_error": rms_errror,
  }

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
    )

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={
      "predictions": predictions,
      "observations": observations,
      "distributions": distributions,
    }
    )

  if mode == tf.estimator.ModeKeys.EVAL:
    # Evaluation metric.
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, )


def build_estimator(
    config,
    params,
):
  return tf.estimator.Estimator(
    model_fn=model_fn,
    config=config,
    params=params,
  )