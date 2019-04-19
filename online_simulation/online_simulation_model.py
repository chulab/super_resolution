"""Model that performs simulation of US images online."""

import tensorflow as tf

import logging

from online_simulation import online_simulation_utils
from trainer import loss_utils

def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    bit_depth=2,
    learning_rate=0.001,
    decay_step=500,
    decay_rate=.9,
    psfs=None,
  )


def entry_flow(input_layer):
  network = input_layer
  network = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      use_bias=True,
      activation=tf.nn.leaky_relu
    ).apply(network)
  network=tf.keras.layers.BatchNormalization().apply(network)
  network=tf.keras.layers.Conv2D(
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      use_bias=True,
      activation=tf.nn.leaky_relu,
      strides=2,
    ).apply(network)
  network=tf.keras.layers.BatchNormalization().apply(network)
  return network


def conv_block(
    input_layer,
    filters=64,
    kernel_size=[3, 3],
    depthwise_kernel_size=[3, 3],
    depthwise_multiplier=1,
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
  # Pool downsample
  network = tf.keras.layers.DepthwiseConv2D(
    depth_multiplier=depthwise_multiplier,
    kernel_size=depthwise_kernel_size,
    dilation_rate=1,
    padding="same",
    strides=2,
  ).apply(network)
  return network


def network(input_layer, training):
  """Defines network.

  Args:
    `input_layer`: `tf.Tensor` node which outputs shapes `[b, h, w, c]`.
    These represent observations.
    training: Bool which sets whether network is in a training or evaluation/
      test mode. (Drop out is turned on during training but off during
      eval.)
  """
  network = input_layer

  network = entry_flow(network)

  for i in range(4):
    network = conv_block(network)

  network = tf.layers.dropout(network, training=training)

  return network


def downsample_by_pool(tensor, kernel_size):
  pool_layer=tf.keras.layers.AveragePooling2D(kernel_size, padding='same')
  return pool_layer.apply(tensor)


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

  print("features", features)
  print("labels", labels)

  distributions = features
  distributions = tf.cast(distributions, tf.float32)

  distribution_hook = tf.train.LoggingTensorHook(
    tensors={
      "distributions": distributions,
      "max": tf.math.reduce_max(distributions)
    },
    every_n_iter=50,
  )
  train_hooks.append(distribution_hook)

  # Use `Variable` nodes here because `constant` for some reason really slows
  # down the graph.
  tf_psfs = [tf.Variable(p, trainable=False) for p in params.psfs]
  simulator = online_simulation_utils.USsimulator(psfs=tf_psfs)
  observations = online_simulation_utils.observation_from_distribution(simulator, distributions)
  print("observations", observations)

  distributions = distributions[tf.newaxis, ..., tf.newaxis]
  distributions = downsample_by_pool(distributions, 32) * (32 ** 2)
  distributions = distributions[..., 0]
  print("distribution", distributions)

  distributions_quantized = loss_utils.quantize_tensor(
    distributions, params.bit_depth, 0., 2 ** params.bit_depth)
  print("distribution quantized", distributions_quantized)

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

    distribution_class = _logit_to_class(distributions_quantized)
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
      every_n_iter=50,
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
        every_n_iter=50,
      )
      eval_hooks.append(image_hook)

      dist_summary = tf.summary.image("distributions", dist_image, 1)
      pred_summary = tf.summary.image("predictions", pred_image, 1)
      diff_summary = tf.summary.image("difference",
                                      tf.abs(dist_image - pred_image), 1)

      images_summaries = [dist_summary, pred_summary, diff_summary]

      images_summaries = tf.summary.merge(images_summaries)

      image_summary_hook = tf.train.SummarySaverHook(
        summary_op=images_summaries, save_secs=120)
      eval_hooks.append(image_summary_hook)

    with tf.variable_scope("predictions"):
      predict_output = {
        "distribution_class": distribution_class,
        "observations": observations,
        "prediction_class": prediction_class
      }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predict_output
      )

  with tf.variable_scope("loss"):
    real_proportion = (tf.reduce_sum(
      distributions_quantized,
      axis=[0, 1, 2],
      keepdims=True,
    ) + 10) / (tf.cast(tf.size(distributions_quantized), tf.float32) + 10)
    proportional_weights = tf.reduce_sum(
      (1 / real_proportion) * distributions_quantized,
      axis=-1)
    proportion_hook = tf.train.LoggingTensorHook(
      tensors={"proportional_weights": proportional_weights[0], },
      every_n_iter=50,
    )
    train_hooks.append(proportion_hook)

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
      tensors={"batch_accuracy": batch_accuracy, },
      every_n_iter=100
    )
    train_hooks.append(accuracy_hook)

    accuracy = tf.metrics.accuracy(
      labels=distribution_class,
      predictions=prediction_class,
      weights=proportional_weights,
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
    training_hooks=train_hooks,
    evaluation_hooks=eval_hooks,
  )