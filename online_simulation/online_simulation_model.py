"""Model that performs simulation of US images online."""

import tensorflow as tf
import math
import logging

from online_simulation import online_simulation_utils
from trainer import loss_utils
from trainer.model_builder import squeeze_excitation
from preprocessing import preprocess
from online_simulation import online_dataset_utils


_PROBABILITY = "PROBABILITY"
_SCATTERER = "SCATTERER"


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    bit_depth=2,
    learning_rate=0.001,
    decay_step=500,
    decay_rate=.9,
    psf_dimension=None,
    grid_dimension=None,
    psf_descriptions=None,
    log_steps=100,
    objective=_SCATTERER,
    squeeze_excite=False,
    downsample_bits=4,
    prepool_bits=0,
  )


def entry_flow(input_layer, repeats=2):
  """Begining of CNN."""
  with tf.name_scope("entry_flow"):
    network = input_layer
    # For residual, we want same number of filters as input.
    filters = input_layer.shape.as_list()[-1]

    for _ in range(repeats):
      residual = tf.keras.layers.AveragePooling2D(2, padding='same')(network)
      network = tf.keras.layers.Activation('relu')(network)
      network = tf.keras.layers.SeparableConv2D(
          filters=filters,
          kernel_size=[3, 3],
          padding="same",
          use_bias=True,
          strides=2,
        ).apply(network)
      network = tf.keras.layers.BatchNormalization()(network)
      network = tf.keras.layers.add([network, residual])

    return network


def _conv_block(
    input_layer,
    filters=64,
    kernel_size=[3, 3],
    squeeze_excite=False,
):
  with tf.name_scope("conv_block"):
    network = input_layer
    residual = network
    network = tf.keras.layers.Activation('relu').apply(network)
    network = tf.keras.layers.SeparableConv2D(
      filters=filters,
      depth_multiplier=1,
      kernel_size=kernel_size,
      padding="same",
      use_bias=True,
    ).apply(network)
    network = tf.keras.layers.BatchNormalization().apply(network)
    network = tf.keras.layers.Activation('relu')(network)
    network = tf.keras.layers.SeparableConv2D(
      filters=filters,
      depth_multiplier=1,
      kernel_size=kernel_size,
      padding="same",
      use_bias=True,
      dilation_rate=2,
    ).apply(network)
    network = tf.keras.layers.BatchNormalization().apply(network)
    network = tf.keras.layers.Activation('relu')(network)
    network = tf.keras.layers.SeparableConv2D(
      filters=filters,
      depth_multiplier=1,
      kernel_size=kernel_size,
      padding="same",
      use_bias=True,
      dilation_rate=4,
    ).apply(network)
    network = tf.keras.layers.BatchNormalization().apply(network)
    if squeeze_excite:
      network = squeeze_excitation(network)
    network = tf.keras.layers.add([network, residual])
    return network


def _downsample_block(
    input_layer,
    depthwise_kernel_size=[3, 3],
    depth_multiplier=1,
):
  with tf.name_scope("downsample_block"):
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
    dilation_rate=1,
    padding="same",
    strides=2,
  ).apply(input_layer)
  network = tf.keras.layers.BatchNormalization().apply(network)
  return network


def network(input_layer, average_image, training, squeeze_excite,
  downsample_bits, prepool_bits):
  """Definition of network.

  Args:
    `input_layer`: `tf.Tensor` node which outputs shapes `[b, h, w, c]`.
    These represent observations.
    training: Bool which sets whether network is in a training or evaluation/
      test mode. (Drop out is turned on during training but off during
      eval.)
  """
  with tf.name_scope("model"):
    layers = {}

    network = tf.concat([input_layer, average_image], -1)
    if prepool_bits > 0:
      prepool_factor = 2 ** prepool_bits
      network = _downsample_by_pool(network, prepool_factor)

    filters = network.shape.as_list()[-1]
    network = entry_flow(network, repeats=(downsample_bits - prepool_bits - 2))

    downsampled_average_image = _downsample_by_pool(average_image, 2 **
      downsample_bits)

    for i in range(2):
      network = _conv_block(network, filters=filters,
        squeeze_excite=squeeze_excite)
      network = _downsample_block(network)
      layers['downsample_{}'.format(i)] = network

    for i in range(3):
      network = _conv_block(network, filters=filters, kernel_size=5,
        squeeze_excite=squeeze_excite)

    logging.info("size after network {}".format(network))

    network = tf.concat([network, downsampled_average_image], -1)

    network = tf.layers.dropout(network, training=training)

    return network


def _downsample_by_pool(tensor, kernel_size):
  pool_layer = tf.keras.layers.AveragePooling2D(kernel_size, padding='same')
  return pool_layer.apply(tensor)


def process_target(target, downsample_factor, bit_depth, lambda_multiplier=None):
  """Perform common processing on target tensor.

  Typically `target` will be either the scatterer distribution or
  probability distribution.

  Args:
    target: `tf.Tensor` of shape `[Batch, Height, Width]`.
    downsample_factor: Ratio of `original_dimension/output_dimension`
      for desired processed target. I.e. a downsample factor of 4
      corresponds to the sampling of an image of shape `[400, 400]` ->
      `[100, 100]`
    bit_depth: bit depth of output. The number of channels in the
      `quantized` output will be `2 ** bit_depth`.
    lambda_multiplier: Poisson lambda used to generate dataset.

  Returns:
    Dictionary containing:
      * full_resolution: Same as `target`.
      * downsample: `tf.Tensor` resulting from downsampling `target`.
      * class: Result of quantizing `downsample` into `2 ** bit_depth` classes.
      * one-hot: one-hot representation of `class`
  """
  downsample = loss_utils.downsample_target(target, downsample_factor)
  if lambda_multiplier is not None:
    # only consider 3 stds
    max_value = lambda_multiplier + 3 * math.sqrt(lambda_multiplier)
  else:
    max_value = 1
  quantized_class, one_hot = loss_utils.quantize_tensor(
    downsample, 2 ** bit_depth, 0., max_value)
  return {
    "full_resolution": target,
    "downsample": downsample,
    "class": quantized_class,
    "one_hot": one_hot,
  }


def model_fn(features, labels, mode, params):
  """Define model graph for super resolution.

  Args:
    features: dict containing:
      * scatterer_distribution: a `tf.Tensor` with shape
        `[batch_size, height, width, channels]`
      * probability_distribution: same as `scatterer_distribution` but
        the raw probabilities leading to the specific distribution of
        scatterers.
    mode: str. must be one of `tf.estimator.ModeKeys`.
    params: `tf.contrib.training.HParams` object containing hyperparameters for
      model.

  Returns:
    `tf.Estimator.EstimatorSpec` object.
  """
  train_hooks = []
  eval_hooks = []

  with tf.name_scope("distributions"):
    DOWNSAMPLE = 2 ** params.downsample_bits
    full_resolution_scatterer = features['scatterer_distribution']
    scatterer_targets = process_target(
      full_resolution_scatterer, DOWNSAMPLE, params.bit_depth,
      lambda_multiplier = params.lambda_multiplier)
    logging.info("scatterer_targets {}".format(scatterer_targets))

    full_resolution_probability = (
      features['probability_distribution'])
    probability_targets = process_target(
      full_resolution_probability, DOWNSAMPLE, params.bit_depth)
    logging.info("probability_targets {}".format(probability_targets))

    mean, variance = tf.nn.moments(features['scatterer_distribution'][0], [0, 1])
    feature_hook = tf.train.LoggingTensorHook(
      tensors={"scatterers": features['scatterer_distribution'][0],
               "probability": features['probability_distribution'][0],
               "max": tf.reduce_max(features['scatterer_distribution'][0]),
               "mean": mean,
               "variance": variance},
      every_n_iter=params.log_steps,
    )
    train_hooks.append(feature_hook)

    if params.objective == _PROBABILITY:
      target = probability_targets
    if params.objective == _SCATTERER:
      target = scatterer_targets

  with tf.variable_scope("observations"):
    psfs = online_simulation_utils.make_psf(
      psf_dimension=params.psf_dimension,
      grid_dimension=params.grid_dimension,
      descriptions=params.psf_descriptions,
    )
    logging.info("psfs count {}".format(len(psfs)))
    logging.info("psfs {}".format(psfs))

    sim = online_simulation_utils.USSimulator(
      psfs=psfs,
      image_grid_size=full_resolution_scatterer.shape.as_list()[1:],
      grid_dimension=params.grid_dimension,
    )

    raw = sim.observation_from_distribution(full_resolution_scatterer)

    logging.info("observations {}".format(raw))

    logging.info("Observation variables {}".format(tf.all_variables()))

    observations = {
      "raw": raw,
      "average": tf.reduce_mean(raw, -1, keepdims=True),
      "normalized": preprocess.per_tensor_scale(raw, -1., 1.),
      "descriptions": params.psf_descriptions,
    }

    with tf.name_scope("image_summary"):
      split_observations = tf.split(observations['raw'], len(psfs), axis=-1)
      for obs, p in zip(split_observations, psfs):
        title = "obs_{frequency}_{angle}_{mode}".format(
          frequency=p.psf_description.frequency,
          angle=p.angle,
          mode=p.psf_description.mode
        )
        tf.summary.image(title, obs, 1)
      tf.summary.image("average_observation", observations['average'], 1)

  with tf.name_scope("features"):
    if mode == tf.estimator.ModeKeys.TRAIN:
      training = True
    else:
      training = False

    features = network(observations['raw'], observations['average'], training,
      params.squeeze_excite, params.downsample_bits, params.prepool_bits)
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
    prediction_class = loss_utils._logit_to_class(predictions_quantized)
    predictions = {
      "logits": predictions_quantized,
      "class": prediction_class,
    }
    logging.info("predictions {}".format(predictions))
    logging.info("After features variables {}".format(tf.all_variables()))

    with tf.name_scope("images"):
      def _class_to_image(category):
        """Convert `[B, H, W]` -> `[B, H, W, 1]`."""
        return tf.cast(category, tf.float32)[..., tf.newaxis]

      target_image = _class_to_image(target['class'])
      prediction_image = _class_to_image(predictions['class'])
      difference = tf.abs(target_image - prediction_image)

      dist_fr_summary = tf.summary.image(
        "full_resolution_scatterer",
        scatterer_targets['full_resolution'][..., tf.newaxis], 1)
      dist_summary = tf.summary.image("target_image", target_image, 1)
      pred_summary = tf.summary.image("prediction_image", prediction_image, 1)
      diff_summary = tf.summary.image("difference", difference, 1)

      # for Google Slides summary 
      dist_t_summary = tf.summary.tensor_summary("distribution_tensor", target_image)
      pred_t_summary = tf.summary.tensor_summary("predictions_tensor", prediction_image)
      diff_t_summary = tf.summary.tensor_summary("difference_tensor", target_image-prediction_image)

      images_summaries = [
        dist_fr_summary, dist_summary, pred_summary, diff_summary,
        dist_t_summary, pred_t_summary, diff_t_summary]
      images_summaries = tf.summary.merge(images_summaries)
      image_summary_hook = tf.train.SummarySaverHook(save_steps=1,
        output_dir= params.job_dir + "/eval", summary_op = images_summaries)
      eval_hooks.append(image_summary_hook)

  with tf.name_scope("predict_mode"):
    predict_output = {
      "full_resolution_scatterer": full_resolution_scatterer,
      "full_resolution_probability": full_resolution_probability,
      "downsample_scatterer": scatterer_targets['downsample'],
      "scatterer_class": scatterer_targets['class'],
      "downsample_probability": probability_targets['downsample'],
      "probability_class": probability_targets['class'],
      "prediction_class": predictions['class'],
      "normalized_observations": observations['normalized'],
      "observations_unnormalized": observations['raw'],
      "average_observation": observations['average'],
    }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predict_output
    )

  with tf.name_scope("loss"):
    proportional_weights = loss_utils.inverse_class_weight(target['one_hot'])
    proportion_hook = tf.train.LoggingTensorHook(
      tensors={"proportional_weights": proportional_weights[0],
               "min_weight": tf.reduce_min(proportional_weights)},
      every_n_iter=params.log_steps,
    )
    train_hooks.append(proportion_hook)

    softmax_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=target['one_hot'],
      logits=predictions['logits'],
      weights=proportional_weights,
    )
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
    tf.summary.scalar("softmax_loss", softmax_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())

  with tf.name_scope("metrics"):
    batch_accuracy = tf.reduce_mean(
      tf.cast(tf.equal(target['class'], predictions['class']), tf.float32))
    tf.summary.scalar("batch_accuracy", batch_accuracy)
    accuracy_hook = tf.train.LoggingTensorHook(
      tensors={"batch_accuracy": batch_accuracy, },
      every_n_iter=params.log_steps
    )
    train_hooks.append(accuracy_hook)

    mean_squared_error = tf.metrics.mean_squared_error(
      labels=target['class'], predictions=predictions['class'])

    accuracy_no_weight = tf.metrics.accuracy(
      labels=target['class'], predictions=predictions['class'])

    accuracy_weight = tf.metrics.accuracy(
      labels=target['class'], predictions=predictions['class'],
      weights=proportional_weights,)

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
