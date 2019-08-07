"""Online version of Model builder. Refer to trainer/model_builder.py for
descriptions of hparams."""

import tensorflow as tf

import logging

from online_simulation import online_simulation_utils
import math
from trainer import loss_utils, model_builder
from preprocessing import preprocess


_PROBABILITY = "PROBABILITY"
_SCATTERER = "SCATTERER"


def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  hparams = model_builder.make_hparams()
  hparams.add_hparam("psf_dimension", None)
  hparams.add_hparam("grid_dimension", None)
  hparams.add_hparam("psf_descriptions", None)
  hparams.add_hparam("objective", _SCATTERER)
  hparams.add_hparam("log_steps", 100)
  hparams.add_hparam("concat_avg", False)
  hparams.add_hparam("downsample_bits", 4)
  return hparams


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
    scatterer_targets = process_target(full_resolution_scatterer,
      DOWNSAMPLE, params.bit_depth, params.lambda_multiplier)
    logging.info("scatterer_targets {}".format(scatterer_targets))

    full_resolution_probability = (
      features['probability_distribution'])
    probability_targets = process_target(
      full_resolution_probability, DOWNSAMPLE, params.bit_depth)
    logging.info("probability_targets {}".format(probability_targets))

    if params.objective == _PROBABILITY:
      target = probability_targets
    if params.objective == _SCATTERER:
      target = scatterer_targets

    target_hook = tf.train.LoggingTensorHook(
      tensors={"full_reso_scat": features['scatterer_distribution'][0],
               "full_reso_prob": features['probability_distribution'][0],
               "full_reso": target['full_resolution'][0],
               "class": target['class'][0]},
      every_n_iter=params.log_steps,
    )
    train_hooks.append(target_hook)

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

    average_obs = tf.reduce_mean(raw, -1, keepdims=True)
    processed_average_obs = process_target(average_obs, DOWNSAMPLE,
      params.bit_depth, params.lambda_multiplier)

    observations = {
      "raw": raw,
      "average": tf.reduce_mean(raw, -1, keepdims=True),
      "normalized": preprocess.per_tensor_scale(raw, -1., 1.),
      "descriptions": params.psf_descriptions,
      "average_downsampled": processed_average_obs["downsample"],
      "average_downsampled_class": processed_average_obs["class"],
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

    if params.concat_avg:
      concated = tf.concat([observations['raw'], observations['average']], -1)
      embedded = model_builder.get_embedding(concated, params.embedding)
    else:
      embedded = model_builder.get_embedding(observations['raw'],
        params.embedding)
    input = tf.keras.layers.Input(tensor=embedded)

    if params.concat_avg:
      num_classes = None
    else:
      num_classes= 2 ** params.bit_depth

    if params.model_type == 'unet':
      model = model_builder.build_unet_from_propagator(input, params,
        num_classes)
      predictions_quantized = model(embedded)
    else:
      model = model_builder.build_model_from_propagator(input, params,
        num_classes)
      predictions_quantized = model(embedded)

    if params.concat_avg:
      downsampled_average_image = tf.keras.layers.AveragePooling2D(DOWNSAMPLE,
        padding='same').apply(observations['average'])
      predictions_quantized = tf.concat([predictions_quantized,
        downsampled_average_image], -1)
      predictions_quantized = tf.keras.layers.Conv2D(
        filters=2 ** params.bit_depth,
        kernel_size=[1, 1],
        dilation_rate=1,
        padding="same",
        activation=None,
        use_bias=False,
      ).apply(predictions_quantized)

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
      "average_observation_downsampled": observations['average_downsampled'],
      "average_observation_downsampled_class": observations['average_downsampled_class'],
    }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predict_output
    )

  with tf.name_scope("loss"):
    less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
    lr = tf.cond(less_equal, lambda: tf.constant(params.learning_rate),
      lambda: tf.constant(params.learning_rate / 1000))

    proportional_weights = loss_utils.bets_and_rewards_weight(
      target['one_hot'], target['class'], prediction_class, params)

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
      learning_rate=lr,
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

    accuracy = tf.metrics.accuracy(
      labels=target['class'], predictions=predictions['class'])

    proportional_weights = loss_utils.inverse_class_weight(
      target['one_hot'])
    accuracy_weight = tf.metrics.accuracy(
      labels=target['class'], predictions=predictions['class'],
      weights=proportional_weights,)

    precision = tf.metrics.precision(
      labels=target['class'], predictions=predictions['class'])

    recall = tf.metrics.recall(
      labels=target['class'], predictions=predictions['class'])

    f1 = tf.where(tf.equal(precision[0] + recall[0], 0.),
      tf.constant(0, dtype=tf.float32), 2 * precision[0] * recall[0] /
      (precision[0] + recall[0]))

    non_zero = tf.where(tf.equal(0, tf.cast(target['class'], dtype=tf.int32)),
      -1 * tf.ones_like(target['class']), target['class'])
    non_zero_correct = tf.math.reduce_sum(tf.cast(
      tf.equal(non_zero, predictions['class']), dtype=tf.int32))
    total_non_zero =tf.math.reduce_sum(tf.cast(tf.not_equal(0,
      tf.cast(target['class'], dtype=tf.int32)), dtype=tf.int32))
    non_zero_acc = tf.where(tf.equal(total_non_zero, 0),
      tf.constant(0, dtype=tf.float64), non_zero_correct / total_non_zero)

    eval_metric_ops = {
      "accuracy_weighted": accuracy_weight,
      "accuracy": accuracy,
      "mean_squared_error": mean_squared_error,
      "precision": precision,
      "recall": recall,
      "f1": tf.metrics.mean(f1),
      "non_zero_acc": tf.metrics.mean(non_zero_acc),
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
