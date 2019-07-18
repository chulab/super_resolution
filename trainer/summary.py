'''Utility functions for logging'''

def log_tensor()

def add_nonzero_fraction_hook(distribution_class, prediction_class, hooks):
  '''Finds nonzero fractions in distributions and predictions and adds
  new nonzero fraction hook to hooks.

  Args:
    distribution_class: tf.Tensor of shape `[batch, height, width]`.
    prediction_class: tf.Tensor of shape `[batch, height, width]`.
    hooks: list of hooks.

  Returns:
    hooks: updated list of hooks.
  '''
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

  return hooks


def add_image_hook(distribution_class, prediction_class, hooks):
  '''Visualize images from distribution_class and prediction_class
  plus their differences.

  Args:
    distribution_class: tf.Tensor of shape `[batch, height, width]`.
    prediction_class: tf.Tensor of shape `[batch, height, width]`.
    hooks: list of hooks.

  Returns:
    eval_summary_hook: image evaluation hook for estimator.
  '''
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

  dist_summary = tf.summary.image("distributions", dist_image, 1)
  pred_summary = tf.summary.image("predictions", pred_image, 1)
  diff_summary = tf.summary.image("difference", (dist_image - pred_image) ** 2, 1)

  images_summaries = [dist_summary, pred_summary, diff_summary]

  images_summaries = tf.summary.merge(images_summaries)

  eval_summary_hook = tf.train.SummarySaverHook(
    summary_op=images_summaries, save_secs=120)

  return eval_summary_hook
