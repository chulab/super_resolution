"""Main module for training super resolution network."""

import argparse
import logging
import os

import tensorflow as tf

from analysis import plot_utils
from preprocessing import input


def predict(
    output_directory,
    model_checkpoint_directory,
    predict_dataset_directory,
    predict_parse_fns,
    model_fn,
    model_hparams,
):
  """Run the training and evaluate using the high level API."""
  tf_config = os.environ.get('TF_CONFIG')
  logging.info("TF_CONFIG {}".format(tf_config))

  # Load `RunConfig`.
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(
    model_dir=output_directory,
  )

  predict_dataset = input.make_input_fn(
      dataset_directory=predict_dataset_directory,
      parse_fns=predict_parse_fns,
    )
  iterator = predict_dataset.make_one_shot_iterator()
  observations, distribution = iterator.get_next()

  # Rebuild the model
  predictions = model_fn(
    observations,
    distribution,
    tf.estimator.ModeKeys.EVAL,
    model_hparams).predictions

  # Manually load the latest checkpoint.
  saver = tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_checkpoint_directory)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Loop through the batches and store predictions and labels
    output = []
    for i in range(1):
      try:
        preds_eval, dist_eval = sess.run([predictions, distribution])
        # Add distributions to `predictions`.
        preds_eval.update({'distribution': dist_eval})
        output.append(preds_eval)
      except tf.errors.OutOfRangeError:
        break

    return output


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--predict_dataset_directory',
    type=str,
    required=True
  )

  args, _ = parser.parse_known_args()

  return args


def run_train_and_evaluate(
    output_directory,
    model_checkpoint_directory,
    model_fn,
    hparams,
    predict_parse_fns,
):

  args = parse_args()

  predict(
    output_directory=output_directory,
    model_checkpoint_directory=model_checkpoint_directory,
    predict_dataset_directory=args.predict_dataset_directory,
    predict_parse_fns=predict_parse_fns,
    model_fn=model_fn,
    model_hparams=hparams,
  )