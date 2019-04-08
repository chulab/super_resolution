"""Main module for training super resolution network."""

import argparse
import json
import logging
import os

import tensorflow as tf

from preprocessing import input


def train_and_evaluate(
    output_directory,
    train_dataset_directory,
    eval_dataset_directory,
    train_steps,
    eval_steps,
    train_parse_fns,
    eval_parse_fns,
    model_fn,
    model_hparams,
    warm_start_from,
    profile_steps,
    save_checkpoint_steps,
    log_step_count,
):
  """Run the training and evaluate using the high level API."""

  def train_input():
    """Input function returning batches from the training data set.
    """
    return input.make_input_fn(
      dataset_directory=train_dataset_directory,
      parse_fns=train_parse_fns,
      mode=input._TRAIN
    )

  def eval_input():
    """Input function returning batches from the evaluation data set.
    """
    return input.make_input_fn(
      dataset_directory=eval_dataset_directory,
      parse_fns=eval_parse_fns,
      mode=input._EVAL
    )

  tf_config = os.environ.get('TF_CONFIG')
  logging.info("TF_CONFIG {}".format(tf_config))

  # Load `RunConfig`.
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(
    model_dir=output_directory,
    log_step_count_steps=log_step_count,
    save_checkpoints_steps=save_checkpoint_steps
  )

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params=model_hparams,
    warm_start_from=warm_start_from
  )

  # Hook to log step timing.
  hook = tf.train.ProfilerHook(
    save_steps=profile_steps,
    output_dir=output_directory,
    show_memory=True
  )

  logging.info("Defining `train_spec`.")
  train_spec = tf.estimator.TrainSpec(
    input_fn=train_input,
    max_steps=train_steps,
    hooks=[hook]
  )

  logging.info("Defining `eval_spec`.")
  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input,
    steps=eval_steps,
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--train_dataset_directory',
    type=str,
    required=True
  )

  parser.add_argument(
    '--eval_dataset_directory',
    type=str,
    required=True
  )

  parser.add_argument(
    '--train_steps',
    type=int,
    required=True
  )

  parser.add_argument(
    '--eval_steps',
    type=int,
    default=100,
  )

  parser.add_argument(
    '--profile_steps',
    type=int,
    default=200,
  )

  parser.add_argument(
    '--save_checkpoint_steps',
    type=int,
    default=200,
  )

  parser.add_argument(
    '--log_step_count',
    type=int,
    default=20,
  )

  args, _ = parser.parse_known_args()

  return args


def run_train_and_evaluate(
    output_directory,
    model_fn,
    hparams,
    train_parse_fns,
    eval_parse_fns,
    warm_start_from=None,
):
  args = parse_args()

  logging.info("Using tensorflow {}".format(tf.__version__))

  # Modify `output_dir` for each trial if hyperparameter tuning.
  output_directory = os.path.join(
    output_directory,
    json.loads(
      os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '')
    )

  logging.info("Using `output_directory` {}".format(output_directory))

  train_and_evaluate(
    output_directory=output_directory,
    train_dataset_directory=args.train_dataset_directory,
    eval_dataset_directory=args.eval_dataset_directory,
    train_steps=args.train_steps,
    eval_steps=args.eval_steps,
    train_parse_fns=train_parse_fns,
    eval_parse_fns=eval_parse_fns,
    model_fn=model_fn,
    model_hparams=hparams,
    warm_start_from=warm_start_from,
    profile_steps=args.profile_steps,
    save_checkpoint_steps=args.save_checkpoint_steps,
    log_step_count=args.log_step_count,
  )