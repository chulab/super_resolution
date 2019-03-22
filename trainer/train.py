"""Main module for training super resolution network."""
import argparse
import logging
import os
import sys

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from preprocessing import input
from preprocessing import parser
from simulation import create_observation_spec

from trainer import model


def train_and_evaluate(
    train_steps,
    output_directory,
    train_dataset_directory,
    eval_dataset_directory,
    train_parse_fn,
    eval_parse_fn,
    interleave_cycle_length,
    shuffle_buffer_size,
    batch_size,
    num_parallel_reads,
    model_hparams,
):
  """Run the training and evaluate using the high level API."""

  def train_input():
    """Input function returning batches from the training data set.
    """
    return input.input_fn(
      train_dataset_directory,
      train_parse_fn,
      interleave_cycle_length,
      shuffle_buffer_size,
      batch_size,
      num_parallel_reads,
    )

  def eval_input():
    """Input function returning batches from the evaluation data set.
    """
    return input.input_fn(
      eval_dataset_directory,
      eval_parse_fn,
      interleave_cycle_length,
      shuffle_buffer_size=1, # Do not shuffle eval data.
      batch_size=1,
      num_parallel_reads=1,
    )

  logging.info("Defining `train_spec`.")
  train_spec = tf.estimator.TrainSpec(
    input_fn=train_input,
    max_steps=train_steps,
  )

  logging.info("Defining `eval_spec`.")
  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input,
    steps=100,
  )

  # Load `RunConfig`.
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=output_directory)

  estimator = model.build_estimator(
    config=run_config,
    params=model_hparams,
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('-o', '--output_dir',
                      dest='output_dir',
                      help='Path to save output of model including checkpoints.',
                      type=str,
                      required=True)

  parser.add_argument(
    '--distribution_blur_sigma',
    dest='distribution_blur_sigma',
    type=float,
    required=True,
  )

  parser.add_argument(
    '--observation_blur_sigma',
    dest='observation_blur_sigma',
    type=float,
    required=True,
  )

  parser.add_argument(
    '--distribution_downsample_size',
    dest='distribution_downsample_size',
    type=lambda s: [int(size) for size in s.split(',')],
    required=True,
  )

  parser.add_argument(
    '--observation_downsample_size',
    dest='observation_downsample_size',
    type=lambda s: [int(size) for size in s.split(',')],
    required=True,
  )

  parser.add_argument(
    '--example_size',
    dest='example_size',
    type=lambda s: [int(size) for size in s.split(',')],
    required=True,
  )

  parser.add_argument(
    '--train_steps',
    dest='train_steps',
    type=int,
    required=True,
  )

  parser.add_argument(
    '--train_dataset_directory',
    dest='train_dataset_directory',
    type=str,
    required=True,
  )

  parser.add_argument(
    '--eval_dataset_directory',
    dest='eval_dataset_directory',
    type=str,
    required=True,
  )

  parser.add_argument(
    '--observation_spec_path',
    dest='observation_spec_path',
    type=str,
    required=True,
  )

  parser.add_argument(
    '--interleave_cycle_length',
    dest='interleave_cycle_length',
    type=int,
    default=1,
    required=False,
  )

  parser.add_argument(
    '--shuffle_buffer_size',
    dest='shuffle_buffer_size',
    type=int,
    default=1,
    required=False,
  )

  parser.add_argument(
    '--batch_size',
    dest='batch_size',
    type=int,
    default=1,
    required=False,
  )

  parser.add_argument(
    '--num_parallel_reads',
    dest='num_parallel_reads',
    type=int,
    default=1,
    required=False,
  )

  return parser.parse_known_args()


def main():
  args = parse_args()

  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path
  )

  train_parse_fn = parser.Parser(
    observation_spec=observation_spec,
    reverse_rotation=True,
    distribution_blur_sigma=args.distribution_blur_sigma,
    observation_blur_sigma=args.observation_blur_sigma,
    distribution_downsample_size=args.distribution_downsample_size,
    observation_downsample_size=args.observation_downsample_size,
    example_size=args.example_size,
  ).parse

  eval_parse_fn = parser.Parser(
    observation_spec=observation_spec,
    reverse_rotation=True,
    distribution_blur_sigma=args.distribution_blur_sigma,
    observation_blur_sigma=args.observation_blur_sigma,
    distribution_downsample_size=args.distribution_downsample_size,
    observation_downsample_size=args.observation_downsample_size,
    example_size=args.example_size,
  ).parse

  hparams = model.make_hparams()
  hparams.learning_rate = args.learning_rate

  train_and_evaluate(
    train_steps=args.train_steps,
    output_directory=args.output_dir,
    train_dataset_directory=args.train_dataset_directory,
    eval_dataset_directory=args.eval_dataset_directory,
    train_parse_fn=train_parse_fn,
    eval_parse_fn=eval_parse_fn,
    interleave_cycle_length=args.interleave_cycle_length,
    shuffle_buffer_size=args.shuffle_buffer_size,
    batch_size=args.batch_size,
    num_parallel_reads=args.num_parallel_reads,
    model_hparams=hparams,
  )


if __name__ == "__main__":
  main()