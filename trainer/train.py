"""Main module for training super resolution network.

Example usage (run on data in `simulation/training_data`
python trainer/train.py -o trainer/test_output --distribution_blur_sigma 1e-3 --observation_blur_sigma 1e-3 --distribution_downsample_size 100,100 --observation_downsample_size 100,100 --example_size 101,101 --train_steps 10 --train_dataset_directory simulation/test_data/ --eval_dataset_directory simulation/test_data/ --observation_spec_path simulation/test_data/test_observation_spec.json
"""
import argparse
import logging
import os
import sys

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from preprocessing import input
from preprocessing import parser
from preprocessing import test_input_pipeline

from simulation import create_observation_spec

from trainer import model as straight_model
from trainer import angle_first_model


_STRAIGHT="STRAIGHT"
_ANGLE_FIRST="ANGLE_FIRST"


def train_and_evaluate(
    train_steps,
    output_directory,
    train_dataset_directory,
    eval_dataset_directory,
    train_parse_fn,
    eval_parse_fn,
    estimator_fn,
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

  # test_input_pipeline.save_input(train_input(), "train", output_directory)
  # test_input_pipeline.save_input(eval_input(), "eval", output_directory)
  # logging.info("Saved input examples.")

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

  tf_config = os.environ.get('TF_CONFIG')
  logging.info("TF_CONFIG {}".format(tf_config))

  # Load `RunConfig`.
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=output_directory)

  estimator = estimator_fn(
    config=run_config,
    params=model_hparams,
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--job-dir',
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
    '--model_type',
    dest='model_type',
    type=str,
    default=_STRAIGHT,
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

  parser.add_argument(
    '--learning_rate',
    dest='learning_rate',
    type=float,
    default=.001,
    required=False,
  )

  args, _ = parser.parse_known_args()

  return args


def _set_up_logging():
  """Sets up logging."""

  # Check for environmental variable.
  file_location = os.getenv('JOB_DIRECTORY', '.')

  print("Logging file writing to {}".format(file_location), flush=True)

  logging.basicConfig(
    filename=os.path.join(file_location, 'training.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(process)d - %(message)s'
  )

  logging.debug("Initialize debug.")


def main():

  _set_up_logging()

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
  logging.info("Initialized `train_parse_fn`.")

  eval_parse_fn = parser.Parser(
    observation_spec=observation_spec,
    reverse_rotation=True,
    distribution_blur_sigma=args.distribution_blur_sigma,
    observation_blur_sigma=args.observation_blur_sigma,
    distribution_downsample_size=args.distribution_downsample_size,
    observation_downsample_size=args.observation_downsample_size,
    example_size=args.example_size,
  ).parse
  logging.info("Initialized `eval_parse_fn`.")

  if args.model_type==_STRAIGHT:
    model=straight_model
  elif args.model_type==_ANGLE_FIRST:
    model=angle_first_model
  else:
    raise ValueError('Not a valid model type. Got {}'.format(args.model_type))

  estimator_fn = model.build_estimator
  hparams = model.make_hparams()

  hparams['learning_rate'] = args.learning_rate
  hparams['observation_spec'] = observation_spec

  train_and_evaluate(
    train_steps=args.train_steps,
    output_directory=args.job_dir,
    train_dataset_directory=args.train_dataset_directory,
    eval_dataset_directory=args.eval_dataset_directory,
    train_parse_fn=train_parse_fn,
    eval_parse_fn=eval_parse_fn,
    estimator_fn=estimator_fn,
    interleave_cycle_length=args.interleave_cycle_length,
    shuffle_buffer_size=args.shuffle_buffer_size,
    batch_size=args.batch_size,
    num_parallel_reads=args.num_parallel_reads,
    model_hparams=hparams,
  )


if __name__ == "__main__":
  main()