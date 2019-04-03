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

from simulation import create_observation_spec

from trainer import residual_frequency_first_model as model

_STRAIGHT="STRAIGHT"
_ANGLE_FIRST="ANGLE_FIRST"


def train_and_evaluate(
    output_directory,
    train_dataset_directory,
    eval_dataset_directory,
    train_steps,
    train_parse_fns,
    eval_parse_fns,
    parallel_calls,
    estimator_fn,
    interleave_cycle_length,
    shuffle_buffer_size,
    batch_size,
    prefetch,
    model_hparams,
):
  """Run the training and evaluate using the high level API."""

  def train_input():
    """Input function returning batches from the training data set.
    """
    return input.input_fn(
      dataset_directory=train_dataset_directory,
      parse_fns=train_parse_fns,
      parallel_calls=parallel_calls,
      interleave_cycle_length=interleave_cycle_length,
      shuffle_buffer_size=shuffle_buffer_size,
      batch_size=batch_size,
      prefetch=prefetch,
    )

  def eval_input():
    """Input function returning batches from the evaluation data set.
    """
    return input.input_fn(
      dataset_directory=eval_dataset_directory,
      parse_fns=eval_parse_fns,
      parallel_calls=parallel_calls,
      interleave_cycle_length=interleave_cycle_length,
      shuffle_buffer_size=shuffle_buffer_size,
      batch_size=batch_size,
      prefetch=prefetch,
    )

  tf_config = os.environ.get('TF_CONFIG')
  logging.info("TF_CONFIG {}".format(tf_config))

  # Load `RunConfig`.
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(
    model_dir=output_directory,
    log_step_count_steps=1,
  )

  estimator = estimator_fn(
    config=run_config,
    params=model_hparams,
  )

  # Hook to log step timing.
  hook = tf.train.ProfilerHook(
    save_steps=1000,
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
    steps=100,
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--job-dir',
                      help='Path to save output of model including checkpoints.',
                      type=str,
                      required=True)

  parser.add_argument(
    '--train_dataset_directory',
    type=str,
    required=True,
  )

  parser.add_argument(
    '--eval_dataset_directory',
    type=str,
    required=True,
  )

  parser.add_argument(
    '--train_steps',
    type=int,
    required=True,
  )

  parser.add_argument(
    '--observation_spec_path',
    type=str,
    required=True,
  )

  parser.add_argument(
    '--parallel_calls',
    type=int,
    default=1,
    required=False,
  )

  parser.add_argument(
    '--interleave_cycle_length',
    type=int,
    default=1,
    required=False,
  )

  parser.add_argument(
    '--shuffle_buffer_size',
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
    '--prefetch',
    type=int,
    default=1,
    required=False,
  )

  parser.add_argument(
    '--hparams',
    type=str,
    help='Comma separated list of "name=value" pairs.',
    default=''
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

  args = parse_args()

  _set_up_logging()

  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path, False
  )

  estimator_fn = model.build_estimator

  parse_fns = model.input_fns()
  hparams = model.make_hparams()
  hparams.parse(args.hparams)
  # Must manually replace `observation_spec`.
  hparams.observation_spec=observation_spec

  logging.info("HParams {}".format(hparams))

  train_and_evaluate(
    output_directory=args.job_dir,
    train_dataset_directory=args.train_dataset_directory,
    eval_dataset_directory=args.eval_dataset_directory,
    train_steps=args.train_steps,
    train_parse_fns=parse_fns,
    eval_parse_fns=parse_fns,
    parallel_calls=[args.parallel_calls] * len(parse_fns),
    estimator_fn=estimator_fn,
    interleave_cycle_length=args.interleave_cycle_length,
    shuffle_buffer_size=args.shuffle_buffer_size,
    batch_size=args.batch_size,
    prefetch=args.prefetch,
    model_hparams=hparams,
  )


if __name__ == "__main__":
  main()