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

from analysis import plot_utils

from preprocessing import input
from preprocessing import parser

from simulation import create_observation_spec

from trainer import model as straight_model
from trainer import angle_first_model


_STRAIGHT="STRAIGHT"
_ANGLE_FIRST="ANGLE_FIRST"

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('-o', '--output_dir',
                      dest='output_dir',
                      help='Path to save output.',
                      type=str,
                      required=True)

  parser.add_argument('--model_dir',
                      dest='model_dir',
                      help='Path to model checkpoints.',
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
    '--predict_dataset_directory',
    dest='predict_dataset_directory',
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
    required=True
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

  parser.add_argument('--plot_all_observations', dest='plot_all_observations', action='store_true')
  parser.set_defaults(plot_all_observations=False)

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

  predict_parse_fn = parser.Parser(
    observation_spec=observation_spec,
    reverse_rotation=True,
    distribution_blur_sigma=args.distribution_blur_sigma,
    observation_blur_sigma=args.observation_blur_sigma,
    distribution_downsample_size=args.distribution_downsample_size,
    observation_downsample_size=args.observation_downsample_size,
    example_size=args.example_size,
  ).parse
  logging.info("Initialized `predict_parse_fn`.")

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

  # Set up input
  predict_dataset = input.input_fn(
    args.predict_dataset_directory,
    predict_parse_fn,
    1,
    1,
    1,
    1,
  )
  iterator = predict_dataset.make_one_shot_iterator()
  observations, distribution = iterator.get_next()

  # Rebuild the model
  predictions = model.model_fn(
    observations, distribution, tf.estimator.ModeKeys.EVAL, hparams).predictions

  # Manually load the latest checkpoint
  saver = tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(args.model_dir)
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

  # Set up output dir.
  output_dir = os.path.join(args.output_dir, 'prediction')
  os.makedirs(output_dir, exist_ok=True)

  plot_utils.plot_observation_prediction__distribution(
    output[0]['observations'][0, 0, ..., 0],
    output[0]['distribution'][0],
    output[0]['predictions'][0],
    observation_spec.grid_dimension,
    output_dir)


  if args.plot_all_observations:
    plot_utils.plot_observation_example(
      output[0]['observations'][0], observation_spec, output_dir,
    )


if __name__ == "__main__":
  main()