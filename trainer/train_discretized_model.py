"""Main module for training angle-only model."""

import argparse
import logging
import os
import sys

import tensorflow as tf

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import create_observation_spec

from trainer import discretized_model as model
from trainer import train

from utils import logging_utils

_PREDICT="PREDICT"
_TRAIN="TRAIN"

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--mode',
    help='Either `train` or `predict`',
    default=_TRAIN
  )

  parser.add_argument(
    '--job-dir',
    help='Path to save test_output of model including checkpoints.',
    type=str,
    required=True
  )

  parser.add_argument(
    '--observation_spec_path',
    type=str,
    required=True,
  )

  parser.add_argument(
    '--hparams',
    type=str,
    help='Comma separated list of "name=value" pairs.',
    default=''
  )

  parser.add_argument(
    '--warm_start_from_dir',
    type=str,
    help='Checkpoint directory to warm start from.',
    default=''
  )

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

  ## HPARAMETER TUNING ARGS
  parser.add_argument(
    '--learning_rate',
    type=float,
  )
  parser.add_argument(
    '--decay_rate',
    type=float,
  )
  parser.add_argument(
    '--conv_blocks',
    type=int,
  )
  parser.add_argument(
    '--spatial_blocks',
    type=int,
  )
  parser.add_argument(
    '--filters_per_scale',
    type=int,
  )
  parser.add_argument(
    '--residual_blocks',
    type=int,
  )
  parser.add_argument(
    '--residual_channels',
    type=int,
  )


  args, _ = parser.parse_known_args()

  return args


def main():

  args = parse_args()

  logging_utils.set_up_logging()

  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path, args.cloud_train
  )

  parse_fns = model.input_fns()

  hparams = model.make_hparams()
  hparams.parse(args.hparams)
  # Must manually replace `observation_spec`.
  hparams.observation_spec=observation_spec

  if args.learning_rate is not None:
    hparams.learning_rate = args.learning_rate
    hparams.decay_rate = args.decay_rate
    hparams.conv_blocks = args.conv_blocks
    hparams.spatial_blocks = args.spatial_blocks
    hparams.filters_per_scale = args.filters_per_scale
    hparams.residual_blocks = args.residual_blocks
    hparams.residual_channels = args.residual_channels

  logging.info("HParams {}".format(hparams))

  # Optionally warm start variables.
  if args.warm_start_from_dir != '':
    warm_start_from = tf.estimator.WarmStartSettings(
      args.warm_start_from_dir,
    )
  else:
    warm_start_from=None

  if args.mode==_TRAIN:
    train.run_train_and_evaluate(
      output_directory=args.job_dir,
      model_fn=model.model_fn,
      hparams=hparams,
      train_parse_fns=parse_fns,
      eval_parse_fns=parse_fns,
      warm_start_from=warm_start_from,
    )

  if args.mode==_PREDICT:
    pass


if __name__ == "__main__":
  main()