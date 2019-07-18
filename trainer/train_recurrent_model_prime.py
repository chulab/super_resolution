"""Main module for training angle-only model."""

import argparse
import logging
import os
import sys

import tensorflow as tf
import numpy as np
from analysis import plot_utils, recorder_utils, summaries
from cloud_utils import save_utils
from oauth2client.client import GoogleCredentials

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import create_observation_spec

from trainer import recurrent_model_prime as model
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
    help='Path to save output of model including checkpoints.',
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

  parser.add_argument(
    '--train_steps',
    type=int,
    required=True
  )

  parser.add_argument(
    '--service_account_path',
    type=str,
    help='path to service account credentials for Google utilities',
    default='gs://chu_super_resolution_data/service-account.json'
  )

  parser.add_argument(
    '--slide_id',
    type=str,
    help='Google slide id for summary',
    default='187e6QMPZApaDQoqC5v_BqQ4Y0mZUsUD1VkviSu_hCnY'
  )

  parser.add_argument(
    '--frequency_indices',
    type=lambda s: [int(index) for index in s.split(',')],
    required=True,
    default='0'
  )

  parser.add_argument(
    '--angle_indices',
    type=lambda s: [int(index) for index in s.split(',')],
    required=True,
    default='0'
  )

  parser.add_argument(
    '--example_shape',
    type=lambda s: [int(size) for size in s.split(',')],
    required=True,
  )

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

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
  hparams.add_hparam("job_dir", args.job_dir)
  hparams.add_hparam("frequency_indices", args.frequency_indices)
  hparams.add_hparam("angle_indices", args.angle_indices)


  # Must manually replace `observation_spec`.
  hparams.observation_spec=observation_spec
  hparams.example_shape = args.example_shape[0] // hparams.observation_pool_downsample

  logging.info("HParams {}".format(hparams))

  # Optionally warm start variables.
  if args.warm_start_from_dir != '':
    warm_start_from = tf.estimator.WarmStartSettings(
      args.warm_start_from_dir,
    )
  else:
    warm_start_from=None

  if "recurrent" in hparams.recurrent:
    model_fn = model.model_fn_recurrent
  else:
    model_fn = model.model_fn

  if args.mode==_TRAIN:
    train.run_train_and_evaluate(
      output_directory=args.job_dir,
      model_fn=model_fn,
      hparams=hparams,
      train_parse_fns=parse_fns,
      eval_parse_fns=parse_fns,
      warm_start_from=warm_start_from,
    )

    tb_dir = args.job_dir + "/eval"
    if args.cloud_train:
      save_dir = args.job_dir
    else:
      save_dir = "gs://chu_super_resolution_experiment/test_output"

    steps, _ = plot_utils.get_all_tensor_from_tensorboard(tb_dir, 'predictions/distribution_tensor')
    steps = sorted(list(dict.fromkeys(steps)))
    summaries.summarize_template_2(args, observation_spec, hparams, steps, tb_dir, save_dir)

  if args.mode==_PREDICT:
    pass


if __name__ == "__main__":
  main()
