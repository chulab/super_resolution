#!/usr/bin/env bash

## Convenience function to define enviromental variables for dataset.

dataset_directory=''

train_directory_prefix='train'
eval_directory_prefix='eval'
observation_spec_filename='observation_spec.json'

# Load CLI arguments.

## GET INPUTS.
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

## OVERRIDE DEFAULTS WITH PASSED PARAMETERS
case $key in
    -d|--dataset_directory)
        if [ ! -z "$2" ]; then
            dataset_directory=$2
            shift
        else
        echo 'ERROR: "--dataset_directory" requires a non-empty option argument.'
        exit 1
        fi
    shift
    ;;
esac
done

train_directory=${dataset_directory}/${train_directory_prefix}
eval_directory=${dataset_directory}/${eval_directory_prefix}
observation_spec_path=${dataset_directory}/${observation_spec_filename}

# Check directories exist.
if [ ! -d ${train_directory} ]; then
  # If directory does not exist, then creates it.
  echo "ERROR: train_directory does not exist. Got ${train_directory}"
  exit 1
fi

if [ ! -d ${eval_directory} ]; then
  # If directory does not exist, then creates it.
  echo "ERROR: train_directory does not exist. Got ${eval_directory}"
  exit 1
fi

if [ ! -f ${observation_spec_path} ]; then
  # If directory does not exist, then creates it.
  echo "ERROR: observation_spec_path does not exist. Got ${observation_spec_path}"
  exit 1
fi

# EXPORT VARIABLES.
export TRAIN_DATASET=${train_directory}
echo "Set \`TRAIN_DATASET\` to ${train_directory}"

export EVAL_DATASET=${eval_directory}
echo "Set \`EVAL_DATASET\` to ${eval_directory}"

export OBSERVATION_SPEC=${observation_spec_path}
echo "Set \`OBSERVATION_SPEC\` to ${observation_spec_path}"