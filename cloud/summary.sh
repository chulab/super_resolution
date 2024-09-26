#!/usr/bin/env bash

BASE_NAME=imgs2img

BUCKET=gs://chu_super_resolution_experiment

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
# NOW=25_06_2019_16_52_14
SUFFIX=
# PRIME=25

JOB_NAME=${BASE_NAME}_${NOW}_${SUFFIX}
JOB_DIR=${BUCKET}"/"${JOB_NAME}
# JOB_NAME_PRIME=${JOB_NAME}${PRIME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

MODULE_NAME=trainer.train_imgs2img_model

CLOUD_DATA_TRAIN='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/train'
CLOUD_DATA_EVAL='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/eval'
CLOUD_OBSERVATION_SPEC='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/observation_spec.json'
LOCAL_OUTPUT='trainer/test_output'

SUMMARY_NAME='imgs2img_28_06_2019_18_00_19_'
SUMMARY_OUTPUT=${BUCKET}"/"${SUMMARY_NAME}
# Train locally
gcloud ml-engine local train\
   --job-dir $SUMMARY_OUTPUT \
   --module-name $MODULE_NAME \
   --package-path trainer/ \
   -- \
   --cloud_train \
   --mode TRAIN \
   --train_dataset_directory $CLOUD_DATA_TRAIN\
   --eval_dataset_directory $CLOUD_DATA_EVAL \
   --observation_spec_path $CLOUD_OBSERVATION_SPEC \
   --example_shape 501,501 \
   --train_steps 2000 \
   --batch_size 32 \
   --prefetch 2 \
   --interleave_cycle_length 6 \
   --num_parallel_reads 5 \
    --profile_steps 1000 \
    --log_step_count 200 \
    --hparams \
learning_rate=.00001,\
observation_pool_downsample=30,\
distribution_pool_downsample=30,\
bit_depth=4,\
bets=-logn,\
rewards=1/sqrtn,\
