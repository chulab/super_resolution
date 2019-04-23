#!/usr/bin/env bash

BASE_NAME=discretized

BUCKET=gs://chu_super_resolution_experiment

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

MODULE_NAME=trainer.train_discretized_model

CLOUD_DATA_TRAIN='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/train'
CLOUD_DATA_EVAL='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/eval'
CLOUD_OBSERVATION_SPEC='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/observation_spec.json'

# Train on Cloud.
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --staging-bucket $STAGING_BUCKET \
    --module-name $MODULE_NAME \
    --package-path trainer/ \
    --config cloud/config_discretized_hparameter_optimization.yaml \
    -- \
    --cloud_train \
    --mode TRAIN \
    --train_dataset_directory $CLOUD_DATA_TRAIN \
    --eval_dataset_directory $CLOUD_DATA_EVAL \
    --observation_spec_path $CLOUD_OBSERVATION_SPEC \
    --example_shape 501,501 \
    --train_steps 1500 \
    --batch_size 5 \
    --prefetch 1 \
    --interleave_cycle_length 8 \
    --num_parallel_reads 5 \
    --profile_steps 1000 \
    --log_step_count 20 \
        --hparams \
conv_blocks=4,\
spatial_blocks=5,\
spatial_kernel_size=5,\
spatial_scales=[1,2,4,8],\
filters_per_scale=16,\
residual_channels=64,\
residual_kernel_size=3,\
residual_scale=.1,\
bit_depth=4,\
observation_pool_downsample=8,\
distribution_pool_downsample=30,