#!/usr/bin/env bash

BASE_NAME=basic_job

BUCKET=gs://chu_super_resolution_experiment

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --staging-bucket $STAGING_BUCKET \
    --module-name trainer.train_basic_model\
    --package-path trainer/ \
    --config cloud/config_hparameter_optimization.yaml \
    -- \
    --cloud_train \
    --mode TRAIN \
    --train_dataset_directory $CLOUD_DATA_TRAIN \
    --eval_dataset_directory $CLOUD_DATA_EVAL \
    --observation_spec_path $CLOUD_OBSERVATION_SPEC \
    --pool_downsample 10 \
    --example_shape 501,501 \
    --train_steps 10000 \
    --batch_size 5 \
    --prefetch 2 \
    --interleave_cycle_length 5 \
    --num_parallel_reads 5 \
    --profile_steps 400 \
    --log_step_count 20 \
    --hparams learning_rate=.001,residual_blocks=8,residual_channels=64,residual_kernel_size=4,residual_scale=.1,spatial_blocks=4,spatial_kernel_size=4,spatial_scales=[2,4,8,16]