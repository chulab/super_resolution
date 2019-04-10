#!/usr/bin/env bash

BASE_NAME=discretized

BUCKET=gs://chu_super_resolution_experiment

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

MODULE_NAME=trainer.train_discretized_model

# Train on Cloud.
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --staging-bucket $STAGING_BUCKET \
    --module-name $MODULE_NAME \
    --package-path trainer/ \
    --config cloud/config_gpu.yaml \
    -- \
    --cloud_train \
    --mode TRAIN \
    --train_dataset_directory $CLOUD_DATA_TRAIN \
    --eval_dataset_directory $CLOUD_DATA_EVAL \
    --observation_spec_path $CLOUD_OBSERVATION_SPEC \
    --pool_downsample 10 \
    --example_shape 501,501 \
    --train_steps 2000 \
    --batch_size 5 \
    --prefetch 2 \
    --interleave_cycle_length 6 \
    --num_parallel_reads 5 \
    --profile_steps 500 \
    --log_step_count 20 \
        --hparams \
learning_rate=.001,\
conv_blocks=2,\
spatial_blocks=5,\
spatial_kernel_size=5,\
spatial_scales=[1,2,4,8,16],\
residual_blocks=8,\
residual_channels=64,\
residual_kernel_size=3,\
residual_scale=.1,\
bit_depth=2,\


#
## Train locally
#gcloud ml-engine local train\
#    --job-dir trainer/test_output \
#    --module-name $MODULE_NAME \
#    --package-path trainer/ \
#    -- \
#    --mode TRAIN \
#    --train_dataset_directory simulation/test_data\
#    --eval_dataset_directory simulation/test_data \
#    --observation_spec_path simulation/test_data/test_observation_spec.json \
#    --example_shape 101,101 \
#    --train_steps 10 \
#    --batch_size 5 \
#    --prefetch 2 \
#    --interleave_cycle_length 6 \
#    --num_parallel_reads 5 \
#    --profile_steps 20 \
#    --log_step_count 1 \
#    --hparams \
#learning_rate=.0001,\
#conv_blocks=2,\
#spatial_blocks=5,\
#spatial_kernel_size=5,\
#spatial_scales=[1,2,4,8,16],\
#residual_blocks=8,\
#residual_channels=64,\
#residual_kernel_size=3,\
#residual_scale=.1,\
#bit_depth=2,\

