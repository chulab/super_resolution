#!/usr/bin/env bash

BASE_NAME=imgs2img

BUCKET=gs://chu_super_resolution_experiment

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
SUFFIX=

JOB_NAME=${BASE_NAME}_${NOW}_${SUFFIX}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

MODULE_NAME=trainer.train_imgs2img_model

CLOUD_DATA_TRAIN=gs://chu_super_resolution_data/simulation/circle_3_18/train
CLOUD_DATA_EVAL=gs://chu_super_resolution_data/simulation/circle_3_18/eval
CLOUD_OBSERVATION_SPEC=gs://chu_super_resolution_data/simulation/circle_3_18/observation_spec.json

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
    --example_shape 501,501 \
    --train_steps 1500 \
    --batch_size 5 \
    --prefetch 2 \
    --interleave_cycle_length 6 \
    --num_parallel_reads 5 \
    --profile_steps 1000 \
    --log_step_count 20 \
        --hparams \
learning_rate=.00001,\
observation_pool_downsample=30,\
distribution_pool_downsample=30,\
bit_depth=4,\


LOCAL_OUTPUT='trainer/test_output'

# Train locally
# gcloud ml-engine local train\
#    --job-dir $LOCAL_OUTPUT \
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
# bit_depth=2,\
# learning_rate=.0001,\
# observation_pool_downsample=10,\
# distribution_pool_downsample=10,\
#
# rm -r $LOCAL_OUTPUT
