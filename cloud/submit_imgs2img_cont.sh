#!/usr/bin/env bash

BASE_NAME=imgs2img_cont

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

MODULE_NAME=trainer.train_imgs2img_cont

CLOUD_DATA_TRAIN='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/train'
CLOUD_DATA_EVAL='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/eval'
CLOUD_OBSERVATION_SPEC='gs://chu_super_resolution_data/simulation/circle_3_18_envelope/observation_spec.json'
LOCAL_OUTPUT='trainer/test_output'


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
    --train_steps 2000 \
    --batch_size 5 \
    --prefetch 2 \
    --interleave_cycle_length 6 \
    --num_parallel_reads 5 \
    --profile_steps 1000 \
    --log_step_count 200 \
    --frequency_indices 0,1,2,3,4,5,6,7 \
        --hparams \
learning_rate=.0001,\
observation_pool_downsample=30,\
distribution_pool_downsample=30,\
bit_depth=4,\
bets=None,\
rewards=1/n,\
scale_steps=5000,\
num_encoder_layers=4,\
num_decoder_layers=4,\
hidden_size=512,\

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
#     --profile_steps 1000 \
#     --log_step_count 200 \
#     --frequency_indices 0,1 \
#     --hparams \
# learning_rate=.00001,\
# observation_pool_downsample=10,\
# distribution_pool_downsample=10,\
# bit_depth=4,\
# bets=1/n,\
# rewards=1/n,\
# scale_steps=5,\
# num_encoder_layers=4,\
# num_decoder_layers=4,\
#
# rm -r $LOCAL_OUTPUT
