#!/usr/bin/env bash
BASE_NAME=online_simulation

BUCKET=gs://chu_super_resolution_experiment
STAGING_BUCKET=$BUCKET
NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
ONLINE_JOB_DIR=${BUCKET}"/"${JOB_NAME}

LOCAL_OUTPUT='online_simulation/test_output'

MODULE_NAME=online_simulation.train_online_simulation_model

# CONFIG=cloud/sweep_simulation.yaml
CONFIG=cloud/config_gpu.yaml
# CONFIG=cloud/multi_gpu_config.yaml

# CONFIG=online_simulation/sweep_scatterer_density.yaml

# MODE="online"
# # MODE="local"
#
# if [ "$MODE" == "local" ]; then
#   JOB_DIR=$LOCAL_OUTPUT
# if [ "$MODE" == "online" ]; then
#   JOB_DIR=$ONLINE_JOB_DIR
# fi


#Train on Cloud.
JOB_DIR=$ONLINE_JOB_DIR
gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --staging-bucket $STAGING_BUCKET \
    --module-name $MODULE_NAME \
    --package-path online_simulation/ \
    --config $CONFIG \
    -- \
    --mode TRAIN \
    --dataset_params \
"\
physical_dimension=0.0032,\
max_radius=1.5e-3,\
max_count=10\
" \
    --model_params \
"\
bit_depth=4,\
log_steps=200,\
decay_step=1000\
" \
    --simulation_params \
"" \
    --train_params \
"eval_steps=200,\
profile_steps=1000000,\
log_step_count=10,\
" \
    --train_steps 100 \
    --learning_rate .0005 \
    --angle_count 10 \
    --angle_limit 90 \
    --frequency_count 8


# LOCAL_OUTPUT='online_simulation/test_output'
#
# # Train locally
# gcloud ai-platform local train\
#    --job-dir $LOCAL_OUTPUT \
#    --module-name $MODULE_NAME \
#    --package-path online_simulation/ \
#    -- \
#    --mode TRAIN \
#    --dataset_params \
# "\
# physical_dimension=0.0032,\
# max_radius=1.5e-3,\
# max_count=10\
# " \
#    --model_params \
# "\
# bit_depth=4,\
# decay_step=1000\
# " \
#    --simulation_params "" \
#    --train_params \
# "train_steps=10000,\
# eval_steps=75,\
# profile_steps=1000" \
#    --angle_count 1 \
#    --angle_limit 90 \
#    --frequency_count 1 \
#    --mode_count 1 \
#
# rm -r $LOCAL_OUTPUT/*
