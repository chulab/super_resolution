#!/usr/bin/env bash
BASE_NAME=online_builder

BUCKET=gs://chu_super_resolution_experiment
STAGING_BUCKET=$BUCKET
NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
ONLINE_JOB_DIR=${BUCKET}"/"${JOB_NAME}

LOCAL_OUTPUT='online_simulation/test_output'

MODULE_NAME=online_simulation.train_online_model_builder

# CONFIG=cloud/sweep_simulation.yaml
CONFIG=cloud/config_gpu_3.yaml
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


# Train on Cloud.
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
max_count=10,\
grid_dimension=5e-6,\
" \
    --model_params \
"\
downsample_bits=5,\
learning_rate=.001,\
bit_depth=8,\
bets=-logn,\
rewards=1/sqrtn,\
scale_steps=5000,\
forward_height=4,\
reverse_height=4,\
model_type=prop,\
prop_layers=[3, 3, 3],\
forward_prop=residual,\
mid_prop=residual,\
reverse_prop=residual,\
pooler=residual,\
recurrent=False,\
diff_scale=abs,\
last_loss_only=False,\
concat_avg=True,\
embedding=xception-64_3/attention,\
" \
    --pooler_filters 64,64 \
    --scatterer_density 2e12 \
    --simulation_params \
"" \
    --min_frequency 1e6 \
    --max_frequency 3e6 \
    --train_params \
"eval_steps=200,\
profile_steps=1000000,\
log_step_count=20,\
" \
    --train_steps 2500 \
    --learning_rate .001 \
    --angle_count 16 \
    --angle_limit 90 \
    --frequency_count 8 \


# LOCAL_OUTPUT='online_simulation/test_output'

# Train locally
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
# max_count=10,\
# grid_dimension=1e-5,\
# " \
#    --model_params \
# "\
# learning_rate=.001,\
# bit_depth=8,\
# bets=-logn,\
# rewards=1/sqrtn,\
# scale_steps=5000,\
# forward_height=4,\
# reverse_height=4,\
# model_type=prop,\
# prop_layers=[2, 2, 2],\
# forward_prop=xception,\
# mid_prop=residual,\
# reverse_prop=residual,\
# pooler=residual,\
# recurrent=False,\
# diff_scale=abs,\
# last_loss_only=False,\
# concat_avg=True,\
# embedding=xception-12/xception-12,\
# " \
#    --pooler_filters 4,8 \
#    --forward_kwargs '{"conv_name": "sepconv", "se_block": true}' \
#    --scatterer_density 2e12 \
#    --min_frequency 2.5e6 \
#    --max_frequency 5e6 \
#    --simulation_params "" \
#    --train_steps 10 \
#    --train_params \
# "train_steps=10,\
# eval_steps=10,\
# profile_steps=1000" \
#    --angle_count 4 \
#    --angle_limit 90 \
#    --frequency_count 2 \
#    --mode_count 1 \
#
# rm -r $LOCAL_OUTPUT/*
