#!/usr/bin/env bash


BASE_NAME=online_recurrent

BUCKET=gs://chu_super_resolution_experiment

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

MODULE_NAME=online_simulation.train_online_recurrent_model

#CONFIG=cloud/sweep_simulation.yaml
CONFIG=cloud/config_gpu.yaml


#WARM_START_FROM='gs://chu_super_resolution_experiment/online_simulation_19_04_2019_17_44_58/23/model.ckpt-2000'

SUMMARY_NAME='online_recurrent_15_07_2019_23_48_11'
SUMMARY_OUTPUT=${BUCKET}"/"${SUMMARY_NAME}

# Train locally
gcloud ml-engine local train\
  --job-dir $SUMMARY_OUTPUT \
  --module-name $MODULE_NAME \
  --package-path online_simulation/ \
  -- \
  --mode TRAIN \
  --dataset_params \
"\
physical_dimension=3e-3,\
max_radius=1.5e-3,\
max_count=10\
" \
  --model_params \
"\
bit_depth=4,\
learning_rate=.0001,\
bets=None,\
rewards=1/n,\
scale_steps=5000,\
unet_height=4,\
initial_filters=200,\
unet_type=attention_vanilla,\
recurrent=False,\
num_split=2,\
diff_scale=abs,\
last_loss_only=False,\
embedding=conv_se_attention,\
" \
  --simulation_params \
"" \
  --train_params \
"eval_steps=100,\
profile_steps=1000" \
  --train_steps 4000 \
  --learning_rate .0001 \
  --angle_count 20 \
  --angle_limit 90 \
  --frequency_count 8
