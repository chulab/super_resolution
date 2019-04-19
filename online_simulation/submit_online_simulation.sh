#!/usr/bin/env bash


BASE_NAME=online_simulation

BUCKET=gs://chu_super_resolution_experiment

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

MODULE_NAME=online_simulation.train_online_simulation_model

# Train on Cloud.
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --staging-bucket $STAGING_BUCKET \
    --module-name $MODULE_NAME \
    --package-path online_simulation/ \
    --config online_simulation/sweep_simulation.yaml \
    -- \
    --cloud_train \
    --mode TRAIN \
    --dataset_params \
"\
physical_dimension=3e-3,\
max_radius=1.5e-3\
" \
    --model_params \
"\
bit_depth=4\
" \
    --simulation_params \
"" \
    --train_params \
"train_steps=2000,\
eval_steps=75,\
profile_steps=500"



#LOCAL_OUTPUT='online_simulation/test_output'
#
## Train locally
#gcloud ml-engine local train\
#    --job-dir $LOCAL_OUTPUT \
#    --module-name $MODULE_NAME \
#    --package-path online_simulation/ \
#    -- \
#    --mode TRAIN \
#    --dataset_params \
#"\
#physical_dimension=3e-3,\
#max_radius=1.5e-3\
#" \
#    --model_params \
#"\
#bit_depth=4\
#" \
#    --simulation_params \
#"" \
#    --train_params \
#"train_steps=2000,\
#eval_steps=75,\
#profile_steps=500" \
#
#rm -r $LOCAL_OUTPUT/

