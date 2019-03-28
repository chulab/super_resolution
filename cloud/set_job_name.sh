#!/usr/bin/env bash

base_name=$1

NOW=$(date '+%d_%m_%Y_%H_%M_%S')

export JOB_NAME=${base_name}_${NOW}
export JOB_DIR=gs://chu_super_resolution_experiment/${JOB_NAME}