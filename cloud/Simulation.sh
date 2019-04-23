#!/usr/bin/env bash

OUTPUT_DIRECTORY='/home/stevechulab/data/simulation/line_4_15/simulations'
DISTRIBUTION_PATH='/home/stevechulab/data/simulation/line_4_15/distributions'
WORKER_COUNT=16
OBSERVATION_SPEC_PATH='/home/stevechulab/data/simulation/line_4_15/observation_spec.json'
TRANSVERSE_PSF_LENGTH=2.5e-3
AXIAL_PSF_LENGTH=2.e-3
DATASET_NAME_PREFIX='line_4_15'
EXAMPLES_PER_SHARD=10


JOB_DIRECTORY='/home/stevechulab/log_data/line_4_15_simulation/'


python simulation/run_simulation.py -o $OUTPUT_DIRECTORY -d $DISTRIBUTION_PATH -w $WORKER_COUNT -os $OBSERVATION_SPEC_PATH -tpsf $TRANSVERSE_PSF_LENGTH -apsf $AXIAL_PSF_LENGTH -n $DATASET_NAME_PREFIX -eps $EXAMPLES_PER_SHARD