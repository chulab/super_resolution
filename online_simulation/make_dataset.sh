#!/usr/bin/env bash

OUTPUT_DIRECTORY='./test_dataset/'
EXAMPLE_COUNT=10
EXAMPLES_PER_SHARD=3


python online_simulation/dataset.py \
--output_directory $OUTPUT_DIRECTORY \
--examples_per_shard $EXAMPLES_PER_SHARD \
--example_count $EXAMPLE_COUNT \
--dataset_params \
"\
physical_dimension=0.0032,\
max_radius=1.5e-3,\
max_count=10,\
scatterer_density=1.e10\
" \
