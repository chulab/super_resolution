#!/bin/bash

#################################
# Runs Distribution Generation ##
#################################


sbatch_setup_commands="# Additional setup commands."

## JOB SPECIFICATIONS.
job_name=distribution_generation
now=$(date +"%FT%H%M%S")
job_directory=${PI_HOME}/job_logs/${now}_${job_name}

## JOB RUNTIME SPECIFICATIONS
time='1:00'
partition=normal
cpu=1
gpu_count=0

# DISTRIBUTION SIMULATION SPECIFIC ARGUMENTS
dataset_name=circle_test
output_directory=${PI_SCRATCH}/super_resolution/data/simulation/${dataset_name}/distributions
type=CIRCLE
size=2.5e-3
grid_dimension=5e-6
examples_per_shard=30
count=100
lambda=.01
min_radius=0.
max_radius=1.e-3
max_count=10
background_noise=0.

## GET INPUTS.
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

## OVERRIDE DEFAULTS WITH PASSED PARAMETERS
case $key in
    -h|-\?|--help)
        Help.
    shift
    ;;
    -jn|--job_name)
        if [ ! -z "$2" ]; then
            job_name=$2
            shift
        else
        echo 'ERROR: "--job_name" requires a non-empty option argument.'
        exit 1
        fi
    shift
    ;;
    -jd|--job_directory)
        if [ ! -z "$2" ]; then
            job_directory=$2
            shift
        else
            echo 'ERROR: "--job_directory" requires a non-empty option argument.'
            exit 1
        fi
    shift
    ;;
    -g|--gpu_count)
        if [ ! -z "$2" ]; then
            gpu_count=$2
            shift
        else
            gpu_count=1
            echo '"gpu_count" set to 1 as no value was given.'
        fi
    shift
    ;;
    -t|--time)
        if [ ! -z "$2" ]; then
            time=$2
            shift
        else
            echo 'ERROR: "--time" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    -c|--cpu)
        if [ ! -z "$2" ]; then
            cpu=$2
            shift
        else
            echo 'ERROR: "--cpu" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    -dn|--dataset_name)
        if [ ! -z "$2" ]; then
            dataset_name=$2
            shift
        else
            echo 'ERROR: "--datset_name" requires a non-empty option argument.'
            exit 1
        fi
    shift
    ;;
esac
done

if [ $gpu_count -gt 0 ]; then
    partition='gpu'
    sbatch_setup_commands+=$'\n'"#SBATCH --gres gpu:${gpu_count}"
fi

## CHECK ARGUMENTS.
if [ ! -d ${output_directory} ]; then
  # If directory does not exist, then creates it.
  echo "ERROR: `output_directory` does not exist. Got ${output_directory}"
  exit 1
fi

## SET UP JOB DIRECTORIES.
if [ ! -d ${job_directory} ]; then
  # If directory does not exist, then creates it.
  echo "Job directory does not exist. Making: ${job_directory}"
  mkdir -p ${job_directory}
fi

SBATCH_FILE="${job_directory}/sbatch_file.txt"

/bin/cat <<EOT >${SBATCH_FILE}
#!/bin/bash
## JOB SUBMITTED AT ${now}
## NAME
#SBATCH --job-name=${job_name}

## NOTIFICATION
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=toyonaga@stanford.edu

# A file for STDOUT from job.
#SBATCH --output="${job_directory}/output.txt"
# A file for STDERR from job.
#SBATCH --error="${job_directory}/error.txt"

##  RUNTIME RESOURCES.
# Note that if gpu's are requested, the call to gres is included in
# the appended 'sbatch_setup_commands'.
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --cpus-per-task=${cpu}
#SBATCH --mem-per-cpu=4G

$sbatch_setup_commands

ml py-numpy/1.14.3_py36
ml py-scipy/1.1.0_py36

ml viz
ml py-matplotlib/2.2.2_py36

python3.6 $PI_HOME/super_resolution/super_resolution/training_data/generate_scatterer_dataset.py \
-o ${output_directory}
-n ${dataset_name} \
-t ${type} \
-s ${size} ${size} \
-gd ${grid_dimension} ${grid_dimension}\
-eps ${examples_per_shard} \
-c ${count} \
-l ${lambda} \
--min_radius ${min_radius} \
--max_radius ${max_radius} \
--max_count ${max_count} \
--background_noise ${background_noise} \

python3.6 $PI_HOME/super_resolution/super_resolution/training_data/save_demo_image.py \
-f ${output_directory}/${dataset_name}_00001_* \
-gd ${grid_dimension} ${grid_dimension}

EOT

sbatch ${SBATCH_FILE}