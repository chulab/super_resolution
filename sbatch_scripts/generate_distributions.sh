#!/bin/bash

#################################
# Runs Distribution Generation ##
#################################


sbatch_setup_commands="# Additional setup commands."

## JOB SPECIFICATIONS.
job_name=distribution_generation
now=$(date +"%FT%H%M%S")
directory=${PI_HOME}/job_logs/${now}_${job_name}

## JOB RUNTIME SPECIFICATIONS
time='1:00'
partition=normal
cpu=1
gpu_count=0

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
    -n|--job_name)
        if [ ! -z "$2" ]; then
            job_name=$2
            shift
        else
        echo 'ERROR: "--job_name" requires a non-empty option argument.'
        exit 1
        fi
    shift
    ;;
    -d|--directory)
        if [ ! -z "$2" ]; then
            directory=$2
            shift
        else
            echo 'ERROR: "--directory" requires a non-empty option argument.'
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
esac
done

if [ $gpu_count -gt 0 ]; then
    partition='gpu'
    sbatch_setup_commands+=$'\n'"#SBATCH --gres gpu:${gpu_count}"
fi

## SET UP JOB DIRECTORIES.
if [ ! -d directory ]; then
  # If directory does not exist, then creates it.
  mkdir -p $directory
fi

SBATCH_FILE="${directory}/sbatch_file.txt"

/bin/cat <<EOT >${SBATCH_FILE}
#!/bin/bash
## NAME
#SBATCH --job-name=${job_name}

## NOTIFICATION
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=toyonaga@stanford.edu

# A file for STDOUT from job.
#SBATCH --output="${directory}/output.txt"
# A file for STDERR from job.
#SBATCH --error="${directory}/error.txt"

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

python3.6 $PI_HOME/super_resolution/super_resolution/training_data/generate_scatterer_dataset.py \
-o $PI_SCRATCH/super_resolution/data/simulation/circle_dataset_3_14/distributions \
-n 'circle_3_14' \
-t 'CIRCLE' \
-s 2.5e-3 2.5e-3 \
-gd 5e-6 5e-6 \
-eps 30 \
-c 1000 \
-l .01 \
--min_radius 0. \
--max_radius 1.e-3 \
--max_count 10 \
--background_noise 0. \
--normalize False

python3.6 $PI_HOME/super_resolution/super_resolution/training_data/save_demo_image.py \
-f $PI_SCRATCH/super_resolution/data/simulation/circle_dataset_3_14/distributions/circle_3_14_1_of_34.npy \
-gd 5e-6 5e-6

EOT

sbatch ${SBATCH_FILE}
