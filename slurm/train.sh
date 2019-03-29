#!/bin/bash

##################
# Runs Training ##
##################

sbatch_setup_commands="# Additional setup commands."

## JOB SPECIFICATIONS.
job_name=training
now=$(date +"%FT%H%M%S")
job_directory=${PI_HOME}/job_logs/${now}_${job_name}

## JOB RUNTIME SPECIFICATIONS
time='1:00'
partition=normal
cpu=1
gpu_count=0

# TRAIN SPECIFIC ARGUMENTS
output_dir="trainer/test_output"
model_type="ANGLE_FIRST"
distribution_blur_sigma="1e-3"
observation_blur_sigma="1e-3"
distribution_downsample_size="100,100"
observation_downsample_size="100,100"
example_size="101,101"
train_steps="100"
train_dataset_directory="simulation/test_data/"
eval_dataset_directory="simulation/test_data/"
observation_spec_path="simulation/test_data/test_observation_spec.json"
learning_rate=".001"
prefetch=1
num_parallel_reads=1

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
            echo 'ERROR: "--gpu_count" requires non-empty option.'
            exit 1
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
    --output_dir)
        if [ ! -z "$2" ]; then
            output_dir=$2
            shift
        else
            echo 'ERROR: "--output_dir" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --distribution_blur_sigma)
        if [ ! -z "$2" ]; then
            distribution_blur_sigma=$2
            shift
        else
            echo 'ERROR: "--distribution_blur_sigma" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --observation_blur_sigma)
        if [ ! -z "$2" ]; then
            observation_blur_sigma=$2
            shift
        else
            echo 'ERROR: "--observation_blur_sigma" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --observation_downsample_size)
        if [ ! -z "$2" ]; then
            observation_downsample_size=$2
            shift
        else
            echo 'ERROR: "--observation_downsample_size" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --distribution_downsample_size)
        if [ ! -z "$2" ]; then
            distribution_downsample_size=$2
            shift
        else
            echo 'ERROR: "--distribution_downsample_size" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --example_size)
        if [ ! -z "$2" ]; then
            example_size=$2
            shift
        else
            echo 'ERROR: "--example_size" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --train_dataset_directory)
        if [ ! -z "$2" ]; then
            train_dataset_directory=$2
            shift
        else
            echo 'ERROR: "--train_dataset_directory" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --eval_dataset_directory)
        if [ ! -z "$2" ]; then
            eval_dataset_directory=$2
            shift
        else
            echo 'ERROR: "--eval_dataset_directory" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --observation_spec_path)
        if [ ! -z "$2" ]; then
            observation_spec_path=$2
            shift
        else
            echo 'ERROR: "--observation_spec_path" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --learning_rate)
        if [ ! -z "$2" ]; then
            learning_rate=$2
            shift
        else
            echo 'ERROR: "--learning_rate" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --train_steps)
        if [ ! -z "$2" ]; then
            train_steps=$2
            shift
        else
            echo 'ERROR: "--train_steps" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --model_type)
        if [ ! -z "$2" ]; then
            model_type=$2
            shift
        else
            echo 'ERROR: "--model_type" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --prefetch)
        if [ ! -z "$2" ]; then
            prefetch=$2
            shift
        else
            echo 'ERROR: "--prefetch" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --num_parallel_reads)
        if [ ! -z "$2" ]; then
            num_parallel_reads=$2
            shift
        else
            echo 'ERROR: "--num_parallel_reads" requires non-empty option.'
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
if [ ! -d ${output_dir} ]; then
  # If directory does not exist, then creates it.
  echo "ERROR: `output_dir` does not exist. Got ${output_dir}"
  exit 1
fi

if [ ! -d ${train_dataset_directory} ]; then
  # If directory does not exist, then creates it.
  echo 'ERROR: train_dataset_directory does not exist. Got ${train_dataset_directory}'
  exit 1
fi

if [ ! -d ${eval_dataset_directory} ]; then
  # If directory does not exist, then creates it.
  echo 'ERROR: eval_dataset_directory does not exist. Got ${eval_dataset_directory}'
  exit 1
fi

if [ ! -f ${observation_spec_path} ]; then
  # If directory does not exist, then creates it.
  echo 'ERROR: "observation_spec_path" does not exist. Got ${observation_spec_path}'
  exit 1
fi

## SET UP JOB DIRECTORIES.
if [ ! -d ${job_directory} ]; then
  # If directory does not exist, then creates it.
  echo "Job directory does not exist. Making: ${job_directory}"
  mkdir -p ${job_directory}
fi

export JOB_DIRECTORY=${job_directory}

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

$sbatch_setup_commands

ml py-scipy/1.1.0_py36
ml py-tensorflow/1.12.0_py36

python3.6 $PI_HOME/super_resolution/super_resolution/trainer/train.py \
--job-dir ${output_dir} \
--model_type ${model_type} \
--distribution_blur_sigma ${distribution_blur_sigma} \
--observation_blur_sigma ${observation_blur_sigma} \
--distribution_downsample_size ${distribution_downsample_size} \
--observation_downsample_size ${observation_downsample_size} \
--example_size ${example_size} \
--train_steps ${train_steps} \
--train_dataset_directory ${train_dataset_directory} \
--eval_dataset_directory ${eval_dataset_directory} \
--observation_spec_path ${observation_spec_path} \
--learning_rate ${learning_rate}
--prefetch ${prefetch}
--num_parallel_reads ${num_parallel_reads}
--slurm_train

EOT

sbatch ${SBATCH_FILE}
