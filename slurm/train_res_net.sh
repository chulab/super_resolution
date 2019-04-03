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
time='15:00:00'
partition=normal,hns
cpu=6
gpu_count=1

# TRAIN SPECIFIC ARGUMENTS
output_dir="trainer/test_output"
distribution_blur_sigma="1e-4"
observation_blur_sigma="1e-4"
example_shape="501,501"
train_steps="40000"
train_dataset_directory="/scratch/groups/schu/super_resolution/data/simulation/circle_3_18/train"
eval_dataset_directory="/scratch/groups/schu/super_resolution/data/simulation/circle_3_18/eval"
observation_spec_path="/scratch/groups/schu/super_resolution/data/simulation/circle_3_18/observation_spec.json"
hparams="learning_rate=.0001"
prefetch=5
parallel_calls=6
batch_size=5

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
    --example_shape)
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
    --hparams)
        if [ ! -z "$2" ]; then
            hparams=$2
            shift
        else
            echo 'ERROR: "--hparams" requires non-empty option.'
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
    --parallel_calls)
        if [ ! -z "$2" ]; then
            parallel_calls=$2
            shift
        else
            echo 'ERROR: "--parallel_calls" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --batch_size)
        if [ ! -z "$2" ]; then
            batch_size=$2
            shift
        else
            echo 'ERROR: "--batch_size" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
esac
done

if [ $gpu_count -gt 0 ]; then
    partition='gpu,hns'
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

python3.6 $PI_HOME/super_resolution/super_resolution/trainer/train_res_net.py \
--job-dir ${output_dir} \
--train_dataset_directory ${train_dataset_directory} \
--eval_dataset_directory ${eval_dataset_directory} \
--observation_spec_path ${observation_spec_path} \
--train_steps ${train_steps} \
--parallel_calls ${parallel_calls} \
--batch_size ${batch_size} \
--prefetch ${prefetch} \
--hparams ${hparams} \
--distribution_blur_sigma ${distribution_blur_sigma} \
--observation_blur_sigma ${observation_blur_sigma} \
--example_shape ${example_shape}
EOT

sbatch ${SBATCH_FILE}
