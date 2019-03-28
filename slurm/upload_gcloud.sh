#!/bin/bash

##################################
# Uploads files to google cloud ##
##################################

sbatch_setup_commands="# Additional setup commands."

## JOB SPECIFICATIONS.
job_name=upload_gcloud
now=$(date +"%FT%H%M%S")
job_directory=${PI_HOME}/job_logs/${now}_${job_name}

## JOB RUNTIME SPECIFICATIONS
time='15:00'
partition=normal
cpu=1
gpu_count=0

# TRAIN SPECIFIC ARGUMENTS
local_dir="."
cloud_dir="."

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
    -jn|--local_dir)
        if [ ! -z "$2" ]; then
            local_dir=$2
            shift
        else
        echo 'ERROR: "--job_name" requires a non-empty option argument.'
        exit 1
        fi
    shift
    ;;
    -jd|--cloud_dir)
        if [ ! -z "$2" ]; then
            cloud_dir=$2
            shift
        else
            echo 'ERROR: "--job_directory" requires a non-empty option argument.'
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
if [ ! -d ${local_dir} ]; then
  # If directory does not exist, then creates it.
  echo "ERROR: `local_dir` does not exist. Got ${local_dir}"
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

gsutil \
-m \
cp -r \
${local_dir} \
${cloud_dir}

EOT

echo ${SBATCH_FILE}

sbatch ${SBATCH_FILE}
