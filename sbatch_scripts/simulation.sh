#!/bin/bash

####################
# Runs Simulation ##
####################


sbatch_setup_commands="# Additional setup commands."

## JOB SPECIFICATIONS.
job_name=simulation
now=$(date +"%FT%H%M%S")
job_directory=${PI_HOME}/job_logs/${now}_${job_name}

## JOB RUNTIME SPECIFICATIONS
time='1:00'
partition=normal
cpu=1
mem_per_cpu=8G
gpu_count=0

# SIMULATION SPECIFIC ARGUMENTS
output_directory=''
distribution_path=''
worker_count=$((10*${cpu})) # Based on sherlock specs.
observation_spec_path=''
transverse_psf_length=2.e-3
axial_psf_length=1.3e-3
dataset_name_prefix=simulation
examples_per_shard=10

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
    -d|--job_directory)
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
    -m|--mem_per_cpu)
        if [ ! -z "$2" ]; then
            mem_per_cpu=$2
            shift
        else
            echo 'ERROR: "--mem_per_cpu" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --output_directory)
        if [ ! -z "$2" ]; then
            output_directory=$2
            shift
        else
            echo 'ERROR: "--output_directory" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --distribution_path)
        if [ ! -z "$2" ]; then
            distribution_path=$2
            shift
        else
            echo 'ERROR: "--distribution_path" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --observation_spec_path)
        if [ ! -z "$2" ]; then
            observation_spec_path=$2
            shift
        else
            echo 'ERROR: "--output_directory" requires non-empty option.'
            exit 1
        fi
    shift
    ;;
    --worker_count)
        if [ ! -z "$2" ]; then
            worker_count=$2
            shift
        else
            echo 'ERROR: "--worker_count" requires non-empty option.'
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

## CHECK DIRECTORIES.
if [ ! -d ${output_directory} ]; then
  # If directory does not exist, then creates it.
  echo "ERROR: output_directory does not exist. Got ${output_directory}"
  exit 1
fi

if [ ! -d ${distribution_path} ]; then
  # If directory does not exist, then creates it.
  echo 'ERROR: distribution_path does not exist. Got ${distribution_path}'
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
#SBATCH --mem-per-cpu=${mem_per_cpu}

$sbatch_setup_commands

ml py-numpy/1.14.3_py36
ml py-scipy/1.1.0_py36

ml viz
ml py-matplotlib/2.2.2_py36

python3.6 $PI_HOME/super_resolution/super_resolution/simulation/run_simulation.py \
-o ${output_directory} \
-d ${distribution_path} \
-w ${worker_count} \
-os ${observation_spec_path} \
-tpsf ${transverse_psf_length} \
-apsf ${axial_psf_length} \
-n ${dataset_name_prefix} \
-eps ${examples_per_shard}

python3.6 $PI_HOME/super_resolution/super_resolution/simulation/plot_simulations.py \
-f ${output_directory}/test_simulation_0000000 \
-os ${observation_spec_path}
EOT

sbatch ${SBATCH_FILE}