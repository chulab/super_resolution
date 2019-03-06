#!/bin/bash
sbatch_setup_commands="# Additional setup commands. \n"

## BORG JOB SPECIFICATIONS.
job_name=test
directory=output

## JOB RUNTIME SPECIFICATIONS
time='1:00'
partition=normal
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
    ;;
    -n|--job_name)
    if [ ! -z "$2" ]; then
            job_name=$2
            shift
    else
    echo 'ERROR: "--job_name" requires a non-empty option argument.'
    exit 1
    fi
    ;;
    -d|--directory)
        if [ ! -z "$2" ]; then
            directory=$2
            shift
        else
            echo 'ERROR: "--directory" requires a non-empty option argument.'
    exit 1
        fi
        ;;
    -g|--gpu_count)
    if [ ! -z "$2" ]; then
            gpu_count=$2
            shift
        else
            gpu_count=1
    echo '"gpu_count" set to 1 as no value was given.'
        fi
        ;;
-?*)
    die "WARNING: Unkown option: $1"
    ;;
*)
    break
esac
done

if [ $gpu_count -gt 0 ]; then
  partition='gpu'
  sbatch_setup_commands="${sbatch_setup_commands}#SBATCH --gres gpu:${gpu_count} \n"
fi

## SET UP JOB DIRECTORIES.
if [ ! -d directory ]; then
  # If directory does not exist, then creates it.
  mkdir -p $directory
fi

echo $sbatch_setup_commands

sbatch <<EOT
#!/bin/bash

## NAME
#SBATCH --job-name=${job_name}

## NOTIFICATION
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=toyonaga@stanford.edu

# A file for STDOUT from job.
#SBATCH --output="${directory}/output/output.txt"
# A file for STDERR from job.
#SBATCH --error="${directory}/output/error.txt"

##  RUNTIME RESOURCES.
# Note that if gpu's are requested, the call to gres is included in
# the appended 'sbatch_setup_commands'.
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --mem-per-cpu=4G

$sbatch_setup_commands

srun hostname
srun sleep 60
EOT
