#!/bin/bash


if [[ -z "$1" ]]; then
    echo "Nothing to run, pass the command to run with bsub as an argument."
    exit
fi


DATE_TIME=$(date +%Y-%m-%d_%H:%M:%S%:z)

JOB_NAME="$DATE_TIME"


#QUEUE=research-rh74
QUEUE=production-rh74

GPU_NODE="V100"
#GPU_NODE="any"

MEM_LIMIT=16384
#MEM_LIMIT=32768
#MEM_LIMIT=65536

if [[ "$GPU_NODE" = "any" ]]; then
    bsub -q $QUEUE -P gpu -gpu "num=1:j_exclusive=yes" -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" -o "${JOB_NAME}.stdout.txt" -e "${JOB_NAME}.stderr.txt" "$@"
fi

# specify compute node
#COMPUTE_NODE=gpu-009
COMPUTE_NODE=gpu-011

if [[ "$GPU_NODE" = "V100" ]]; then
    bsub -q $QUEUE -P gpu -gpu "num=1:j_exclusive=yes" -m ${COMPUTE_NODE}.ebi.ac.uk -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" -o "${JOB_NAME}.stdout.txt" -e "${JOB_NAME}.stderr.txt" "$@"
fi
