#!/bin/bash


DATE_TIME=$(date +%Y-%m-%d_%H:%M:%S%:z)

JOB_NAME="$DATE_TIME"


#QUEUE=research-rh74
QUEUE=production-rh74

#MEM_LIMIT=16384
#MEM_LIMIT=20000
#MEM_LIMIT=32768
MEM_LIMIT=65536

#bsub -q $QUEUE -P gpu -gpu "num=1:j_exclusive=yes" -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" -o "${JOB_NAME}.stdout.txt" -e "${JOB_NAME}.stderr.txt" "$@"

# specify compute node
COMPUTE_NODE=gpu-009
#COMPUTE_NODE=gpu-010
#COMPUTE_NODE=gpu-011

bsub -q $QUEUE -P gpu -gpu "num=1:j_exclusive=yes" -m ${COMPUTE_NODE}.ebi.ac.uk -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" -o "${JOB_NAME}.stdout.txt" -e "${JOB_NAME}.stderr.txt" "$@"
