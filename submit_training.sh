#!/bin/bash


#NUM_MOST_FREQUENT_SYMBOLS=3
#NUM_MOST_FREQUENT_SYMBOLS=101
#NUM_MOST_FREQUENT_SYMBOLS=1013
#NUM_MOST_FREQUENT_SYMBOLS=10059
#NUM_MOST_FREQUENT_SYMBOLS=20147
NUM_MOST_FREQUENT_SYMBOLS=25028
#NUM_MOST_FREQUENT_SYMBOLS=30591


#NUM_MOST_FREQUENT_SYMBOLS=25028; RANDOM_STATE=5; bash submit_training.sh python sequence_pipeline.py --random_state $RANDOM_STATE --num_most_frequent_symbols $NUM_MOST_FREQUENT_SYMBOLS --train --test"


if [[ -z "$1" ]]; then
    echo "Nothing to run, pass the command to run with bsub as an argument."
    exit
fi


DATE_TIME=$(date +%Y-%m-%d_%H:%M:%S%:z)

JOB_NAME="n=${NUM_MOST_FREQUENT_SYMBOLS}_${DATE_TIME}"


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
