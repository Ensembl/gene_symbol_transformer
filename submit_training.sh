#!/usr/bin/env bash


# Copyright 2020 EMBL-European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#NUM_MOST_FREQUENT_SYMBOLS=3
#NUM_MOST_FREQUENT_SYMBOLS=101
#NUM_MOST_FREQUENT_SYMBOLS=1013
#NUM_MOST_FREQUENT_SYMBOLS=10059
#NUM_MOST_FREQUENT_SYMBOLS=20147
NUM_MOST_FREQUENT_SYMBOLS=25028
#NUM_MOST_FREQUENT_SYMBOLS=30591

DATETIME="2020-11-11T22:51"

#DATETIME="2020-11-11T22:49"; NUM_MOST_FREQUENT_SYMBOLS=25028; RANDOM_STATE=5; bash submit_training.sh python sequence_pipeline.py --random_state $RANDOM_STATE --num_most_frequent_symbols $NUM_MOST_FREQUENT_SYMBOLS --train --test"


if [[ -z "$1" ]]; then
    echo "Nothing to run, pass the command to run with bsub as an argument."
    exit
fi


JOB_NAME="n=${NUM_MOST_FREQUENT_SYMBOLS}_${DATETIME}"


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
