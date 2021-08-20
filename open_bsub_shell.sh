#!/usr/bin/env bash


# See the NOTICE file distributed with this work for additional information
# regarding copyright ownership.
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


# exit on any error
set -e


JOB_TYPE=standard
#JOB_TYPE=gpu
#JOB_TYPE=parallel

MEM_LIMIT=8192
#MEM_LIMIT=16384
#MEM_LIMIT=32768


# stardard compute node shell
################################################################################
if [[ "$JOB_TYPE" = "standard" ]]; then
    echo "starting standard shell"

    # specify compute node
    COMPUTE_NODE="any"
    #COMPUTE_NODE="hx-noah-13-01"

    if [[ "$COMPUTE_NODE" = "any" ]]; then
        bsub -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT]" $SHELL
    else
        bsub -m ${COMPUTE_NODE}.ebi.ac.uk -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT]" $SHELL
    fi
fi
################################################################################


# GPU node shell
################################################################################
# https://sysinf.ebi.ac.uk/doku.php?id=ebi_cluster_good_computing_guide#gpu_hosts
# https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-gpu

COMPUTE_NODE=gpu-009
#COMPUTE_NODE=gpu-011

NUM_GPUS=1
#NUM_GPUS=2
#NUM_GPUS=4
#NUM_GPUS=6
#NUM_GPUS=8

if [[ "$JOB_TYPE" = "gpu" ]]; then
    echo "starting gpu shell"
    bsub -P gpu -gpu "num=$NUM_GPUS:j_exclusive=yes" -m ${COMPUTE_NODE}.ebi.ac.uk -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
fi
################################################################################


# parallel jobs shell
################################################################################
MIN_TASKS=8
#MIN_TASKS=16

if [[ "$JOB_TYPE" = "parallel" ]]; then
    echo "starting parallel shell"
    bsub -Is -tty -n $MIN_TASKS -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
    #bsub -Is -tty -n $MIN_TASKS -M $MEM_LIMIT -R"select[model=XeonE52650, mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
fi
################################################################################
