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

#MEM_LIMIT=8192  # 8 GiBs
MEM_LIMIT=16384  # 16 GiBs
#MEM_LIMIT=32768  # 32 GiBs
#MEM_LIMIT=49152  # 48 GiBs
#MEM_LIMIT=65536  # 64 GiBs


# stardard compute node shell
################################################################################
if [[ "$JOB_TYPE" = "standard" ]]; then
    echo "starting standard shell"

    # specify compute node
    COMPUTE_NODE="any"
    #COMPUTE_NODE="hl-codon-02-04"

    if [[ "$COMPUTE_NODE" = "any" ]]; then
        bsub -q production -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT]" $SHELL
    else
        bsub -q production -m "$COMPUTE_NODE.ebi.ac.uk" -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT]" $SHELL
    fi
fi
################################################################################


# GPU node shell
################################################################################
# https://sysinf.ebi.ac.uk/doku.php?id=ebi_cluster_good_computing_guide#gpu_hosts
# https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-gpu

NUM_GPUS=1
#NUM_GPUS=2
#NUM_GPUS=4
#NUM_GPUS=6
#NUM_GPUS=8

#GPU_MEMORY=16384  # 16 GiBs
GPU_MEMORY=32256  # 31.5 GiBs
#GPU_MEMORY=32510  # ~32 GiBs, total Tesla V100 memory


# https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=jobs-submitting-that-require-gpu-resources
if [[ "$JOB_TYPE" = "gpu" ]]; then
    # specify gpu node
    GPU_NODE="any"
    #GPU_NODE=codon-gpu-001
    #GPU_NODE=codon-gpu-002

    if [[ "$GPU_NODE" = "any" ]]; then
        echo "starting gpu shell"
        bsub -q gpu -gpu "num=$NUM_GPUS:gmem=$GPU_MEMORY:j_exclusive=yes" -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
    else
        echo "starting gpu shell on $GPU_NODE"
        bsub -q gpu -m $GPU_NODE.ebi.ac.uk -gpu "num=$NUM_GPUS:gmem=$GPU_MEMORY:j_exclusive=yes" -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
    fi
fi
################################################################################


# parallel jobs shell
################################################################################
MIN_TASKS=4
#MIN_TASKS=8
#MIN_TASKS=16

if [[ "$JOB_TYPE" = "parallel" ]]; then
    echo "starting parallel shell with $MIN_TASKS tasks"
    bsub -q production -Is -tty -n $MIN_TASKS -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
fi
################################################################################
