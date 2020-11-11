#!/bin/bash


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


# https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.man_top.1.html


# settings used to install torch
#MEM_LIMIT=8192
#MIN_TASKS=8
#SCRATCH_SIZE=4096
#bsub -M $MEM_LIMIT -Is -n $MIN_TASKS -R"rusage[mem=$MEM_LIMIT, scratch=$SCRATCH_SIZE] select[model=XeonE52650, mem>$MEM_LIMIT] span[hosts=1]" $SHELL


# -Is [-tty]
# Submits an interactive job and creates a pseudo-terminal with shell mode when the job starts.
# -n min_tasks[,max_tasks]
# Submits a parallel job and specifies the number of tasks in the job.
# -M mem_limit [!]
# Sets a memory limit for all the processes that belong to the job.
# -q "queue_name ..."
# Submits the job to one of the specified queues.
# -R "res_req" [-R "res_req" ...]
# Runs the job on a host that meets the specified resource requirements.

#JOB_TYPE=standard
JOB_TYPE=gpu
#JOB_TYPE=parallel

#QUEUE=research-rh74
QUEUE=production-rh74

MEM_LIMIT=16384
#MEM_LIMIT=20000
#MEM_LIMIT=32768
#MEM_LIMIT=65536

#MIN_TASKS=8
MIN_TASKS=16


# open a shell on a stardard compute node
if [[ "$JOB_TYPE" = "standard" ]]; then
    bsub -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
fi

#COMPUTE_NODE=gpu-009
COMPUTE_NODE=gpu-011

# open a shell on a GPU node
# https://sysinf.ebi.ac.uk/doku.php?id=ebi_cluster_good_computing_guide#gpu_hosts
# https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.gpu.1.html
if [[ "$JOB_TYPE" = "gpu" ]]; then
    #bsub -q $QUEUE -P gpu -gpu - -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
    # specify exclusive use of the GPU
    bsub -q $QUEUE -P gpu -gpu "num=1:j_exclusive=yes" -m ${COMPUTE_NODE}.ebi.ac.uk -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
fi

# open a parallel jobs shell
if [[ "$JOB_TYPE" = "parallel" ]]; then
    bsub -Is -tty -n $MIN_TASKS -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
fi
