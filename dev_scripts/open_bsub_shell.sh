#!/bin/bash


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
# -R "res_req" [-R "res_req" ...]
# Runs the job on a host that meets the specified resource requirements.


#MIN_TASKS=8
#MIN_TASKS=16
MEM_LIMIT=16384

#bsub -Is -tty -n $MIN_TASKS -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
bsub -Is -tty -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" $SHELL
