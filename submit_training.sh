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


datetime=$(date +%Y-%m-%d_%H:%M:%S)

pipeline_command="python fully_connected_pipeline.py --datetime $datetime -ex experiment_settings.yaml --train --test"


job_type=standard
#job_type=gpu
#job_type=parallel

compute_node=gpu-009
#compute_node=gpu-011

min_tasks=8
#min_tasks=16

mem_limit=16384
#mem_limit=32768
#mem_limit=65536

job_name="$datetime"


# stardard compute node shell
if [[ "$job_type" = "standard" ]]; then
    bsub -M $mem_limit -R"select[mem>$mem_limit] rusage[mem=$mem_limit]" \
        -o "experiments/${job_name}.stdout.log" -e "experiments/${job_name}.stderr.log" \
        "$pipeline_command"
elif [[ "$job_type" = "gpu" ]]; then
    bsub -P gpu -gpu "num=1:j_exclusive=yes" -m ${compute_node}.ebi.ac.uk \
        -M $mem_limit -R"select[mem>$mem_limit] rusage[mem=$mem_limit]" \
        -o "experiments/${job_name}.stdout.log" -e "experiments/${job_name}.stderr.log" \
        "$pipeline_command"
elif [[ "$job_type" = "parallel" ]]; then
    bsub -n $min_tasks -R"span[hosts=1]" \
        -M $mem_limit -R"select[mem>$mem_limit] rusage[mem=$mem_limit]" \
        -o "experiments/${job_name}.stdout.log" -e "experiments/${job_name}.stderr.log" \
        "$pipeline_command"
else
    echo "Error: specify a valid job type."
fi
