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


# Submit an LSF job to run training and testing on a trained network.
# usage:
# bash submit_training.sh <experiment settings yaml file path>


# exit on any error
set -e


# editable variables
################################################################################
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
################################################################################


if [[ -z $1 ]]; then
    echo -e "usage:\n\tbash submit_training.sh <experiment settings yaml file path>"
    kill -INT $$
fi


# save the first argument as the experiment settings file path
experiment_settings="$1"


function parse_yaml() {
    python -c "import yaml; print(yaml.safe_load(open('$1'))['$2'])"
}


num_symbols=$(parse_yaml "$experiment_settings" "num_symbols")

# generate a local naive datetime string
datetime=$(date +%Y-%m-%d_%H:%M:%S)

job_name="n=${num_symbols}_${datetime}"

pipeline_command="python fully_connected_pipeline.py --datetime $datetime -ex $experiment_settings --train --test"


# submit job to a stardard compute node
if [[ "$job_type" = "standard" ]]; then
    bsub -M $mem_limit -R"select[mem>$mem_limit] rusage[mem=$mem_limit]" \
        -o "experiments/${job_name}.stdout.log" -e "experiments/${job_name}.stderr.log" \
        "$pipeline_command"
# submit job to a GPU node
elif [[ "$job_type" = "gpu" ]]; then
    bsub -P gpu -gpu "num=1:j_exclusive=yes" -m ${compute_node}.ebi.ac.uk \
        -M $mem_limit -R"select[mem>$mem_limit] rusage[mem=$mem_limit]" \
        -o "experiments/${job_name}.stdout.log" -e "experiments/${job_name}.stderr.log" \
        "$pipeline_command"
# submit a parallel job to a compute node
elif [[ "$job_type" = "parallel" ]]; then
    bsub -n $min_tasks -R"span[hosts=1]" \
        -M $mem_limit -R"select[mem>$mem_limit] rusage[mem=$mem_limit]" \
        -o "experiments/${job_name}.stdout.log" -e "experiments/${job_name}.stderr.log" \
        "$pipeline_command"
else
    echo "Error: specify a valid job type."
fi
