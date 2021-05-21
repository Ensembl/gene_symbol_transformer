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


# Submit an LSF job to run testing on a trained network.
# usage:
# bash submit_testing.sh <checkpoint file path>


# exit on any error
set -e


# editable variables
################################################################################
mem_limit=8192
#mem_limit=16384
#mem_limit=32768
################################################################################


if [[ -z $1 ]]; then
    echo -e "usage:\n\tbash submit_testing.sh <checkpoint file path>"
    kill -INT $$
fi


# save the first argument as the checkpoint file path
checkpoint_path="$1"


checkpoint_basename="$(basename $checkpoint_path)"
checkpoint_filename="${checkpoint_basename%.*}"
job_name="$checkpoint_filename"


pipeline_command="python gene_symbol_classifier.py --checkpoint $checkpoint_path --test"


# submit job
bsub -M $mem_limit -R"select[mem>$mem_limit] rusage[mem=$mem_limit]" \
    -o "experiments/${job_name}.stdout.log" -e "experiments/${job_name}.stderr.log" \
    "$pipeline_command"
