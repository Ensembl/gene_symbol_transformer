#!/bin/bash


MIN_TASKS=8
#MIN_TASKS=16
MEM_LIMIT=16384

PORT=54321
#PORT=58080

bsub -Is -tty -q production-rh74 -n $MIN_TASKS -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" jupyter lab --no-browser --port=$PORT
