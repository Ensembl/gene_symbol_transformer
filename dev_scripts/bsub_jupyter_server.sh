#!/bin/bash


MEM_LIMIT=16384

PORT=54321

bsub -Is -tty -q production-rh74 -M $MEM_LIMIT -R"select[mem>$MEM_LIMIT] rusage[mem=$MEM_LIMIT] span[hosts=1]" jupyter lab --no-browser --port=$PORT
