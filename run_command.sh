#!/usr/bin/env bash


DATETIME=$(date +%Y-%m-%d_%H:%M)

#NUM_MOST_FREQUENT_SYMBOLS=3
NUM_MOST_FREQUENT_SYMBOLS=101
#NUM_MOST_FREQUENT_SYMBOLS=1013
#NUM_MOST_FREQUENT_SYMBOLS=10059
#NUM_MOST_FREQUENT_SYMBOLS=20147
#NUM_MOST_FREQUENT_SYMBOLS=25028
#NUM_MOST_FREQUENT_SYMBOLS=30591

RANDOM_STATE=5

PYTHON_SCIPT="transformer_pipeline.py"

# train directly on a GPU node, disable buffering in Python script output, save output to a file with tee
python -u "$PYTHON_SCIPT" --datetime "$DATETIME" --random_state $RANDOM_STATE --num_most_frequent_symbols $NUM_MOST_FREQUENT_SYMBOLS --train --test | tee -a "networks/n=${NUM_MOST_FREQUENT_SYMBOLS}_${DATETIME}.log"
