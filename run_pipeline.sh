#!/usr/bin/env bash


#TARGET_SCRIPT="lstm_pipeline.py"
#TARGET_SCRIPT="cnn_pipeline.py"
#TARGET_SCRIPT="fully_connected_pipeline.py"
TARGET_SCRIPT="$1"


PIPELINE_SCRIPTS=(
    "lstm_pipeline.py"
    "cnn_pipeline.py"
    "fully_connected_pipeline.py"
)

VALID_SCRIPT=false
for PIPELINE_SCRIPT in "${PIPELINE_SCRIPTS[@]}"; do
    if [[ "$TARGET_SCRIPT" = "$PIPELINE_SCRIPT" ]]; then
        VALID_SCRIPT=true
    fi
done

if [[ $VALID_SCRIPT != true ]]; then
    echo "pass one of the pipeline scripts as an argument:"
    echo "{"
    for PIPELINE_SCRIPT in "${PIPELINE_SCRIPTS[@]}"; do
        echo "    \"$PIPELINE_SCRIPT\","
    done
    echo "}"
    exit
fi

DATETIME=$(date +%Y-%m-%d_%H:%M:%S)

#NUM_MOST_FREQUENT_SYMBOLS=3
#NUM_MOST_FREQUENT_SYMBOLS=101
#NUM_MOST_FREQUENT_SYMBOLS=1013
#NUM_MOST_FREQUENT_SYMBOLS=10059
#NUM_MOST_FREQUENT_SYMBOLS=20147
NUM_MOST_FREQUENT_SYMBOLS=25028
#NUM_MOST_FREQUENT_SYMBOLS=30591

RANDOM_STATE=5

# train directly on a GPU node, disable buffering in Python script output, save output to a file with tee
python -u "$TARGET_SCRIPT" --datetime "$DATETIME" --random_state $RANDOM_STATE --num_most_frequent_symbols $NUM_MOST_FREQUENT_SYMBOLS --train --test | tee -a "networks/n=${NUM_MOST_FREQUENT_SYMBOLS}_${DATETIME}.log"
