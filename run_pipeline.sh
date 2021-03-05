#!/usr/bin/env bash


#target_script="fully_connected_pipeline.py"
#target_script="lstm_pipeline.py"
target_script="$1"

# validate training pipeline script name
pipeline_scripts=(
    "fully_connected_pipeline.py"
    "lstm_pipeline.py"
)

valid_script=false
for pipeline_script in "${pipeline_scripts[@]}"; do
    if [[ "$target_script" = "$pipeline_script" ]]; then
        valid_script=true
    fi
done

if [[ $valid_script != true ]]; then
    echo "pass one of the pipeline scripts as an argument:"
    echo "{"
    for pipeline_script in "${pipeline_scripts[@]}"; do
        echo "    \"$pipeline_script\","
    done
    echo "}"
    exit
fi

datetime=$(date +%Y-%m-%d_%H:%M:%S)

random_state=5
#random_state=7
#random_state=11

num_most_frequent_symbols=3
#num_most_frequent_symbols=101
#num_most_frequent_symbols=1013
#num_most_frequent_symbols=10059
#num_most_frequent_symbols=20147
#num_most_frequent_symbols=25028
#num_most_frequent_symbols=26007
#num_most_frequent_symbols=27137
#num_most_frequent_symbols=28197
#num_most_frequent_symbols=29041
#num_most_frequent_symbols=30591

#sequence_length=750
sequence_length=1000
#sequence_length=1500
#sequence_length=2000

#batch_size=32
#batch_size=64
#batch_size=128
#batch_size=256
#batch_size=512
#batch_size=1024
batch_size=2048
#batch_size=4096
#batch_size=8192

#learning_rate=0.01
learning_rate=0.001

num_epochs=1
#num_epochs=3
#num_epochs=10
#num_epochs=100
#num_epochs=1000

# train directly on a compute node
python "$target_script" \
    --train \
    --test \
    --datetime "$datetime" \
    --random_state $random_state \
    --num_most_frequent_symbols $num_most_frequent_symbols \
    --sequence_length $sequence_length \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs
