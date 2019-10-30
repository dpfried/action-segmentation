#!/bin/bash

output_name=$1
shift
output_path="expts/crosstask/${output_name}"

mkdir $output_path

export PYTHONPATH="src/":$PYTHONPATH

python -u src/main.py \
    --dataset crosstask \
    --model_output_path $output_path \
    $@ \
    | tee ${output_path}/log.txt
