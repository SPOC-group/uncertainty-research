#!/bin/bash

# List of hidden_width values to try
NUM_MODELS_VALUES=(2 5 10 15 20)

# Iterate over each hidden_width value and run the Python script
for NUM_MODELS in "${NUM_MODELS_VALUES[@]}"
do
    python ../main_experiment_ensembling.py --num_models $NUM_MODELS --n 20 --d 50 --hidden_width 100
done