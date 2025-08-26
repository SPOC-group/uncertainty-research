#!/bin/bash

# List of hidden_width values to try
HIDDEN_WIDTH_VALUES=(20 50 100 150 200 250 500)

# Iterate over each hidden_width value and run the Python script
for HIDDEN_WIDTH in "${HIDDEN_WIDTH_VALUES[@]}"
do
    python ../main_experiment_ensembling.py --num_models 20 --n 100 --d 50 --hidden_width $HIDDEN_WIDTH
donezoter
