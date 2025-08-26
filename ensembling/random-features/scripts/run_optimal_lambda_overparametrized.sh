#!/bin/bash

# we have optimal regularization here
python ../main_experiment_ensembling.py --num_models 20 --n 100 --d 200 --hidden_width 500 --pretraining_weight_decay 0.025 --lr 0.1 --num_epochs 10000