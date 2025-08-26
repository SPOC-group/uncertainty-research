#!/bin/bash

# with optimal regularization
python ../main_experiment_ensembling.py --num_models 20 --n 100 --d 20 --hidden_width 50 --lr 0.01 --pretraining_weight_decay 0.0037