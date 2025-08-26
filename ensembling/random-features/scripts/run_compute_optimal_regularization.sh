#!/bin/bash

HIDDEN_WIDTH_VALUES=(200 500 1000 1500)

# define freeze first layer, n and d as below
FREEZE_FIRST_LAYER="True"
N=1000
D=100

for HIDDEN_WIDTH in "${HIDDEN_WIDTH_VALUES[@]}"
    do
        python optimize_regularization.py --n $N --d $D --hidden_width $HIDDEN_WIDTH --lr 0.01 --n_trials 20 --model_class "RandomFeatures" --freeze_first_layer $FREEZE_FIRST_LAYER --save_file "RandomFeatures_freeze_first_layer=($FREEZE_FIRST_LAYER)_n=($N)_d=($D)_hidden_width=$HIDDEN_WIDTH.json" 
    done