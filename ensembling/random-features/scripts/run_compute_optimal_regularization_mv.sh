#!/bin/bash

HIDDEN_WIDTH_VALUES=(100 250 500 1000 1250 1500 2000)
FREEZE_FIRST_LAYER="True"
N=100
D=50

for HIDDEN_WIDTH in "${HIDDEN_WIDTH_VALUES[@]}"
    do
        python optimize_weight_decay.py --n $N --d $D --hidden_width $HIDDEN_WIDTH --lr 0.005 --n_trials 20 --model_class "MeanVarianceModel" --save_file "MeanVarianceModel_freeze_first_layer=($FREEZE_FIRST_LAYER)_n=($N)_d=($D)_hidden_width=$HIDDEN_WIDTH.json"
    done