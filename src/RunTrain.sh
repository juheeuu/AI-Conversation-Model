#!/usr/bin/env bash
# examples
# bash RunTrain.sh 0 cornell HRED 30 50
# bash RunTrain.sh 0 cornell VHRED 30 50

export CUDA_VISIBLE_DEVICES=$1

python train.py --data="$2" --model="$3" --batch_size="$4" --eval_batch_size="$5" --n_epoch="$6" --pretrained_wv=True --users=False --learning_rate="$7"

wait

