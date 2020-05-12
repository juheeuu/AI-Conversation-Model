#!/usr/bin/env bash
# examples
# bash RunTrain.sh 0 cornell HRED 30 50 1e-3
# bash RunTrain.sh 0 cornell VHRED 30 50
# bash RunTrain.sh 0,1 ubuntu HRED 30 50 1e-3
# bash RunTrain.sh 0,1,2,3 twitter_s ZHENG 20 20 5e-5
# bash RunTrain.sh 0,1 cornell2 DialogGPT 20 20 5e-5
# bash RunTrain.sh 0,1 bigbangtheory DialoGPT 20 20 1 5e-5

export CUDA_VISIBLE_DEVICES=$1

python train.py --data="$2" --model="$3" --batch_size="$4" --eval_batch_size="$5" --n_epoch="$6" --pretrained_wv=False --users=True --learning_rate="$7" --user_size=9036

wait

