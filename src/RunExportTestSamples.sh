#!/usr/bin/env bash
# examples
# bash RunExportTestSamples.sh 0 cornell HRED 30 3 1 /data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/results/cornell2/HRED/20200217_112412/4.pkl
# bash RunExportTestSamples.sh 0 cornell VHRED 30 3 1 30.pkl
# bash RunExportTestSamples.sh 0 twitter_s ZHENG 1 /data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/results/twitter_s/ZHENG/20200225_153929/10.pkl

export CUDA_VISIBLE_DEVICES=$1

python export_test_responses.py --data="$2" --model="$3" --batch_size="$4" --pretrained_wv=False --users=True  --checkpoint="$5" --user_size=603 --beam_size="$6" --n_context=0

wait


#20200513_114436

# python export_test_responses.py --data=cornell2 --model=dialogpt --batch_size=1 --pretrained_wv=False --users=False  --checkpoint="$5" --user_size=603 --beam_size="$6" --n_context=0