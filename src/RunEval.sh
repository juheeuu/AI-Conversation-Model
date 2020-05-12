#!/usr/bin/env bash
# examples
# bash RunEval.sh responses_test_3_1_5_28.txt
# bash RunEval.sh /data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/results/twitter_s/HRED/20200226_113051/responses_test_1_1_4_4.txt twitter_s /data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/results/twitter_s/HRED/20200226_113051/4.pkl

python eval.py "$1" "$2" "$3"

wait

