# How to run the code
## dataset structure
- directory structure 
```
{data_root}/train/convs.pkl
           /valid/convs.pkl
           /test/convs.pkl
```
- In convs.pkl, the dataset consists with list of conversations.

```[[(u0, hi!), (u1,hi!)], [...]]```

## Train

```
python train.py --data=persona_chat \
--model=DialoGPT \
--batch_size=10 \
--eval_batch_size=10 \
--n_epoch=100 \
--pretrained_wv=False \
--learning_rate=5e-5 
```

- To train reverse version, add this 
    -  `--reversed=True --pretrained_path=small_reverse.pkl`
- To train dialogpt + user version, add this
    - `--users=True --user_size={your user size}`
- To train dialogpt + user reversed version, add this
    - `--users=True --user_size={your user size} --reversed=True --pretrained_path=small_reverse.pkl` 




## Export example 
```
export_test_responses.py --data=peronachat\
--model=DialoGPT \
--batch_size=1 --pretrained_wv=False \
--checkpoint={your checkpoint path} --beam_size=1 \
--n_context=1 --user_size={size of users of your dataset}
```
- you can manage the number of context with n_context 
- to export dialogpt + mmi add this 
    - `--mmi=True reversed_pretrained_path={your pretrained path}`
- to export dialogpt + user version, add this 
    - `--users=True`
- to export dialogpt + user MMI version, add this 
    - `--users=True --mmi=True reversed_pretrained_path={your pretrained path}`



## Evaluation 
```
bash RunEval.sh {output file} {dataset type} {forward model path}
```
