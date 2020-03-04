import argparse
import os 
import json 
from config import get_config 
from transformers import OpenAIGPTTokenizer
from utils import PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, UNK_TOKEN, SEP_TOKEN, get_loader
import torch 
import solvers

def main():

    config = get_config(mode="test")

    if config.data_name == "cornell2":
        vocab = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        special_tokens = {
            'pad_token': PAD_TOKEN,
            'bos_token': SOS_TOKEN,
            'eos_token': EOS_TOKEN,
            'sep_token': SEP_TOKEN,
        }
        vocab.add_special_tokens(special_tokens)
        config.vocab_size = len(vocab)
        config.vocab = vocab
        config.pad_id = vocab.pad_token_id
        config.eos_id = vocab.eos_token_id 
        config.sos_id = vocab.bos_token_id 

        convs = [
            # [["u0", "how's the weather today in Daejeon?"], ["u1", "It's rainy... "], ["u0", "Did you take your umbrella?"], ["u1", "Sure I did"]],
            [["u0", "how's the weather today?"], ["u1", "Sure I did"]],
            [["u0", "did you have a nice weekends?"], ["u1", "sure"], ["u0", "where did you go?"]],
            # [["u0", "did you have a nice weekends?"], ["u1", "sure, It was wonderful :)"]],
            [["u0", "did you take your umbrella?"], ["u1", "sure, It was wonderful :)"]], 
            [["u0", "I hurt my legs"], ["u1", "oh,, i'm sorry to hear that"]],
            [["u200", "Do u love me?"], ["u1", "oh,, i'm sorry to hear that"]],
            [["u0", "I hurt my legs"], ["u1", "oh,, i'm sorry to hear that"], ["u0", "thanks"]],
            [["u0", "how's the weather today in Daejeon?"], ["u1", "Sure I did"]],
            # [["u0", "how's the weather today in Daejeon?"], ["u1", "It's sunny today!"], ["u0", "Did you take your umbrella?"], ["u1", "Sure I did"]],
            # [["u0", "hello"], ["u1", "i hate you"], ["u0", "what??"]],
            # [["u0", "hello"], ["u1", "i love you"], ["u0", "what??"]],
            [["u0", "hello"], ["u1", "i dont't have a girlfriend likes you"], ["u0", "i know"]]
        ]
    
    else: 
        raise ValueError("{} Sorry... We don't support that data".format(config.data_name))   

    models_path = os.path.join(config.dataset_dir, "model_infos.json")
    with open(models_path) as f: 
        models = json.load(f)["models"]

    project_dir = config.dataset_dir.parent.parent

    total_outputs = []
    model_names = []
    
    for model_i, model in enumerate(models):
        config.model = model["name"]
        config.checkpoint = os.path.join(project_dir, "results", config.data_name, model["name"], model["path"])
        model_names.append(model["name"] + "/" + model["path"])

        if model.get('config'):
            for key in model["config"]:
                setattr(config, key, model["config"][key])
        
        data_loader = get_loader(convs=convs,
                                vocab=vocab,
                                batch_size=1,
                                model=config.model,
                                dataset=config.data_name,
                                config=config,
                                shuffle=False)

        model_solver = getattr(solvers, "Solver{}".format(config.model))

        solver = model_solver(config, None, data_loader, vocab=vocab, is_train=False)

        solver.build()
        inputs, outputs = solver.export_samples(config.beam_size, file_write=False)

        for i, utter in enumerate(outputs):
            if model_i == 0: 
                total_outputs.append([utter])
            else:
                total_outputs[i].append(utter)

    result_path = os.path.join(project_dir, "results", config.data_name, "qualitative_samples.txt")

    with open(result_path, 'w') as fw:
        for input_utter, outputs in zip(inputs, total_outputs): 
            # print(input_utter, file=fw)
            # for i, output in enumerate(outputs):
            #     print("{} : {}".format(model_names[i], output), file=fw)
            # print('============================', file=fw)
            print(input_utter)
            for i, output in enumerate(outputs):
                print("{} : {}".format(model_names[i], output.split('<eos>')[0]))
            print('============================')


if __name__ == "__main__":
    main()