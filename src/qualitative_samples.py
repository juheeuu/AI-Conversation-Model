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

        convs = [
            [["u44", "What's wrong with that?"], ["u29", "We don't have her I.D. yet, but one of your girls was killed last night at the King Edward Hotel."], ["u44", "What's wrong with that?"]],
            [["u0", "hello"], ["u1", "i love you"], ["u0", "what??"]],
            [["u0", "hello"], ["u1", "i hate you"], ["u0", "what??"]],
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
        inputs, outputs = solver.export_samples(file_write=False)

        for i, utter in enumerate(outputs):
            if model_i == 0: 
                total_outputs.append([utter])
            else:
                total_outputs[i].append(utter)

    result_path = os.path.join(project_dir, "results", config.data_name, "qualitative_samples.txt")

    with open(result_path, 'w') as fw:
        for input_utter, outputs in zip(inputs, total_outputs): 
            print(input_utter, file=fw)
            for i, output in enumerate(outputs):
                print("{} : {}".format(model_names[i], output), file=fw)
            print('============================', file=fw)

if __name__ == "__main__":
    main()