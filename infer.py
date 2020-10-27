import data
import models
import soundfile as sf
import torch
import argparse
import time
import os
import pandas as pd
import numpy as np

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='Wave file to infer')
parser.add_argument('--dataset', type=str, help='Wave file to infer')
parser.add_argument('--model', type=str, help='Wave file to infer')
args = parser.parse_args()
file_path = args.file
dataset = args.dataset
model_path = args.model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config(os.path.join(model_path, "experiment.cfg"))
_,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load(os.path.join(model_path, 'training', 'model_state.pth'), map_location=device)) # load trained model

if file_path:
    signal, _ = sf.read(file_path)
    signal = torch.tensor(signal.astype(np.float64), device=device).float().unsqueeze(0)

    start = time.time()
    intent = model.decode_intents(signal)
    end = time.time()
    print(intent)
    print("Inference time:", end - start)


if dataset:
    csv_col = ['path', 'speakerId', 'transcription', 'action', 'number', 'object', 'location']
    accuracy = 0
    iter = 0
    frames = dict()
    files = []
    df = pd.read_csv(dataset)
    for row in zip(*[df[col].values.tolist() for col in csv_col]):
        frames[row[0]] = {csv_col[0]: row[0],
                          csv_col[1]: row[1],
                          csv_col[2]: row[2],
                          csv_col[3]: row[3],
                          csv_col[4]: row[4],
                          csv_col[5]: row[5],
                          csv_col[6]: row[6]}


        file_path = row[0]
        action = row[3]
        number = row[4]
        object = row[5]
        location = row[6]
        intent_true = [action, number, object, location]

        if not 'Time' in file_path and not 'Frequency' in file_path and not 'Impulse' in file_path  and not 'Gaussian' in file_path:
            signal, _ = sf.read(os.path.join('fp-dataset', file_path))
            signal = torch.tensor(signal.astype(np.float64), device=device).float().unsqueeze(0)

            intent = model.decode_intents(signal)
            if intent_true == intent[0]:
                accuracy += 1
            else:
                files.append(file_path) 

            iter +=1
        
    print("Accuracy", accuracy/iter)
    if files:
        files.sort()
        for fp in files:
            if not 'Time' in fp and not 'Frequency' in fp and not 'Impulse' in fp  and not 'Gaussian' in fp:
                print(fp)

