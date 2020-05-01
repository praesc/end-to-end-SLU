import data
import models
import soundfile as sf
import torch
import argparse
import time
import os

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='Wave file to infer')
parser.add_argument('--model', type=str, help='Wave file to infer')
args = parser.parse_args()
file_path = args.file
model_path = args.model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config(os.path.join(model_path, "experiment.cfg"))
_,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load(os.path.join(model_path, 'training', 'model_state.pth'), map_location=device)) # load trained model

signal, _ = sf.read(file_path)
signal = torch.tensor(signal, device=device).float().unsqueeze(0)

start = time.time()
intent = model.decode_intents(signal)
end = time.time()
print(intent)
print("Inference time:", end - start)
