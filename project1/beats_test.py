


import torch
from BEATs import BEATs, BEATsConfig
import requests
import io
import os
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="kevins4202/beats",
    filename="beats_model.pt",
    token=os.environ.get("HF_TOKEN") 
)

# load the pre-trained checkpoints
model_checkpoint = torch.load(model_path, weights_only=True)

model_cfg = BEATsConfig(model_checkpoint['cfg'])
BEATs_model = BEATs(model_cfg)
BEATs_model.load_state_dict(model_checkpoint['model'])
BEATs_model.eval()

# tokenize the audio and generate the labels
audio_input_16khz = torch.randn(1, 10000)
padding_mask = torch.zeros(1, 10000).bool()

representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

print(representation.shape)