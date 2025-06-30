


import torch
from BEATs import BEATs, BEATsConfig

# load the pre-trained checkpoints
model_checkpoint = torch.load('model/beats_model.pt', weights_only=True)

model_cfg = BEATsConfig(model_checkpoint['cfg'])
BEATs_model = BEATs(model_cfg)
BEATs_model.load_state_dict(model_checkpoint['model'])
BEATs_model.eval()

# tokenize the audio and generate the labels
audio_input_16khz = torch.randn(1, 10000)
padding_mask = torch.zeros(1, 10000).bool()

representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

print(representation.shape)