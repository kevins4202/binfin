import torch
from BEATs import BEATs, BEATsConfig
import os
from huggingface_hub import hf_hub_download
import torch.nn as nn


class DeepfakeClassifier(nn.Module):
    def __init__(self, freeze_beats):
        super().__init__()
        self.beats = load_beats()
        if freeze_beats:
            for param in self.beats.parameters():
                param.requires_grad = False
        self.hidden_dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, 2)  # binary classification
        )

    def forward(self, x, mask):
        features, _ = self.beats.extract_features(x, padding_mask=mask)
        
        pooled_features = features.mean(dim=1)
        
        logits = self.classifier(pooled_features)
        
        return logits


def load_beats():
    model_path = hf_hub_download(
        repo_id="kevins4202/beats",
        filename="beats3+.pt",
        token=os.environ.get("HF_TOKEN"),
    )

    # load the pre-trained checkpoints
    model_checkpoint = torch.load(model_path, weights_only=True)

    model_cfg = BEATsConfig(model_checkpoint["cfg"])
    
    BEATs_model = BEATs(model_cfg)
    BEATs_model.load_state_dict(model_checkpoint["model"])

    return BEATs_model

def load_classifier(freeze_beats=True):
    model = DeepfakeClassifier(freeze_beats)

    return model

if __name__ == "__main__":
    model = DeepfakeClassifier(freeze_beats=True)
    print(model)
