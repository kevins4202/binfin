import torch
from BEATs import BEATs, BEATsConfig
import os
from huggingface_hub import hf_hub_download
import torch.nn as nn


class DeepfakeClassifier(nn.Module):
    def __init__(self, freeze_beats, load_base, load_head):
        super().__init__()
        self.beats = load_beats(load_base)
        if freeze_beats:
            for param in self.beats.parameters():
                param.requires_grad = False
        self.head = load_finetuned_head(load_head)

    def forward(self, x, mask):
        features, _ = self.beats.extract_features(x, padding_mask=mask)

        pooled_features = features.mean(dim=1)

        logits = self.head(pooled_features)

        return logits


def load_beats(load_base):
    BEATs_model = None
    
    if load_base:
        model_path = hf_hub_download(
            repo_id="kevins4202/beats",
            filename="beats3+.pt",
            token=os.environ.get("HF_TOKEN"),
        )

        # load the pre-trained checkpoints
        model_checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
        model_cfg = BEATsConfig(model_checkpoint["cfg"])
        BEATs_model = BEATs(model_cfg)
        BEATs_model.load_state_dict(model_checkpoint["model"])
    else:
        BEATs_model = BEATs(BEATsConfig())

    return BEATs_model


def load_finetuned_head(load_head):
    head = nn.Sequential(
        nn.Linear(768, 768 // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(768 // 2, 2)
    )

    if load_head:
        head_path = hf_hub_download(
            repo_id="kevins4202/beats_finetuned",
            filename="beats_finetuned_classifier.pt",
            token=os.environ.get("HF_TOKEN"),
        )
        head_checkpoint = torch.load(head_path, weights_only=True, map_location="cpu")

        head.load_state_dict(head_checkpoint)

    return head


def load_classifier(freeze_beats=True, load_head=False):
    model = DeepfakeClassifier(freeze_beats, load_head)

    return model


if __name__ == "__main__":
    model = DeepfakeClassifier(freeze_beats=True)
    print(model)
