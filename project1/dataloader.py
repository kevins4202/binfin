import os
import random
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchaudio
from typing import List, Tuple
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_and_extract_audio_data():
    """Download and extract audio data from Hugging Face dataset"""
    # Create data directory if it doesn't exist
    data_dir = Path("data/ADD")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    if (data_dir / "data").exists() and (data_dir / "meta.txt").exists():
        print("Audio data already exists, skipping download...")
        return str(data_dir / "data"), str(data_dir / "meta.txt")
    
    print("Downloading audio data from Hugging Face...")
    
    # Download the zip file using hf_hub_download
    zip_path = hf_hub_download(
        repo_id="kevins4202/binfin-project",
        filename="data.zip",
        repo_type="dataset"
    )
    
    # Extract the zip file
    print("Extracting audio data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/ADD")
    
    # remove the __MACOSX folder
    shutil.rmtree("data/ADD/__MACOSX")
    
    print("Audio data downloaded and extracted successfully!")
    return str(data_dir / "data"), str(data_dir / "meta.txt")

class DeepfakeDataset(Dataset):
    def __init__(self, examples, audio_dir):
        self.dataset = examples
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Customize this based on your dataset's structure
        item = self.dataset[idx]
        audio_path = os.path.join(self.audio_dir, item["filename"])
        audio, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)

        return audio, item["label"]

def collate_fn_audio(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads audio tensors in a batch to the same length.
    `batch` is a list of tuples (waveform, label).
    """
    # Separate waveforms and labels
    tensors, targets = zip(*batch)
    # Batch, Channels, Samples

    # Audio tensors are already in (channels, samples) format
    # We need to pad them to the same length
    # Get the maximum length in the batch
    max_length = max(tensor.shape[1] for tensor in tensors)
    
    # Pad each tensor to the max length
    padded_tensors = []
    masks = []
    for tensor in tensors:
        # tensor is (channels, samples)
        channels, samples = tensor.shape
        mask = torch.zeros(channels, max_length).bool()
        if samples < max_length:
            # Pad with zeros
            padding = torch.zeros(channels, max_length - samples)
            tensor = torch.cat([tensor, padding], dim=1)
            mask[:, samples:] = True
        padded_tensors.append(tensor)
        masks.append(mask)
    
    # Stack tensors to create (batch, channels, samples)
    # 4, 1, max_length
    tensors_padded = torch.stack(padded_tensors, dim=0)
    masks = torch.stack(masks, dim=0)

    tensors_padded = tensors_padded.squeeze(1)
    masks = masks.squeeze(1)

    targets = torch.tensor([int(t) for t in list(targets)], dtype=torch.long)

    return tensors_padded, masks, targets

# dataset = load_dataset("kevins4202/binfin-project")
audio_dir, meta_file = download_and_extract_audio_data()

# Load metadata
# with open("data/ADD/meta.txt", "r") as f:
with open(meta_file, "r") as f:
    lines = f.readlines()
    meta_csv = [line.strip().split("\t") for line in lines]
    examples = [
        {
            "filename": row[0],
            "label": row[1],
        }
        for row in meta_csv
    ]

random.seed(42)
random.shuffle(examples)
n = len(examples)
train_end = int(n * 0.9)
val_end = int(n * 0.95)

train_examples = examples[:train_end]
val_examples = examples[train_end:val_end]
test_examples = examples[val_end:]

# audio_dir = "data/ADD/data"

train_dataset = DeepfakeDataset(train_examples, audio_dir)
val_dataset = DeepfakeDataset(val_examples, audio_dir)
test_dataset = DeepfakeDataset(test_examples, audio_dir)

BATCH_SIZE = 4

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_audio
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_audio
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_audio
)

def get_dataloaders():
    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()["train"], get_dataloaders()["val"], get_dataloaders()["test"]
    print("Train dataset size:", len(train_loader.dataset))
    print("Val dataset size:", len(val_loader.dataset))
    print("Test dataset size:", len(test_loader.dataset))
    