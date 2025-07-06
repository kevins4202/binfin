from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F

from load_beats import load_classifier
from dataloader import get_dataloaders
from metrics import get_eval_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

train_loader, val_loader, test_loader = get_dataloaders()["train"], get_dataloaders()["val"], get_dataloaders()["test"]
print("Data loaders loaded")

model = load_classifier(freeze_beats=True).to(device)
model.train()
model = torch.compile(model)
print("Model loaded and compiled")
print("Total parameters: ", sum(p.numel() for p in model.parameters()))
print("Frozen parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad is False))
print("Trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad is True))

lr = 3e-4
steps = 20

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print("Optimizer loaded")

for step in range(steps):
    print(f"Step {step} of {steps}")
    for batch_idx, batch in enumerate(train_loader):
        print(f"Training batch {batch_idx} of {len(train_loader)}")
        model.train()
        audio, mask, label = batch
        audio = audio.to(device)
        mask = mask.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(audio, mask)
            loss = F.cross_entropy(logits, label)
        
        if step > 0 and step % 5 == 0:
            model.eval()
            with torch.no_grad():
                predictions = torch.tensor([], device=device)
                labels = torch.tensor([], device=device)
                loss_accum = 0
                for batch in val_loader:
                    audio, mask, labels = batch
                    audio = audio.to(device)
                    mask = mask.to(device)
                    labels = labels.to(device)

                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits = model(audio, mask)
                        loss = F.cross_entropy(logits, labels)
                        loss_accum += loss.item() / len(val_loader)
                        predictions = torch.cat([predictions, torch.sigmoid(logits).argmax(dim=1)])
                        labels = torch.cat([labels, labels])
                print(f"Step {step} val loss: {loss_accum / len(val_loader)}")
                print(f"Step {step} val accuracy: {accuracy_score(labels, predictions)}")
        
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        print(f"Step {step} loss: {loss.item()}")

print("Training complete")

model.eval()
with torch.no_grad():
    predictions = torch.tensor([], device=device)
    labels = torch.tensor([], device=device)
    loss_accum = 0
    for batch in test_loader:
        audio, mask, labels = batch
        audio = audio.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(audio, mask)
            loss = F.cross_entropy(logits, labels)
            loss_accum += loss.item() / len(test_loader)
            predictions = torch.cat([predictions, torch.sigmoid(logits).argmax(dim=1)])
            labels = torch.cat([labels, labels])
    print(f"Test loss: {loss_accum / len(test_loader)}")
    eer, auc, f1, acc, threshold, FAR, FRR = get_eval_metrics(predictions, labels)
    print(f"Test EER: {eer}")
    print(f"Test AUC: {auc}")
    print(f"Test F1: {f1}")
    print(f"Test accuracy: {acc}")
    print(f"Test threshold: {threshold}")
    print(f"Test FAR: {FAR}")





