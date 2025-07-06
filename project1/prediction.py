import torch
from dataloader import  get_dataloaders
from metrics import get_eval_metrics
from load_beats import load_classifier

dataloaders = get_dataloaders()

def predict(model, dataloader, threshold=0.5):
    labels = torch.tensor([])
    predictions = torch.tensor([])
    # find the accuracy, precision, recall, f1 score, EER, and AUC
    model.eval()
    with torch.no_grad():
        for inputs, masks, labels_batch in dataloader:
            outputs = model(inputs, masks)

            predictions_batch = torch.sigmoid(outputs.squeeze())
            labels = torch.cat((labels, labels_batch))
            predictions = torch.cat((predictions, predictions_batch))

        print("predictions", predictions)
        print("labels", labels)

        predictions = (predictions > threshold).float()

        return get_eval_metrics(predictions, labels)

if __name__ == "__main__":
    # model = load_beats()
    # print("model loaded, # params", sum(p.numel() for p in model.parameters()))
    # batch, masks, labels = next(iter(dataloaders["test"]))
    # outputs, _ = model.extract_features(batch, masks)
    # print(outputs.shape)
    # print("outputs range", outputs.min(), outputs.max())
    model = load_classifier()

    print("model loaded, # params", sum(p.numel() for p in model.parameters()))

    eer, auc, f1, acc, threshold, FAR, FRR = predict(model, dataloaders["test"])

    print("eer", eer)
    print("auc", auc)
    print("f1", f1)
    print("acc", acc)
    print("threshold", threshold)
    print("FAR", FAR)
    print("FRR", FRR)