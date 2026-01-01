from src.dataset import ATPMatchesDataset
from src.differnce_model import DifferenceModelBN
from src.baseline_model import BaseLineModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle

def get_feature_block_sizes(dataset):
    ct = dataset.pipeline.named_steps['column_transformer']
    winner_size = (
        len(dataset.winner_numerical)
    )
    loser_size = (
        len(dataset.loser_numerical)
    )
    match_size = len(ct.named_transformers_['match_cat'].get_feature_names_out())
    total = winner_size + loser_size + match_size
    return winner_size, loser_size, match_size, total



def train_model(
    model,
    train_loader,
    dev_loader,
    device,
    epochs=50,
    save_path="model.pkl",
    test_loader= None
):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    acc = 0
    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        for batch in train_loader:
            x = batch['features'].to(device)
            y = batch['target'].float().to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        # ---- DEV EVAL ----
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for batch in dev_loader:
                x = batch['features'].to(device)
                y = batch['target'].to(device)

                preds = (torch.sigmoid(model(x)) > 0.5).long()
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d} / {epochs} | "
              f"Loss {loss_sum/len(train_loader):.4f} | "
              f"Dev Acc {acc:.2f}%")

        scheduler.step()
    if test_loader:
        evaluate_on_test(model, test_loader, device)

    to_save = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    with open(save_path, "wb") as f:
        pickle.dump(to_save, f)

def evaluate_on_test(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(device)
            y = batch['target'].to(device)

            preds = (torch.sigmoid(model(x)) > 0.5).long()
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"\nTEST Accuracy: {acc:.2f}%")
    return acc


if __name__ == "__main__":

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)


    dataset = ATPMatchesDataset("data/processed/atp_matches_2000_2024_final.csv")
    train_loader = DataLoader(dataset.TRAIN, batch_size=32, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dataset.DEV, batch_size=32, shuffle=False, num_workers=2)


    winner_size, loser_size, match_size, input_size = get_feature_block_sizes(dataset)

    assert winner_size == loser_size
    assert winner_size * 2 + match_size == input_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = DataLoader(dataset.TEST, batch_size=32, shuffle=False, num_workers=2)

    models = {
        "DifferenceModelBN": DifferenceModelBN(winner_size, match_size),
        "BaseLineModel": BaseLineModel(input_size),
    }

    for name, model in models.items():
        print(f"\nTraining {name}")
        model.to(device)

        save_path = f"{name}.pkl"
        train_model(
            model,
            train_loader,
            dev_loader,
            device,
            epochs=15,
            save_path=save_path,
            test_loader=test_loader
        )
