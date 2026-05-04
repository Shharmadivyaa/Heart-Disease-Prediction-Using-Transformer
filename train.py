"""
Training loop for Heart Disease Transformer.
Includes early stopping, LR scheduling, and metric logging.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from model import HeartDiseaseTransformer
from data import prepare_data


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1  = f1_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, auc, f1, cm


def train(
    epochs: int = 60,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    lr: float = 3e-4,
    batch_size: int = 32,
    patience: int = 10,
    save_path: str = "heart_transformer.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_loader, val_loader, test_loader, scaler, _ = prepare_data(
        n_samples=1200, batch_size=batch_size
    )

    model = HeartDiseaseTransformer(
        num_features=13,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} {'Train Acc':>10} "
          f"{'Val Acc':>9} {'Val AUC':>9} {'LR':>10}")
    print("-" * 73)

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, v_auc, v_f1, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["val_auc"].append(v_auc)

        lr_now = scheduler.get_last_lr()[0]
        print(f"{epoch:>6} {t_loss:>11.4f} {v_loss:>10.4f} {t_acc:>10.4f} "
              f"{v_acc:>9.4f} {v_auc:>9.4f} {lr_now:>10.2e}")

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            patience_counter = 0
            torch.save({"model_state": model.state_dict(),
                        "config": dict(num_features=13, d_model=d_model, nhead=nhead,
                                       num_layers=num_layers, dim_feedforward=d_model * 4),
                        "history": history}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                break

    # Final evaluation on test set
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, test_auc, test_f1, cm = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"TEST RESULTS (best model)")
    print(f"{'='*50}")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  AUC-ROC  : {test_auc:.4f}")
    print(f"  F1 Score : {test_f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"{'='*50}\n")

    return model, history, scaler


if __name__ == "__main__":
    model, history, scaler = train(epochs=60)
    print("Training complete. Model saved to heart_transformer.pth")
