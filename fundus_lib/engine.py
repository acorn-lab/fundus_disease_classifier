# engine.py
# shared training/evaluation loop used by every model notebook. mixed precision,
# class-weighted cross-entropy (for imbalance), early stopping on validation
# weighted-F1. models are expected to return raw logits of shape (B, K); the
# model wrappers in models.py guarantee that interface.

import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score


def class_weights_from_labels(labels, n_classes, device):
    # inverse-frequency weights to counter class imbalance
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    # returns (mean train loss, train accuracy) over the epoch
    model.train()
    running_loss, n_correct, n_total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        n_correct += (logits.detach().float().argmax(1) == labels).sum().item()
        n_total += images.size(0)
    return running_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    # returns y_true (N,) and y_prob (N, K) softmax probabilities. if criterion
    # is supplied, also returns the mean loss over the loader so the caller can
    # track validation loss without a second pass.
    model.eval()
    all_true, all_prob = [], []
    loss_sum, n_total = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if criterion is not None:
            loss = criterion(logits.float(), labels.to(device))
            loss_sum += loss.item() * images.size(0)
            n_total += images.size(0)
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_prob.append(prob)
        all_true.append(labels.numpy())
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    if criterion is not None:
        return y_true, y_prob, loss_sum / max(n_total, 1)
    return y_true, y_prob


def fit(model, train_loader, val_loader, device, n_classes,
        epochs=40, lr=3e-4, weight_decay=1e-4, use_amp=True,
        class_weights=None, patience=20, verbose=True):
    # default patience is intentionally generous: with ~110-sample val sets and
    # 10 classes the val_weighted_f1 curve is noisy from one epoch to the next.
    # patience=8 was killing folds that hit a transient plateau; raising to 20
    # let folds 0/2/4 actually train to convergence in our experiments.
    # standard fine-tuning loop with early stopping on validation weighted-F1.
    # returns the best model (loaded with best weights) and the training history.
    # history rows: {epoch, train_loss, val_loss, train_acc, val_acc,
    #                val_weighted_f1, lr}
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None

    best_f1, best_state, best_epoch = -1.0, None, -1
    history = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        scheduler.step()
        y_true, y_prob, val_loss = evaluate(model, val_loader, device, criterion=criterion)
        y_pred = y_prob.argmax(1)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_weighted_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })
        if verbose:
            print(f"  epoch {epoch:2d} | train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
                  f"| train_acc {train_acc:.3f} val_acc {val_acc:.3f} | val_wF1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch}, wF1 {best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
