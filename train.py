# train.py
import os
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import expected_calibration_error, ensure_dir, save_json, timer, to_numpy


def compute_metrics(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "ece": float(expected_calibration_error(probs, y_true, n_bins=15)),
    }
    # Multiclass AUC: OvR (when feasible)
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    return out


def ce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, y)


def srvqc_loss(outputs: Dict[str, torch.Tensor], y: torch.Tensor, lambdas: torch.Tensor, beta_kl: float = 0.0) -> torch.Tensor:
    """
    outputs["step_logits"]: [K, B, C]
    """
    step_logits = outputs["step_logits"]
    K = step_logits.shape[0]
    loss = 0.0
    for k in range(K):
        loss = loss + lambdas[k] * ce_loss(step_logits[k], y)

    # Optional consistency regularizer (KL between consecutive steps)
    if beta_kl > 0.0 and K > 1:
        kl = 0.0
        for k in range(1, K):
            p = torch.nn.functional.log_softmax(step_logits[k], dim=-1)
            q = torch.nn.functional.softmax(step_logits[k - 1].detach(), dim=-1)
            kl = kl + torch.nn.functional.kl_div(p, q, reduction="batchmean")
        loss = loss + beta_kl * kl
    return loss


def train_one_epoch(model, loader, optimizer, device, is_srvqc: bool, lambdas: Optional[torch.Tensor], beta_kl: float):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

        y = batch["label"]
        optimizer.zero_grad()
        outputs = model(batch)

        if is_srvqc:
            loss = srvqc_loss(outputs, y, lambdas=lambdas.to(device), beta_kl=beta_kl)
        else:
            loss = ce_loss(outputs["logits"], y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.shape[0]
        n += y.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_y = []
    for batch in loader:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)
        y = batch["label"].detach().cpu().numpy()
        out = model(batch)
        logits = out["logits"].detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y)
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0)
    return compute_metrics(logits, y)


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    out_dir: str,
    epochs: int = 30,
    lr: float = 1e-3,
    early_patience: int = 10,
    srvqc_K: int = 0,
    beta_kl: float = 0.0,
):
    ensure_dir(out_dir)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"val_accuracy": -1.0, "epoch": -1}
    patience = 0

    if srvqc_K > 0:
        lambdas = torch.linspace(0.5, 1.0, srvqc_K)  # emphasize later steps
        is_srvqc = True
    else:
        lambdas = None
        is_srvqc = False

    history = []
    t0, elapsed = timer()

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, is_srvqc, lambdas, beta_kl)
        val = eval_model(model, val_loader, device)

        rec = {"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)

        if val["accuracy"] > best["val_accuracy"]:
            best = {"val_accuracy": val["accuracy"], "epoch": epoch}
            patience = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        else:
            patience += 1

        if patience >= early_patience:
            break

    save_json({"best": best, "history": history, "elapsed_sec": elapsed()}, os.path.join(out_dir, "train_log.json"))
    return best
