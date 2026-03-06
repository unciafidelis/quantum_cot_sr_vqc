# eval.py
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

from utils import to_numpy


@torch.no_grad()
def collect_step_readouts_and_labels(srvqc_model, loader, device: str):
    srvqc_model.eval()
    Z = []   # list of [K, B, n_qubits] chunks
    Y = []
    for batch in loader:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)
        y = batch["label"].detach().cpu().numpy()
        out = srvqc_model(batch)
        step_readouts = out["step_readouts"].detach().cpu().numpy()  # [K,B,nq]
        Z.append(step_readouts)
        Y.append(y)
    Z = np.concatenate(Z, axis=1)  # concat in batch dimension
    Y = np.concatenate(Y, axis=0)
    return Z, Y  # Z: [K, N, n_qubits], Y: [N]


def mutual_information_per_step(step_readouts: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
    """
    step_readouts: [K, N, d]
    y: [N], discrete
    Returns MI_k aggregated as mean across features.
    """
    K, N, d = step_readouts.shape
    mi = np.zeros((K,), dtype=np.float64)
    for k in range(K):
        X = step_readouts[k]  # [N, d]
        # mutual_info_classif returns MI per feature
        mi_feat = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=0)
        mi[k] = float(np.mean(mi_feat))
    return mi


def information_gain(mi: np.ndarray) -> np.ndarray:
    ig = np.zeros_like(mi)
    ig[0] = mi[0]
    for k in range(1, len(mi)):
        ig[k] = mi[k] - mi[k - 1]
    return ig


def gradient_variance(model, batch, device: str) -> float:
    """
    Single-batch gradient variance proxy.
    """
    model.train()
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)
    y = batch["label"]
    out = model(batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], y)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().flatten()
        if g.numel() > 0:
            grads.append(g)
    if not grads:
        return float("nan")
    gvec = torch.cat(grads, dim=0)
    return float(torch.var(gvec).item())
