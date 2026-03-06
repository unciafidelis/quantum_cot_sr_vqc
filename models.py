# models.py
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import pennylane as qml


@dataclass
class NoiseConfig:
    p_depol: float = 0.0
    gamma_amp: float = 0.0
    readout_prob: float = 0.0


class FeatureEncoder(nn.Module):
    """
    Encoder ligero compartido para modelos cuánticos:
    tokens -> embedding -> mean pool -> linear -> tanh -> scaled angles
    """
    def __init__(self, vocab_size: int, d_emb: int, n_qubits: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=0)
        self.proj = nn.Linear(d_emb, n_qubits)
        self.n_qubits = n_qubits

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L], attention_mask: [B, L]
        x = self.emb(input_ids)  # [B, L, d]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))  # mean pool
        x = torch.tanh(self.proj(x)) * np.pi  # [-pi, pi]
        return x  # [B, n_qubits]


def make_device(n_qubits: int, shots: Optional[int], seed: int, noise: NoiseConfig):
    # default.mixed supports shots/seed/readout_prob
    return qml.device("default.mixed", wires=n_qubits, shots=shots, seed=seed, readout_prob=noise.readout_prob)


def block_ansatz(x: torch.Tensor, theta_block: torch.Tensor, n_qubits: int, noise: NoiseConfig):
    """
    One block:
      - data reupload: RY(x_i)
      - trainable: Rot(alpha,beta,gamma)
      - entangle ring
      - noise channels (optional)
    """
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    for i in range(n_qubits):
        a, b, c = theta_block[i, 0], theta_block[i, 1], theta_block[i, 2]
        qml.Rot(a, b, c, wires=i)

    # ring entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])

    # noise (applied per qubit after the block)
    if noise.p_depol > 0:
        for i in range(n_qubits):
            qml.DepolarizingChannel(noise.p_depol, wires=i)
    if noise.gamma_amp > 0:
        for i in range(n_qubits):
            qml.AmplitudeDamping(noise.gamma_amp, wires=i)


def build_step_qnode(dev, n_qubits: int, K: int, step: int, noise: NoiseConfig):
    """
    QNode returning Z expectations after applying blocks 1..step.
    step in [1..K].
    """
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(x: torch.Tensor, theta: torch.Tensor):
        # theta: [K, n_qubits, 3]
        for k in range(step):
            block_ansatz(x, theta[k], n_qubits, noise)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit


class SRVQC(nn.Module):
    """
    Stepwise-readout VQC (Quantum CoT):
      - returns step-wise logits, final logits
      - trains with multi-step CE losses (handled in train.py)
    """
    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        n_qubits: int,
        n_classes: int,
        K: int,
        shots: Optional[int],
        seed: int,
        noise: NoiseConfig,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(vocab_size, d_emb, n_qubits)
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.K = K
        self.noise = noise

        # Shared quantum parameters
        # init small to avoid instabilities
        theta0 = 0.01 * torch.randn(K, n_qubits, 3)
        self.theta = nn.Parameter(theta0)

        self.head = nn.Linear(n_qubits, n_classes)

        # Build K QNodes sharing same device; note: device seed fixed
        self.dev = make_device(n_qubits, shots=shots, seed=seed, noise=noise)
        self.qnodes = nn.ModuleList()  # placeholder for torch state dict
        self._circuits = []
        for step in range(1, K + 1):
            self._circuits.append(build_step_qnode(self.dev, n_qubits, K, step, noise))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        x = self.encoder(input_ids, attention_mask)  # [B, n_qubits]

        step_logits = []
        step_readouts = []

        # Note: per-sample loop for simplicity; for speed you can vectorize with qml.batch_input transforms.
        for k in range(1, self.K + 1):
            z_list = []
            for b in range(x.shape[0]):
                z = self._circuits[k - 1](x[b], self.theta)  # [n_qubits]
                z_list.append(torch.stack(z) if isinstance(z, list) else z)
            z_k = torch.stack(z_list, dim=0)  # [B, n_qubits]
            step_readouts.append(z_k)
            step_logits.append(self.head(z_k))  # [B, C]

        final_logits = step_logits[-1]
        return {
            "step_logits": torch.stack(step_logits, dim=0),   # [K, B, C]
            "step_readouts": torch.stack(step_readouts, dim=0),  # [K, B, n_qubits]
            "logits": final_logits,  # [B, C]
        }


class VQCEndOnly(nn.Module):
    """
    Same quantum circuit depth (K blocks) but only final readout used.
    """
    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        n_qubits: int,
        n_classes: int,
        K: int,
        shots: Optional[int],
        seed: int,
        noise: NoiseConfig,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(vocab_size, d_emb, n_qubits)
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.K = K
        self.noise = noise

        theta0 = 0.01 * torch.randn(K, n_qubits, 3)
        self.theta = nn.Parameter(theta0)
        self.head = nn.Linear(n_qubits, n_classes)

        self.dev = make_device(n_qubits, shots=shots, seed=seed, noise=noise)
        self.circuitK = build_step_qnode(self.dev, n_qubits, K, step=K, noise=noise)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        x = self.encoder(input_ids, attention_mask)  # [B, n_qubits]

        z_list = []
        for b in range(x.shape[0]):
            z = self.circuitK(x[b], self.theta)
            z_list.append(torch.stack(z) if isinstance(z, list) else z)
        zK = torch.stack(z_list, dim=0)  # [B, n_qubits]
        logits = self.head(zK)
        return {"logits": logits, "readout": zK}


class MLPClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_emb: int, n_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.emb(batch["input_ids"])
        mask = batch["attention_mask"].unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {"logits": self.mlp(x)}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(-np.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, n_classes: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.emb(batch["input_ids"])
        x = self.pos(x)
        # key_padding_mask: True where padding
        key_padding_mask = (batch["attention_mask"] == 0)
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        # mean pool on non-pad
        mask = batch["attention_mask"].unsqueeze(-1).float()
        h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {"logits": self.head(h)}
