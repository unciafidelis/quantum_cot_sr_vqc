# data.py
import os
import re
import tarfile
import hashlib
import urllib.request
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import ensure_dir, save_json

# Public release URL described by LRA repo (README).
LRA_RELEASE_URL = "https://storage.googleapis.com/long-range-arena/lra_release.gz"

TOKEN_RE = re.compile(r"\[|\]|[A-Za-z_]+|\d+")


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_lra_release(out_dir: str, url: str = LRA_RELEASE_URL, filename: str = "lra_release.gz") -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        return out_path
    print(f"[data] Descargando {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def extract_gz(gz_path: str, out_dir: str) -> str:
    """
    The LRA release is distributed as a .gz tar archive containing a folder (commonly 'lra_release/').
    """
    ensure_dir(out_dir)
    with tarfile.open(gz_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    return out_dir


def find_listops_files(root_dir: str) -> Dict[str, str]:
    """
    Busca archivos TSV de ListOps en el árbol extraído.
    Espera encontrar: *train*.tsv, *val*.tsv, *test*.tsv (nombres varían).
    """
    candidates = {"train": None, "val": None, "test": None}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.endswith(".tsv"):
                continue
            lower = fn.lower()
            full = os.path.join(dirpath, fn)
            if "listops" not in dirpath.lower() and "listops" not in lower:
                continue
            if "train" in lower and candidates["train"] is None:
                candidates["train"] = full
            elif ("val" in lower or "valid" in lower or "dev" in lower) and candidates["val"] is None:
                candidates["val"] = full
            elif "test" in lower and candidates["test"] is None:
                candidates["test"] = full

    missing = [k for k, v in candidates.items() if v is None]
    if missing:
        raise FileNotFoundError(f"No se pudieron localizar splits ListOps {missing}. "
                                f"Revisa el árbol en {root_dir}")
    return candidates


def tokenize_listops(s: str) -> List[str]:
    return TOKEN_RE.findall(s)


def parse_listops_tsv(path: str) -> List[Tuple[List[str], int]]:
    """
    Parser robusto (etiqueta 0-9 y el resto como secuencia).
    Supone TSV con 2 columnas (label, input) o (input, label).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            a, b = parts[0], parts[1]
            # Identify which is label
            if a.isdigit() and 0 <= int(a) <= 9:
                y = int(a)
                seq = b
            elif b.isdigit() and 0 <= int(b) <= 9:
                y = int(b)
                seq = a
            else:
                # fallback: look for last numeric token
                nums = [p for p in parts if p.isdigit()]
                if not nums:
                    continue
                y = int(nums[-1])
                seq = parts[0]

            tokens = tokenize_listops(seq)
            rows.append((tokens, y))
    return rows


def depth_from_tokens(tokens: List[str]) -> int:
    depth = 0
    max_depth = 0
    for t in tokens:
        if t == "[":
            depth += 1
            max_depth = max(max_depth, depth)
        elif t == "]":
            depth = max(0, depth - 1)
    return max_depth


def build_vocab(examples: List[Tuple[List[str], int]], min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter
    c = Counter()
    for toks, _ in examples:
        c.update(toks)
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in c.items():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.zeros((max_len,), dtype=np.int64)
    mask = np.zeros((max_len,), dtype=np.int64)
    for i in range(min(max_len, len(tokens))):
        ids[i] = vocab.get(tokens[i], vocab["<unk>"])
        mask[i] = 1
    return ids, mask


class ListOpsDataset(Dataset):
    def __init__(self, examples, vocab: Dict[str, int], max_len: int):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        toks, y = self.examples[idx]
        ids, mask = encode_tokens(toks, self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long),
            "depth": torch.tensor(depth_from_tokens(toks), dtype=torch.long),
            "length": torch.tensor(len(toks), dtype=torch.long),
        }


def dataset_stats(examples: List[Tuple[List[str], int]]) -> Dict:
    lengths = np.array([len(t) for t, _ in examples], dtype=np.int32)
    depths = np.array([depth_from_tokens(t) for t, _ in examples], dtype=np.int32)
    labels = np.array([y for _, y in examples], dtype=np.int32)
    hist = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    return {
        "n": int(len(examples)),
        "len_min": int(lengths.min()) if len(lengths) else 0,
        "len_p50": float(np.percentile(lengths, 50)) if len(lengths) else 0,
        "len_p90": float(np.percentile(lengths, 90)) if len(lengths) else 0,
        "len_max": int(lengths.max()) if len(lengths) else 0,
        "depth_min": int(depths.min()) if len(depths) else 0,
        "depth_p50": float(np.percentile(depths, 50)) if len(depths) else 0,
        "depth_p90": float(np.percentile(depths, 90)) if len(depths) else 0,
        "depth_max": int(depths.max()) if len(depths) else 0,
        "label_hist": hist,
    }


def make_splits_with_depth_ood(examples, depth_threshold: int = 10, seed: int = 0):
    """
    Train: depth <= threshold
    Test (OOD): depth > threshold
    Validation: random 10% from train partition (deterministic via seed)
    """
    rng = np.random.default_rng(seed)
    shallow = [(t, y) for (t, y) in examples if depth_from_tokens(t) <= depth_threshold]
    deep = [(t, y) for (t, y) in examples if depth_from_tokens(t) > depth_threshold]

    n_val = max(1, int(0.1 * len(shallow)))
    idx = rng.permutation(len(shallow))
    val_idx = set(idx[:n_val].tolist())
    train = [shallow[i] for i in range(len(shallow)) if i not in val_idx]
    val = [shallow[i] for i in range(len(shallow)) if i in val_idx]
    test_ood = deep
    return train, val, test_ood


def prepare_listops(data_dir: str, max_len: int = 512, depth_threshold: Optional[int] = None) -> Dict:
    """
    End-to-end:
      1) download & extract
      2) locate TSV
      3) parse splits
      4) build vocab from train
      5) stats + metadata JSON
    """
    ensure_dir(data_dir)
    gz_path = download_lra_release(data_dir)
    sha = sha256_file(gz_path)
    extract_root = os.path.join(data_dir, "extracted")
    if not os.path.exists(extract_root) or len(os.listdir(extract_root)) == 0:
        extract_gz(gz_path, extract_root)

    files = find_listops_files(extract_root)
    train_ex = parse_listops_tsv(files["train"])
    val_ex = parse_listops_tsv(files["val"])
    test_ex = parse_listops_tsv(files["test"])

    if depth_threshold is not None:
        # Build an OOD split from TRAIN+VAL (optional), keep official test separately if desired
        combined = train_ex + val_ex
        train_ex, val_ex, test_ood = make_splits_with_depth_ood(combined, depth_threshold=depth_threshold)
    else:
        test_ood = None

    vocab = build_vocab(train_ex, min_freq=1)

    meta = {
        "release_url": LRA_RELEASE_URL,
        "release_sha256": sha,
        "files": files,
        "max_len": max_len,
        "stats": {
            "train": dataset_stats(train_ex),
            "val": dataset_stats(val_ex),
            "test": dataset_stats(test_ex),
            "test_ood": dataset_stats(test_ood) if test_ood else None
        },
    }
    save_json(meta, os.path.join(data_dir, "listops_meta.json"))

    return {
        "train_examples": train_ex,
        "val_examples": val_ex,
        "test_examples": test_ex,
        "test_ood_examples": test_ood,
        "vocab": vocab,
        "meta": meta,
    }
