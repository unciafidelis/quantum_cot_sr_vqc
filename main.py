# main.py
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import prepare_listops, ListOpsDataset
from models import SRVQC, VQCEndOnly, MLPClassifier, TinyTransformer, NoiseConfig
from train import fit, eval_model
from eval import collect_step_readouts_and_labels, mutual_information_per_step, information_gain
from utils import set_global_seed, ensure_dir, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["srvqc", "vqc_end", "mlp", "transformer"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_len", type=int, default=512)

    p.add_argument("--n_qubits", type=int, default=8)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--shots", type=int, default=1024)

    p.add_argument("--p_depol", type=float, default=0.0)
    p.add_argument("--gamma_amp", type=float, default=0.0)
    p.add_argument("--readout_prob", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--depth_threshold", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    prep = prepare_listops(args.data_dir, max_len=args.max_len, depth_threshold=args.depth_threshold)
    vocab = prep["vocab"]
    n_classes = 10

    train_ds = ListOpsDataset(prep["train_examples"], vocab, args.max_len)
    val_ds = ListOpsDataset(prep["val_examples"], vocab, args.max_len)
    test_ds = ListOpsDataset(prep["test_examples"], vocab, args.max_len)

    results = {"args": vars(args), "seeds": []}

    for seed in args.seeds:
        set_global_seed(seed)
        run_dir = os.path.join(args.out_dir, f"seed_{seed}")
        ensure_dir(run_dir)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        noise = NoiseConfig(p_depol=args.p_depol, gamma_amp=args.gamma_amp, readout_prob=args.readout_prob)

        if args.model == "srvqc":
            model = SRVQC(
                vocab_size=len(vocab), d_emb=32, n_qubits=args.n_qubits, n_classes=n_classes,
                K=args.K, shots=args.shots, seed=seed, noise=noise
            )
            best = fit(model, train_loader, val_loader, args.device, run_dir,
                       epochs=args.epochs, lr=args.lr, early_patience=10, srvqc_K=args.K, beta_kl=0.0)
            # load best
            model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=args.device))
            test_metrics = eval_model(model.to(args.device), test_loader, args.device)

            # CoT metrics: MI_k on validation set
            Z, Y = collect_step_readouts_and_labels(model.to(args.device), val_loader, args.device)
            mi = mutual_information_per_step(Z, Y, n_neighbors=3)
            ig = information_gain(mi)

            seed_out = {"seed": seed, "best": best, "test": test_metrics,
                        "mi_per_step": mi.tolist(), "ig_per_step": ig.tolist()}
        elif args.model == "vqc_end":
            model = VQCEndOnly(
                vocab_size=len(vocab), d_emb=32, n_qubits=args.n_qubits, n_classes=n_classes,
                K=args.K, shots=args.shots, seed=seed, noise=noise
            )
            best = fit(model, train_loader, val_loader, args.device, run_dir, epochs=args.epochs, lr=args.lr)
            model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=args.device))
            test_metrics = eval_model(model.to(args.device), test_loader, args.device)
            seed_out = {"seed": seed, "best": best, "test": test_metrics}
        elif args.model == "mlp":
            model = MLPClassifier(vocab_size=len(vocab), d_emb=64, n_classes=n_classes)
            best = fit(model, train_loader, val_loader, args.device, run_dir, epochs=args.epochs, lr=args.lr)
            model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=args.device))
            test_metrics = eval_model(model.to(args.device), test_loader, args.device)
            seed_out = {"seed": seed, "best": best, "test": test_metrics}
        else:
            model = TinyTransformer(vocab_size=len(vocab), d_model=128, n_heads=4, n_layers=2, n_classes=n_classes, max_len=args.max_len)
            best = fit(model, train_loader, val_loader, args.device, run_dir, epochs=args.epochs, lr=args.lr)
            model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=args.device))
            test_metrics = eval_model(model.to(args.device), test_loader, args.device)
            seed_out = {"seed": seed, "best": best, "test": test_metrics}

        save_json(seed_out, os.path.join(run_dir, "summary.json"))
        results["seeds"].append(seed_out)

    save_json(results, os.path.join(args.out_dir, "all_results.json"))
    print("[done] saved:", os.path.join(args.out_dir, "all_results.json"))


if __name__ == "__main__":
    main()
