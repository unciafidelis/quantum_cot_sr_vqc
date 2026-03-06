#!/usr/bin/env bash
set -e

# Ejemplo mínimo. Ajusta rutas y flags según tu entorno.
# Requiere: python, torch, pennylane, scikit-learn, xgboost, matplotlib.

DATA_DIR="./data"
OUT_DIR="./runs"

# 1) Preparar dataset y metadata
python -c "from data import prepare_listops; prepare_listops('${DATA_DIR}', max_len=512, depth_threshold=12)"

# 2) Entrenar SR-VQC (ejemplo: 8 qubits, K=4, ruido moderado)
python main.py \
  --model srvqc --data_dir ${DATA_DIR} --out_dir ${OUT_DIR}/srvqc_q8_k4 \
  --n_qubits 8 --K 4 --shots 1024 --p_depol 0.005 --gamma_amp 0.005 --readout_prob 0.02 \
  --epochs 20 --batch_size 4 --lr 1e-3 --seeds 0 1 2 3 4

# 3) Entrenar VQC end-only
python main.py \
  --model vqc_end --data_dir ${DATA_DIR} --out_dir ${OUT_DIR}/vqc_end_q8_k4 \
  --n_qubits 8 --K 4 --shots 1024 --p_depol 0.005 --gamma_amp 0.005 --readout_prob 0.02 \
  --epochs 20 --batch_size 4 --lr 1e-3 --seeds 0 1 2 3 4
