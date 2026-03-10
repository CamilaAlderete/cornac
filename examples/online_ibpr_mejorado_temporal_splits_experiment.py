
import csv
import os
import sys
import time
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import cornac
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.utils.Tee import Tee

RATING_THRESHOLD = 3.0
TOP_K = 20
RESULTS_DIR = "results"

PARTIAL_CONFIG = {
    "name": "fast_partial",
    "label": "OnlineIBPRMejorado partial | cosine_bpr | epochs=1 | max_steps=5 | update_V=False",
    "loss_mode": "cosine_bpr",
    "n_epochs": 1,
    "max_steps": 5,
    "update_V": False,
}

def load_positive_chrono_movielens(variant, rating_threshold=RATING_THRESHOLD):
    data = movielens.load_feedback(fmt="UIRT", variant=variant)
    positive = [
        (str(u), str(i), 1.0, int(ts))
        for u, i, r, ts in data
        if float(r) >= rating_threshold
    ]
    positive.sort(key=lambda x: x[3])
    return positive

def filter_to_base_known(base_raw, adapt_chunks_raw, test_raw):
    known_users = {u for u, _, _, _ in base_raw}
    known_items = {i for _, i, _, _ in base_raw}
    filtered_chunks = []
    for chunk in adapt_chunks_raw:
        filtered = [x for x in chunk if x[0] in known_users and x[1] in known_items]
        if len(filtered) > 0:
            filtered_chunks.append(filtered)
    filtered_test = [x for x in test_raw if x[0] in known_users and x[1] in known_items]
    return filtered_chunks, filtered_test

def build_global_maps(base_raw, adapt_chunks_raw, test_raw, seed):
    all_rows = list(base_raw)
    for chunk in adapt_chunks_raw:
        all_rows.extend(chunk)
    all_rows.extend(test_raw)
    if len(all_rows) == 0:
        raise ValueError("No hay datos luego del filtrado para construir el experimento.")
    global_ds = Dataset.build(all_rows, fmt="UIRT", seed=seed)
    return global_ds.uid_map, global_ds.iid_map

def build_dataset(rows, uid_map, iid_map, seed, exclude_unknowns=False):
    return Dataset.build(
        rows,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=seed,
        exclude_unknowns=exclude_unknowns,
    )

def build_metrics(k=TOP_K):
    return [
        cornac.metrics.AUC(),
        cornac.metrics.MAP(),
        cornac.metrics.NDCG(k=k),
        cornac.metrics.Precision(k=k),
        cornac.metrics.Recall(k=k),
    ]

def evaluate_ranking(model, train_set, test_set, metrics):
    start = time.perf_counter()
    avg_results, _ = ranking_eval(
        model=model,
        metrics=metrics,
        train_set=train_set,
        test_set=test_set,
        val_set=None,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - start
    return OrderedDict((m.name, r) for m, r in zip(metrics, avg_results)), elapsed

def print_result_block(title, train_time, test_time, metrics_dict):
    print("-" * 80)
    print(title)
    print("-" * 80)
    print(f"Train time: {train_time:.4f} s")
    print(f"Test time : {test_time:.4f} s")
    for name, value in metrics_dict.items():
        print(f"{name:12s}: {value:.4f}")
    print()

def save_csv(filepath, rows, fieldnames):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

VARIANT = "1M"
SEED = 42
TEMPORAL_SPLITS = [
    {"name": "split_50_20_30", "base_frac": 0.50, "adapt_frac": 0.20},
    {"name": "split_60_20_20", "base_frac": 0.60, "adapt_frac": 0.20},
    {"name": "split_70_15_15", "base_frac": 0.70, "adapt_frac": 0.15},
]
N_ADAPT_CHUNKS = 4

def split_base_adapt_chunks_test(data, base_frac, adapt_total_frac, n_adapt_chunks=N_ADAPT_CHUNKS):
    n = len(data)
    base_end = int(n * base_frac)
    adapt_end = int(n * (base_frac + adapt_total_frac))
    base_raw = data[:base_end]
    adapt_all = data[base_end:adapt_end]
    test_raw = data[adapt_end:]
    raw_chunks = np.array_split(np.array(adapt_all, dtype=object), n_adapt_chunks)
    adapt_chunks = [chunk.tolist() for chunk in raw_chunks if len(chunk) > 0]
    return base_raw, adapt_chunks, test_raw

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"online_ibpr_mejorado_temporal_splits_{timestamp}.txt")
    csv_path = os.path.join(RESULTS_DIR, f"online_ibpr_mejorado_temporal_splits_{timestamp}.csv")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print(f"Logging results to: {log_path}")
            print(f"CSV summary will be saved to: {csv_path}")
            print()

            data = load_positive_chrono_movielens(VARIANT)
            metrics = build_metrics()
            rows = []

            for split_cfg in TEMPORAL_SPLITS:
                print("=" * 80)
                print(f"TEMPORAL SPLIT EXPERIMENT | {split_cfg['name']}")
                print("=" * 80)

                base_raw, adapt_chunks_all, test_raw_all = split_base_adapt_chunks_test(
                    data, split_cfg["base_frac"], split_cfg["adapt_frac"], N_ADAPT_CHUNKS
                )
                adapt_chunks, test_raw = filter_to_base_known(base_raw, adapt_chunks_all, test_raw_all)
                uid_map, iid_map = build_global_maps(base_raw, adapt_chunks, test_raw, SEED)

                base_train_set = build_dataset(base_raw, uid_map, iid_map, SEED, exclude_unknowns=False)
                test_set = build_dataset(test_raw, uid_map, iid_map, SEED, exclude_unknowns=True)

                ibpr = IBPR(
                    k=50, max_iter=15, learning_rate=0.01, lamda=0.001, batch_size=1024,
                    verbose=True, name=f"IBPR base | {split_cfg['name']}"
                )
                t0 = time.perf_counter()
                ibpr.fit(base_train_set)
                ibpr_train_time = time.perf_counter() - t0
                ibpr_metrics, ibpr_test_time = evaluate_ranking(ibpr, base_train_set, test_set, metrics)
                print_result_block(f"IBPR baseline | {split_cfg['name']}", ibpr_train_time, ibpr_test_time, ibpr_metrics)

                online = OnlineIBPRMejorado(
                    k=50, max_iter=10, learning_rate=0.01, lamda=0.001, batch_size=1024,
                    init_params={"U": ibpr.U.copy(), "V": ibpr.V.copy()},
                    update_V=PARTIAL_CONFIG["update_V"], neg_sampling="uniform",
                    normalize=True, loss_mode=PARTIAL_CONFIG["loss_mode"], verbose=True,
                    name=f"{PARTIAL_CONFIG['label']} | {split_cfg['name']}",
                )

                cumulative_rows = list(base_raw)
                cumulative_partial_train_time = 0.0
                previous_metrics = ibpr_metrics

                for stage_idx, chunk_raw in enumerate(adapt_chunks, start=1):
                    chunk_train_set = build_dataset(chunk_raw, uid_map, iid_map, SEED, exclude_unknowns=False)
                    recent_pairs = np.column_stack((chunk_train_set.uir_tuple[0], chunk_train_set.uir_tuple[1])).astype(np.int64)
                    cumulative_rows.extend(chunk_raw)
                    cumulative_train_set = build_dataset(cumulative_rows, uid_map, iid_map, SEED, exclude_unknowns=False)

                    t0 = time.perf_counter()
                    online.partial_fit_recent(
                        recent_pairs=recent_pairs,
                        history_csr=cumulative_train_set.csr_matrix,
                        max_steps=PARTIAL_CONFIG["max_steps"],
                        n_epochs=PARTIAL_CONFIG["n_epochs"],
                    )
                    step_train_time = time.perf_counter() - t0
                    cumulative_partial_train_time += step_train_time
                    step_metrics, step_test_time = evaluate_ranking(online, cumulative_train_set, test_set, metrics)

                    print_result_block(
                        f"{PARTIAL_CONFIG['label']} | {split_cfg['name']} | stage={stage_idx}/{len(adapt_chunks)}",
                        step_train_time, step_test_time, step_metrics
                    )

                    rows.append({
                        "split_name": split_cfg["name"],
                        "base_frac": split_cfg["base_frac"],
                        "adapt_frac": split_cfg["adapt_frac"],
                        "test_frac": 1.0 - split_cfg["base_frac"] - split_cfg["adapt_frac"],
                        "stage": stage_idx,
                        "chunk_size": len(chunk_raw),
                        "cumulative_partial_train_time_s": cumulative_partial_train_time,
                        "ibpr_AUC": ibpr_metrics["AUC"],
                        "ibpr_MAP": ibpr_metrics["MAP"],
                        "ibpr_NDCG@20": ibpr_metrics["NDCG@20"],
                        "ibpr_Precision@20": ibpr_metrics["Precision@20"],
                        "ibpr_Recall@20": ibpr_metrics["Recall@20"],
                        "AUC": step_metrics["AUC"],
                        "MAP": step_metrics["MAP"],
                        "NDCG@20": step_metrics["NDCG@20"],
                        "Precision@20": step_metrics["Precision@20"],
                        "Recall@20": step_metrics["Recall@20"],
                        "delta_baseline_AUC": step_metrics["AUC"] - ibpr_metrics["AUC"],
                        "delta_baseline_MAP": step_metrics["MAP"] - ibpr_metrics["MAP"],
                        "delta_baseline_NDCG@20": step_metrics["NDCG@20"] - ibpr_metrics["NDCG@20"],
                        "delta_baseline_Precision@20": step_metrics["Precision@20"] - ibpr_metrics["Precision@20"],
                        "delta_baseline_Recall@20": step_metrics["Recall@20"] - ibpr_metrics["Recall@20"],
                        "delta_prev_AUC": step_metrics["AUC"] - previous_metrics["AUC"],
                        "delta_prev_MAP": step_metrics["MAP"] - previous_metrics["MAP"],
                        "delta_prev_NDCG@20": step_metrics["NDCG@20"] - previous_metrics["NDCG@20"],
                        "delta_prev_Precision@20": step_metrics["Precision@20"] - previous_metrics["Precision@20"],
                        "delta_prev_Recall@20": step_metrics["Recall@20"] - previous_metrics["Recall@20"],
                    })
                    previous_metrics = step_metrics

            save_csv(csv_path, rows, list(rows[0].keys()))
            print("=" * 80)
            print("TEMPORAL SPLIT EXPERIMENT SAVED")
            print("=" * 80)
            print(csv_path)

if __name__ == "__main__":
    main()
