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

SEED = 42
RATING_THRESHOLD = 3.0
VARIANT = "10M"  # larger than 1M; change to '1M' if runtime is too high
BASE_FRAC = 0.60
ADAPT_TOTAL_FRAC = 0.20
TEST_FRAC = 0.20
N_ADAPT_CHUNKS = 4
TOP_K = 20
RESULTS_DIR = "results"

SEQUENTIAL_CONFIGS = [
    {
        "name": "fast_partial",
        "label": "OnlineIBPRMejorado sequential partial | cosine_bpr | epochs=1 | max_steps=5 | update_V=False",
        "loss_mode": "cosine_bpr",
        "n_epochs": 1,
        "max_steps": 5,
        "update_V": False,
    },
    {
        "name": "quality_partial",
        "label": "OnlineIBPRMejorado sequential partial | angular | epochs=2 | max_steps=None | update_V=False",
        "loss_mode": "angular",
        "n_epochs": 2,
        "max_steps": None,
        "update_V": False,
    },
]


def load_positive_chrono_movielens(variant=VARIANT, rating_threshold=RATING_THRESHOLD):
    """Load MovieLens in chronological order and binarize ratings >= threshold."""
    data = movielens.load_feedback(fmt="UIRT", variant=variant)
    positive = [
        (str(u), str(i), 1.0, int(ts))
        for u, i, r, ts in data
        if float(r) >= rating_threshold
    ]
    positive.sort(key=lambda x: x[3])
    return positive


def split_base_adapt_chunks_test(data, base_frac=BASE_FRAC, adapt_total_frac=ADAPT_TOTAL_FRAC, n_adapt_chunks=N_ADAPT_CHUNKS):
    n = len(data)
    base_end = int(n * base_frac)
    adapt_end = int(n * (base_frac + adapt_total_frac))

    base_raw = data[:base_end]
    adapt_all = data[base_end:adapt_end]
    test_raw = data[adapt_end:]

    raw_chunks = np.array_split(np.array(adapt_all, dtype=object), n_adapt_chunks)
    adapt_chunks = [chunk.tolist() for chunk in raw_chunks if len(chunk) > 0]
    return base_raw, adapt_chunks, test_raw


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


def build_global_maps(base_raw, adapt_chunks_raw, test_raw):
    all_rows = list(base_raw)
    for chunk in adapt_chunks_raw:
        all_rows.extend(chunk)
    all_rows.extend(test_raw)
    if len(all_rows) == 0:
        raise ValueError("No hay datos luego del filtrado para construir el experimento.")

    global_ds = Dataset.build(all_rows, fmt="UIRT", seed=SEED)
    return global_ds.uid_map, global_ds.iid_map


def build_dataset(rows, uid_map, iid_map, exclude_unknowns=False):
    return Dataset.build(
        rows,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
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


def print_segment_summary(base_raw, adapt_chunks, test_raw):
    print("=" * 80)
    print("SEQUENTIAL PARTIAL RETRAINING SEGMENT SUMMARY")
    print("=" * 80)
    print(f"Variant         : {VARIANT}")
    print(f"Base segment    : {len(base_raw):,}")
    for idx, chunk in enumerate(adapt_chunks, 1):
        print(f"Adapt chunk {idx:>2} : {len(chunk):,}")
    print(f"Test segment    : {len(test_raw):,}")
    print(f"Users in base   : {len({u for u, _, _, _ in base_raw}):,}")
    print(f"Items in base   : {len({i for _, i, _, _ in base_raw}):,}")
    print()


def print_result_block(title, train_time, test_time, metrics_dict):
    print("-" * 80)
    print(title)
    print("-" * 80)
    print(f"Train time: {train_time:.4f} s")
    print(f"Test time : {test_time:.4f} s")
    for name, value in metrics_dict.items():
        print(f"{name:12s}: {value:.4f}")
    print()


def to_csv_row(experiment_name, model_name, stage, chunk_size, cumulative_size, train_time, test_time, metrics_dict,
               baseline_metrics=None, previous_metrics=None):
    row = {
        "experiment": experiment_name,
        "model": model_name,
        "stage": stage,
        "chunk_size": chunk_size,
        "cumulative_train_size": cumulative_size,
        "train_time_s": train_time,
        "test_time_s": test_time,
    }
    row.update(metrics_dict)

    if baseline_metrics is not None:
        for metric_name, metric_value in metrics_dict.items():
            row[f"delta_baseline_{metric_name}"] = metric_value - baseline_metrics[metric_name]

    if previous_metrics is not None:
        for metric_name, metric_value in metrics_dict.items():
            row[f"delta_prev_{metric_name}"] = metric_value - previous_metrics[metric_name]

    return row


def save_csv(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "stage",
        "chunk_size",
        "cumulative_train_size",
        "train_time_s",
        "test_time_s",
        "AUC",
        "MAP",
        "NDCG@20",
        "Precision@20",
        "Recall@20",
        "delta_baseline_AUC",
        "delta_baseline_MAP",
        "delta_baseline_NDCG@20",
        "delta_baseline_Precision@20",
        "delta_baseline_Recall@20",
        "delta_prev_AUC",
        "delta_prev_MAP",
        "delta_prev_NDCG@20",
        "delta_prev_Precision@20",
        "delta_prev_Recall@20",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"online_ibpr_mejorado_sequential_partial_{timestamp}.txt")
    csv_path = os.path.join(RESULTS_DIR, f"online_ibpr_mejorado_sequential_partial_{timestamp}.csv")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print(f"Logging results to: {log_path}")
            print(f"CSV summary will be saved to: {csv_path}")
            print()

            data = load_positive_chrono_movielens()
            base_raw, adapt_chunks_all, test_raw_all = split_base_adapt_chunks_test(data)
            adapt_chunks, test_raw = filter_to_base_known(base_raw, adapt_chunks_all, test_raw_all)

            if len(adapt_chunks) == 0 or len(test_raw) == 0:
                raise ValueError("Los segmentos adapt/test quedaron vacíos luego del filtrado a entidades conocidas.")

            print_segment_summary(base_raw, adapt_chunks, test_raw)

            uid_map, iid_map = build_global_maps(base_raw, adapt_chunks, test_raw)
            base_train_set = build_dataset(base_raw, uid_map, iid_map, exclude_unknowns=False)
            test_set = build_dataset(test_raw, uid_map, iid_map, exclude_unknowns=True)
            metrics = build_metrics()

            print("=" * 80)
            print("STEP 1 - TRAIN BASE IBPR")
            print("=" * 80)
            ibpr = IBPR(
                k=50,
                max_iter=15,
                learning_rate=0.01,
                lamda=0.001,
                batch_size=1024,
                verbose=True,
                name="IBPR (base only)",
            )

            t0 = time.perf_counter()
            ibpr.fit(base_train_set)
            ibpr_train_time = time.perf_counter() - t0
            ibpr_metrics, ibpr_test_time = evaluate_ranking(ibpr, base_train_set, test_set, metrics)
            print_result_block("Baseline IBPR trained on base segment", ibpr_train_time, ibpr_test_time, ibpr_metrics)

            all_rows = [
                to_csv_row(
                    experiment_name="sequential_partial",
                    model_name="IBPR_base",
                    stage=0,
                    chunk_size=0,
                    cumulative_size=len(base_raw),
                    train_time=ibpr_train_time,
                    test_time=ibpr_test_time,
                    metrics_dict=ibpr_metrics,
                    baseline_metrics=None,
                    previous_metrics=None,
                )
            ]

            print("=" * 80)
            print("STEP 2 - SEQUENTIAL PARTIAL RETRAINING")
            print("=" * 80)

            for cfg in SEQUENTIAL_CONFIGS:
                print(f"Running sequential config: {cfg['label']}")

                online_model = OnlineIBPRMejorado(
                    k=50,
                    max_iter=10,
                    learning_rate=0.01,
                    lamda=0.001,
                    batch_size=1024,
                    init_params={"U": ibpr.U.copy(), "V": ibpr.V.copy()},
                    update_V=cfg["update_V"],
                    neg_sampling="uniform",
                    normalize=True,
                    loss_mode=cfg["loss_mode"],
                    verbose=True,
                    name=cfg["label"],
                )

                cumulative_rows = list(base_raw)
                previous_metrics = ibpr_metrics
                cumulative_train_time = 0.0

                for stage_idx, chunk_raw in enumerate(adapt_chunks, start=1):
                    chunk_train_set = build_dataset(chunk_raw, uid_map, iid_map, exclude_unknowns=False)
                    recent_pairs = np.column_stack(
                        (chunk_train_set.uir_tuple[0], chunk_train_set.uir_tuple[1])
                    ).astype(np.int64)

                    cumulative_rows.extend(chunk_raw)
                    cumulative_train_set = build_dataset(cumulative_rows, uid_map, iid_map, exclude_unknowns=False)

                    t0 = time.perf_counter()
                    online_model.partial_fit_recent(
                        recent_pairs=recent_pairs,
                        history_csr=cumulative_train_set.csr_matrix,
                        max_steps=cfg["max_steps"],
                        n_epochs=cfg["n_epochs"],
                    )
                    step_train_time = time.perf_counter() - t0
                    cumulative_train_time += step_train_time

                    step_metrics, step_test_time = evaluate_ranking(
                        online_model, cumulative_train_set, test_set, metrics
                    )

                    print_result_block(
                        title=f"{cfg['label']} | after adapt chunk {stage_idx}/{len(adapt_chunks)} | chunk_size={len(chunk_raw):,}",
                        train_time=step_train_time,
                        test_time=step_test_time,
                        metrics_dict=step_metrics,
                    )
                    print("Delta vs baseline IBPR:")
                    for metric_name, metric_value in step_metrics.items():
                        print(f"  Δbaseline {metric_name:10s}: {metric_value - ibpr_metrics[metric_name]:+.4f}")
                    print("Delta vs previous stage:")
                    for metric_name, metric_value in step_metrics.items():
                        print(f"  Δprev     {metric_name:10s}: {metric_value - previous_metrics[metric_name]:+.4f}")
                    print(f"Cumulative partial train time so far: {cumulative_train_time:.4f} s")
                    print()

                    all_rows.append(
                        to_csv_row(
                            experiment_name="sequential_partial",
                            model_name=cfg["name"],
                            stage=stage_idx,
                            chunk_size=len(chunk_raw),
                            cumulative_size=len(cumulative_rows),
                            train_time=step_train_time,
                            test_time=step_test_time,
                            metrics_dict=step_metrics,
                            baseline_metrics=ibpr_metrics,
                            previous_metrics=previous_metrics,
                        )
                    )

                    previous_metrics = step_metrics

            save_csv(csv_path, all_rows)
            print("=" * 80)
            print("CSV SUMMARY SAVED")
            print("=" * 80)
            print(csv_path)


if __name__ == "__main__":
    main()
