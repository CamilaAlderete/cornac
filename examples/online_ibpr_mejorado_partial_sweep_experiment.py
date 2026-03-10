import csv
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
import cornac
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.utils.Tee import Tee

SEED = 42
RATING_THRESHOLD = 3.0
VARIANT = "1M"
BASE_FRAC = 0.70
ADAPT_FRAC = 0.10
TEST_FRAC = 0.20
TOP_K = 20
RESULTS_DIR = "results"


PARTIAL_CONFIGS = [
    {
        "name": "partial_cosine_e1_steps5",
        "label": "OnlineIBPRMejorado partial | cosine_bpr | epochs=1 | max_steps=5 | update_V=False",
        "n_epochs": 1,
        "max_steps": 5,
        "loss_mode": "cosine_bpr",
        "update_V": False,
    },
    {
        "name": "partial_cosine_e1_steps20",
        "label": "OnlineIBPRMejorado partial | cosine_bpr | epochs=1 | max_steps=20 | update_V=False",
        "n_epochs": 1,
        "max_steps": 20,
        "loss_mode": "cosine_bpr",
        "update_V": False,
    },
    {
        "name": "partial_cosine_e1_all",
        "label": "OnlineIBPRMejorado partial | cosine_bpr | epochs=1 | max_steps=None | update_V=False",
        "n_epochs": 1,
        "max_steps": None,
        "loss_mode": "cosine_bpr",
        "update_V": False,
    },
    {
        "name": "partial_cosine_e2_all",
        "label": "OnlineIBPRMejorado partial | cosine_bpr | epochs=2 | max_steps=None | update_V=False",
        "n_epochs": 2,
        "max_steps": None,
        "loss_mode": "cosine_bpr",
        "update_V": False,
    },
    {
        "name": "partial_cosine_e3_all",
        "label": "OnlineIBPRMejorado partial | cosine_bpr | epochs=3 | max_steps=None | update_V=False",
        "n_epochs": 3,
        "max_steps": None,
        "loss_mode": "cosine_bpr",
        "update_V": False,
    },
    {
        "name": "partial_angular_e1_all",
        "label": "OnlineIBPRMejorado partial | angular | epochs=1 | max_steps=None | update_V=False",
        "n_epochs": 1,
        "max_steps": None,
        "loss_mode": "angular",
        "update_V": False,
    },
    {
        "name": "partial_angular_e2_all",
        "label": "OnlineIBPRMejorado partial | angular | epochs=2 | max_steps=None | update_V=False",
        "n_epochs": 2,
        "max_steps": None,
        "loss_mode": "angular",
        "update_V": False,
    },
    {
        "name": "partial_cosine_e1_all_updateV",
        "label": "OnlineIBPRMejorado partial | cosine_bpr | epochs=1 | max_steps=None | update_V=True",
        "n_epochs": 1,
        "max_steps": None,
        "loss_mode": "cosine_bpr",
        "update_V": True,
    },
]


def load_positive_chrono_movielens_1m(rating_threshold=RATING_THRESHOLD):
    data = movielens.load_feedback(fmt="UIRT", variant=VARIANT)
    positive = [
        (str(u), str(i), 1.0, int(ts))
        for u, i, r, ts in data
        if float(r) >= rating_threshold
    ]
    positive.sort(key=lambda x: x[3])
    return positive


def split_base_adapt_test(data, base_frac=BASE_FRAC, adapt_frac=ADAPT_FRAC):
    n = len(data)
    base_end = int(n * base_frac)
    adapt_end = int(n * (base_frac + adapt_frac))
    base_raw = data[:base_end]
    adapt_raw = data[base_end:adapt_end]
    test_raw = data[adapt_end:]
    return base_raw, adapt_raw, test_raw


def filter_to_base_known(base_raw, adapt_raw, test_raw):
    known_users = {u for u, _, _, _ in base_raw}
    known_items = {i for _, i, _, _ in base_raw}
    adapt_filtered = [x for x in adapt_raw if x[0] in known_users and x[1] in known_items]
    test_filtered = [x for x in test_raw if x[0] in known_users and x[1] in known_items]
    return adapt_filtered, test_filtered


def build_datasets(base_raw, adapt_raw, test_raw):
    all_rows = base_raw + adapt_raw + test_raw
    if len(all_rows) == 0:
        raise ValueError("No hay datos luego del filtrado para construir el experimento.")

    global_ds = Dataset.build(all_rows, fmt="UIRT", seed=SEED)
    uid_map = global_ds.uid_map
    iid_map = global_ds.iid_map

    base_train_set = Dataset.build(
        base_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=False,
    )
    adapt_train_set = Dataset.build(
        adapt_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=False,
    )
    combined_train_set = Dataset.build(
        base_raw + adapt_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=False,
    )
    test_set = Dataset.build(
        test_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=True,
    )
    return base_train_set, adapt_train_set, combined_train_set, test_set


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


def print_segment_summary(base_raw, adapt_raw, test_raw):
    print("=" * 80)
    print("SEGMENT SUMMARY")
    print("=" * 80)
    print(f"Base segment   : {len(base_raw):,}")
    print(f"Adapt segment  : {len(adapt_raw):,}")
    print(f"Test segment   : {len(test_raw):,}")
    print(f"Users in base  : {len({u for u, _, _, _ in base_raw}):,}")
    print(f"Items in base  : {len({i for _, i, _, _ in base_raw}):,}")
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


def to_csv_row(experiment_name, model_name, train_time, test_time, metrics_dict, baseline_metrics=None):
    row = {
        "experiment": experiment_name,
        "model": model_name,
        "train_time_s": train_time,
        "test_time_s": test_time,
    }
    row.update(metrics_dict)
    if baseline_metrics is not None:
        for metric_name, metric_value in metrics_dict.items():
            row[f"delta_{metric_name}"] = metric_value - baseline_metrics[metric_name]
    return row


def save_csv(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "train_time_s",
        "test_time_s",
        "AUC",
        "MAP",
        "NDCG@20",
        "Precision@20",
        "Recall@20",
        "delta_AUC",
        "delta_MAP",
        "delta_NDCG@20",
        "delta_Precision@20",
        "delta_Recall@20",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(normalized)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"online_ibpr_mejorado_partial_sweep_{timestamp}.txt")
    csv_path = os.path.join(RESULTS_DIR, f"online_ibpr_mejorado_partial_sweep_{timestamp}.csv")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print(f"Logging results to: {log_path}")
            print(f"CSV summary will be saved to: {csv_path}")
            print()

            data = load_positive_chrono_movielens_1m()
            base_raw, adapt_raw_all, test_raw_all = split_base_adapt_test(data)
            adapt_raw, test_raw = filter_to_base_known(base_raw, adapt_raw_all, test_raw_all)

            if len(adapt_raw) == 0 or len(test_raw) == 0:
                raise ValueError("Los segmentos adapt/test quedaron vacíos luego del filtrado a entidades conocidas.")

            print_segment_summary(base_raw, adapt_raw, test_raw)

            base_train_set, adapt_train_set, combined_train_set, test_set = build_datasets(
                base_raw, adapt_raw, test_raw
            )
            metrics = build_metrics()

            print("=" * 80)
            print("STEP 1 - TRAIN BASE IBPR")
            print("=" * 80)
            ibpr = IBPR(
                k=50,
                max_iter=20,
                learning_rate=0.01,
                lamda=0.001,
                batch_size=512,
                verbose=True,
                name="IBPR (base only)",
            )

            t0 = time.perf_counter()
            ibpr.fit(base_train_set)
            ibpr_train_time = time.perf_counter() - t0
            ibpr_metrics, ibpr_test_time = evaluate_ranking(ibpr, combined_train_set, test_set, metrics)
            print_result_block("Baseline IBPR trained on base segment", ibpr_train_time, ibpr_test_time, ibpr_metrics)

            recent_pairs = np.column_stack(
                (adapt_train_set.uir_tuple[0], adapt_train_set.uir_tuple[1])
            ).astype(np.int64)

            all_rows = [
                to_csv_row(
                    experiment_name="partial_sweep",
                    model_name="IBPR_base",
                    train_time=ibpr_train_time,
                    test_time=ibpr_test_time,
                    metrics_dict=ibpr_metrics,
                    baseline_metrics=None,
                )
            ]

            print("=" * 80)
            print("STEP 2 - SWEEP PARTIAL RETRAINING CONFIGURATIONS")
            print("=" * 80)

            for cfg in PARTIAL_CONFIGS:
                print(f"Running config: {cfg['label']}")

                online_partial = OnlineIBPRMejorado(
                    k=50,
                    max_iter=10,
                    learning_rate=0.01,
                    lamda=0.001,
                    batch_size=512,
                    init_params={"U": ibpr.U.copy(), "V": ibpr.V.copy()},
                    update_V=cfg["update_V"],
                    neg_sampling="uniform",
                    normalize=True,
                    loss_mode=cfg["loss_mode"],
                    verbose=True,
                    name=cfg["label"],
                )

                t0 = time.perf_counter()
                online_partial.partial_fit_recent(
                    recent_pairs=recent_pairs,
                    history_csr=combined_train_set.csr_matrix,
                    max_steps=cfg["max_steps"],
                    n_epochs=cfg["n_epochs"],
                )
                train_time = time.perf_counter() - t0
                model_metrics, test_time = evaluate_ranking(
                    online_partial, combined_train_set, test_set, metrics
                )

                print_result_block(cfg["label"], train_time, test_time, model_metrics)
                print("Delta vs baseline IBPR:")
                for metric_name, metric_value in model_metrics.items():
                    delta = metric_value - ibpr_metrics[metric_name]
                    print(f"  Δ{metric_name:10s}: {delta:+.4f}")
                print()

                all_rows.append(
                    to_csv_row(
                        experiment_name="partial_sweep",
                        model_name=cfg["name"],
                        train_time=train_time,
                        test_time=test_time,
                        metrics_dict=model_metrics,
                        baseline_metrics=ibpr_metrics,
                    )
                )

            save_csv(csv_path, all_rows)
            print("=" * 80)
            print("CSV SUMMARY SAVED")
            print("=" * 80)
            print(csv_path)


if __name__ == "__main__":
    main()
