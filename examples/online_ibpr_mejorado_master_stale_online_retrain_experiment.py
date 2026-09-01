import csv
import os
import sys
import time
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import datetime
from statistics import mean

import numpy as np
import cornac
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.utils.Tee import Tee
import torch


# ============================================================
# Configuration
# ============================================================
SEEDS = [42, 123, 2024, 777, 999]
MAP_SEED = SEEDS[0]

RATING_THRESHOLD = 3.0
VARIANT = "1M"   # change to "10M" if desired
BASE_FRAC = 0.60
STREAM_FRAC = 0.20
HOLDOUT_FRAC = 0.20
N_STREAM_CHUNKS = 4
TOP_K = 20
RESULTS_DIR = "results"

IBPR_CONFIG = {
    "k": 50,
    "max_iter": 20,
    "learning_rate": 0.01,
    "lamda": 0.001,
    "batch_size": 512,
    "verbose": True,
}

ONLINE_CONFIG = {
    "k": 50,
    "learning_rate": 0.01,
    "lamda": 0.001,
    "batch_size": 512,
    "update_V": False,
    "neg_sampling": "uniform",
    "normalize": True,
    "loss_mode": "cosine_bpr",
    "verbose": True,
}

PARTIAL_UPDATE_CONFIG = {
    "n_epochs": 1,
    "max_steps": None,
}


# ============================================================
# Data helpers
# ============================================================
def load_positive_chrono_movielens(variant=VARIANT, rating_threshold=RATING_THRESHOLD):
    data = movielens.load_feedback(fmt="UIRT", variant=variant)
    positive = [
        (str(u), str(i), 1.0, int(ts))
        for u, i, r, ts in data
        if float(r) >= rating_threshold
    ]
    positive.sort(key=lambda x: x[3])
    return positive



def split_base_stream_holdout(data, base_frac=BASE_FRAC, stream_frac=STREAM_FRAC):
    n = len(data)
    base_end = int(n * base_frac)
    stream_end = int(n * (base_frac + stream_frac))
    base_raw = data[:base_end]
    stream_raw = data[base_end:stream_end]
    holdout_raw = data[stream_end:]
    return base_raw, stream_raw, holdout_raw



def split_stream_into_chunks(stream_raw, n_chunks=N_STREAM_CHUNKS):
    raw_chunks = np.array_split(np.array(stream_raw, dtype=object), n_chunks)
    return [chunk.tolist() for chunk in raw_chunks if len(chunk) > 0]



def filter_to_base_known(base_raw, rows):
    known_users = {u for u, _, _, _ in base_raw}
    known_items = {i for _, i, _, _ in base_raw}
    return [row for row in rows if row[0] in known_users and row[1] in known_items]



def filter_chunks_to_base_known(base_raw, chunks):
    filtered = []
    for chunk in chunks:
        chunk_filtered = filter_to_base_known(base_raw, chunk)
        if chunk_filtered:
            filtered.append(chunk_filtered)
    return filtered



def build_global_maps(base_raw, stream_chunks, holdout_raw):
    all_rows = list(base_raw)
    for chunk in stream_chunks:
        all_rows.extend(chunk)
    all_rows.extend(holdout_raw)
    if not all_rows:
        raise ValueError("No hay datos luego del filtrado para construir el experimento.")

    ds = Dataset.build(all_rows, fmt="UIRT", seed=MAP_SEED)
    return ds.uid_map, ds.iid_map



def build_dataset(
    rows,
    uid_map,
    iid_map,
    exclude_unknowns=False,
    seed=MAP_SEED,
):
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

    return (
        OrderedDict(
            (metric.name, result)
            for metric, result in zip(metrics, avg_results)
        ),
        elapsed,
    )


def to_quality_row(stage, model_name, train_size, train_time_s, eval_time_s, metrics_dict):
    row = {
        "stage": stage,
        "model": model_name,
        "train_size": train_size,
        "cumulative_compute_time_s": train_time_s,
        "eval_time_s": eval_time_s,
    }
    row.update(metrics_dict)
    return row



def save_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ============================================================
# Main
# ============================================================
def run_single_seed(
    seed,
    base_raw,
    stream_chunks,
    holdout_raw,
    uid_map,
    iid_map,
    timestamp,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_path = os.path.join(
        RESULTS_DIR,
        f"master_stale_online_retrain_seed_{seed}_{timestamp}.txt"
    )

    quality_csv = os.path.join(
        RESULTS_DIR,
        f"master_stale_online_retrain_quality_seed_{seed}_{timestamp}.csv"
    )

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print(f"Logging results to: {log_path}")
            print(f"Quality CSV      : {quality_csv}")
            print()

            if not stream_chunks:
                raise ValueError("No quedaron chunks de stream luego del filtrado a entidades conocidas.")
            if not holdout_raw:
                raise ValueError("El holdout quedó vacío luego del filtrado a entidades conocidas.")

            print("=" * 80)
            print("STREAM / CHUNK EXPERIMENT SUMMARY")
            print("=" * 80)
            print(f"Variant                : {VARIANT}")
            print(f"Base rows              : {len(base_raw):,}")
            print(f"Stream rows (filtered) : {sum(len(c) for c in stream_chunks):,}")
            print(f"Holdout rows (filtered): {len(holdout_raw):,}")
            print(f"Stream chunks          : {len(stream_chunks)}")
            for idx, chunk in enumerate(stream_chunks, 1):
                print(f"  Chunk {idx:>2}: {len(chunk):,} interactions")
            print()

            base_train_set = build_dataset(base_raw, uid_map, iid_map, exclude_unknowns=False, seed=seed)
            metrics = build_metrics()

            print("=" * 80)
            print("STEP 1 - TRAIN BASE IBPR")
            print("=" * 80)
            ibpr = IBPR(name="IBPR_STALE", **IBPR_CONFIG)
            t0 = time.perf_counter()
            ibpr.fit(base_train_set)
            ibpr_train_time = time.perf_counter() - t0
            print(f"Base IBPR train time: {ibpr_train_time:.4f} s")
            print()

            print("=" * 80)
            print("STEP 2 - INITIALIZE ONLINE MODEL")
            print("=" * 80)
            online = OnlineIBPRMejorado(
                name="OnlineIBPRMejorado",
                init_params={"U": ibpr.U.copy(), "V": ibpr.V.copy()},
                max_iter=1,
                **ONLINE_CONFIG,
                seed=seed
            )
            # metadata del modelo base
            online.num_users = ibpr.num_users
            online.num_items = ibpr.num_items
            online.train_set = base_train_set
            print("Online model initialized from IBPR warm-start.")
            print()

            # At stage 1, Full Retrain is exactly the same base IBPR.
            # After each observed chunk, it will be retrained from scratch.
            full_retrain = ibpr

            current_rows = list(base_raw)

            cumulative_update_time = 0.0
            cumulative_retrain_time = 0.0

            quality_rows = []

            online_update_times = []
            full_retrain_times = []

            # Prequential-style evaluation: evaluate current models on next chunk, then update online model with that chunk.
            for chunk_idx, chunk_rows in enumerate(stream_chunks, 1):
                stage_name = f"chunk_{chunk_idx}_preupdate_eval"
                current_train_set = build_dataset(current_rows, uid_map, iid_map, exclude_unknowns=False, seed=seed)
                chunk_test_set = build_dataset(chunk_rows, uid_map, iid_map, exclude_unknowns=True, seed=seed)

                print("=" * 80)
                print(f"STAGE {chunk_idx} - EVALUATE CURRENT MODELS ON NEXT CHUNK")
                print("=" * 80)
                print(f"Current train size: {len(current_rows):,}")
                print(f"Chunk test size   : {len(chunk_rows):,}")

                ibpr_metrics, ibpr_eval_time = evaluate_ranking(ibpr, current_train_set, chunk_test_set, metrics)
                online_metrics, online_eval_time = evaluate_ranking(online, current_train_set, chunk_test_set, metrics)
                full_retrain_metrics, full_retrain_eval_time = evaluate_ranking(
                    full_retrain,
                    current_train_set,
                    chunk_test_set,
                    metrics,
                )

                quality_rows.append(
                    to_quality_row(stage_name, "IBPR_STALE", len(current_rows), ibpr_train_time, ibpr_eval_time, ibpr_metrics)
                )
                quality_rows.append(
                    to_quality_row(stage_name, "ONLINE_IBPR", len(current_rows), ibpr_train_time + cumulative_update_time, online_eval_time, online_metrics)
                )
                quality_rows.append(
                    to_quality_row(
                        stage_name,
                        "IBPR_FULL_RETRAIN",
                        len(current_rows),
                        ibpr_train_time + cumulative_retrain_time,
                        full_retrain_eval_time,
                        full_retrain_metrics,
                    )
                )

                print("IBPR base metrics")
                for k, v in ibpr_metrics.items():
                    print(f"  {k:12s}: {v:.4f}")
                print(f"  eval_time_s  : {ibpr_eval_time:.4f}")
                print()

                print("Online current metrics")
                for k, v in online_metrics.items():
                    print(f"  {k:12s}: {v:.4f}")
                print(f"  eval_time_s  : {online_eval_time:.4f}")
                print()

                print("Full Retrain current metrics")
                for k, v in full_retrain_metrics.items():
                    print(f"  {k:12s}: {v:.4f}")
                print(f"  eval_time_s  : {full_retrain_eval_time:.4f}")
                print()

                # Update only the Online model after the chunk arrives.
                post_chunk_rows = current_rows + list(chunk_rows)
                post_chunk_train_set = build_dataset(post_chunk_rows, uid_map, iid_map, exclude_unknowns=False, seed=seed)
                recent_pairs = np.asarray(
                    [[uid_map[u], iid_map[i]] for u, i, _, _ in chunk_rows],
                    dtype=np.int64,
                )

                print(f"Updating OnlineIBPR with chunk {chunk_idx}...")
                t0 = time.perf_counter()
                online.partial_fit_recent(
                    recent_pairs=recent_pairs,
                    history_csr=post_chunk_train_set.csr_matrix,
                    max_steps=PARTIAL_UPDATE_CONFIG["max_steps"],
                    n_epochs=PARTIAL_UPDATE_CONFIG["n_epochs"],
                )

                update_time = time.perf_counter() - t0
                online_update_times.append(update_time)
                cumulative_update_time += update_time

                print(f"Online update time for chunk {chunk_idx}: {update_time:.4f} s")
                print(
                    f"Online cumulative update time           : "
                    f"{cumulative_update_time:.4f} s"
                )
                print()

                # Full retrain from scratch using all data observed so far.
                print(f"Full retraining IBPR after chunk {chunk_idx}...")

                np.random.seed(seed)
                torch.manual_seed(seed)

                full_retrain = IBPR(
                    name="IBPR_FULL_RETRAIN",
                    **IBPR_CONFIG,
                )

                t0 = time.perf_counter()

                full_retrain.fit(post_chunk_train_set)

                retrain_time = time.perf_counter() - t0

                full_retrain_times.append(retrain_time)
                cumulative_retrain_time += retrain_time

                print(
                    f"Full retrain time for chunk {chunk_idx}: "
                    f"{retrain_time:.4f} s"
                )
                print(
                    f"Full retrain cumulative time           : "
                    f"{cumulative_retrain_time:.4f} s"
                )
                print()

                # Only now advance the available history.
                current_rows = post_chunk_rows

            print("=" * 80)
            print("FINAL HOLDOUT EVALUATION")
            print("=" * 80)
            final_train_set = build_dataset(current_rows, uid_map, iid_map, exclude_unknowns=False, seed=seed)
            holdout_test_set = build_dataset(holdout_raw, uid_map, iid_map, exclude_unknowns=True, seed=seed)

            ibpr_final_metrics, ibpr_final_eval_time = evaluate_ranking(ibpr, final_train_set, holdout_test_set, metrics)
            online_final_metrics, online_final_eval_time = evaluate_ranking(online, final_train_set, holdout_test_set, metrics)

            full_retrain_final_metrics, full_retrain_final_eval_time = evaluate_ranking(
                full_retrain,
                final_train_set,
                holdout_test_set,
                metrics,
            )

            quality_rows.append(
                to_quality_row("final_holdout", "IBPR_STALE", len(current_rows), ibpr_train_time, ibpr_final_eval_time, ibpr_final_metrics)
            )
            quality_rows.append(
                to_quality_row("final_holdout", "ONLINE_IBPR", len(current_rows), ibpr_train_time + cumulative_update_time, online_final_eval_time, online_final_metrics)
            )

            quality_rows.append(
                to_quality_row(
                    "final_holdout",
                    "IBPR_FULL_RETRAIN",
                    len(current_rows),
                    ibpr_train_time + cumulative_retrain_time,
                    full_retrain_final_eval_time,
                    full_retrain_final_metrics,
                )
            )

            print("Final holdout - IBPR base")
            for k, v in ibpr_final_metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"  eval_time_s  : {ibpr_final_eval_time:.4f}")
            print()

            print("Final holdout - OnlineIBPR")
            for k, v in online_final_metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"  eval_time_s  : {online_final_eval_time:.4f}")
            print()

            print("Final holdout - IBPR Full Retrain")
            for k, v in full_retrain_final_metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"  eval_time_s  : {full_retrain_final_eval_time:.4f}")
            print()

            print("=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Base train time                 : {ibpr_train_time:.4f} s")
            print(f"Online cumulative update time   : {cumulative_update_time:.4f} s")
            print(f"Online avg chunk update time    : {mean(online_update_times):.4f} s")
            print(
                f"Full retrain cumulative time    : "
                f"{cumulative_retrain_time:.4f} s"
            )
            print(
                f"Full retrain avg time           : "
                f"{mean(full_retrain_times):.4f} s"
            )
            print(
                f"Final holdout ΔRecall@{TOP_K} (Online - Stale): "
                f"{online_final_metrics[f'Recall@{TOP_K}'] - ibpr_final_metrics[f'Recall@{TOP_K}']:.4f}"
            )
            print()

            print(
                f"Final holdout ΔRecall@{TOP_K} (Full Retrain - Stale): "
                f"{full_retrain_final_metrics[f'Recall@{TOP_K}'] - ibpr_final_metrics[f'Recall@{TOP_K}']:.4f}"
            )

            print(
                f"Final holdout ΔRecall@{TOP_K} (Online - Full Retrain): "
                f"{online_final_metrics[f'Recall@{TOP_K}'] - full_retrain_final_metrics[f'Recall@{TOP_K}']:.4f}"
            )

            quality_fields = [
                "stage",
                "model",
                "train_size",
                "cumulative_compute_time_s",
                "eval_time_s",
                "AUC",
                "MAP",
                f"NDCG@{TOP_K}",
                f"Precision@{TOP_K}",
                f"Recall@{TOP_K}",
            ]

            save_csv(quality_csv, quality_fields, quality_rows)
            return quality_rows

def summarize_multiseed(final_rows):
    metrics = [
        "AUC",
        "MAP",
        f"NDCG@{TOP_K}",
        f"Precision@{TOP_K}",
        f"Recall@{TOP_K}",
        "cumulative_compute_time_s",
    ]

    models = [
        "IBPR_STALE",
        "ONLINE_IBPR",
        "IBPR_FULL_RETRAIN",
    ]

    summary_rows = []

    for model in models:
        model_rows = [
            row for row in final_rows
            if row["model"] == model
        ]

        summary = {
            "comparison": model,
        }

        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in model_rows],
                dtype=np.float64,
            )

            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))

        summary_rows.append(summary)

    comparisons = [
        (
            "ONLINE_MINUS_STALE",
            "ONLINE_IBPR",
            "IBPR_STALE",
        ),
        (
            "ONLINE_MINUS_FULL_RETRAIN",
            "ONLINE_IBPR",
            "IBPR_FULL_RETRAIN",
        ),
        (
            "FULL_RETRAIN_MINUS_STALE",
            "IBPR_FULL_RETRAIN",
            "IBPR_STALE",
        ),
    ]

    seeds = sorted({row["seed"] for row in final_rows})

    for comparison_name, model_a, model_b in comparisons:
        summary = {
            "comparison": comparison_name,
        }

        for metric in metrics:
            deltas = []

            for seed in seeds:
                row_a = next(
                    row
                    for row in final_rows
                    if row["seed"] == seed
                    and row["model"] == model_a
                )

                row_b = next(
                    row
                    for row in final_rows
                    if row["seed"] == seed
                    and row["model"] == model_b
                )

                deltas.append(
                    float(row_a[metric]) - float(row_b[metric])
                )

            values = np.asarray(deltas, dtype=np.float64)

            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))

        summary_rows.append(summary)

    return summary_rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = load_positive_chrono_movielens()

    base_raw, stream_raw_all, holdout_raw_all = (
        split_base_stream_holdout(data)
    )

    stream_chunks_all = split_stream_into_chunks(
        stream_raw_all
    )

    stream_chunks = filter_chunks_to_base_known(
        base_raw,
        stream_chunks_all,
    )

    holdout_raw = filter_to_base_known(
        base_raw,
        holdout_raw_all,
    )

    if not stream_chunks:
        raise ValueError(
            "No quedaron chunks luego del filtrado."
        )

    if not holdout_raw:
        raise ValueError(
            "El holdout quedó vacío."
        )

    uid_map, iid_map = build_global_maps(
        base_raw,
        stream_chunks,
        holdout_raw,
    )

    all_final_rows = []

    for seed in SEEDS:
        print()
        print("#" * 80)
        print(f"RUNNING SEED {seed}")
        print("#" * 80)

        quality_rows = run_single_seed(
            seed=seed,
            base_raw=base_raw,
            stream_chunks=stream_chunks,
            holdout_raw=holdout_raw,
            uid_map=uid_map,
            iid_map=iid_map,
            timestamp=timestamp,
        )

        for row in quality_rows:
            if row["stage"] == "final_holdout":
                final_row = dict(row)
                final_row["seed"] = seed
                all_final_rows.append(final_row)

    summary_rows = summarize_multiseed(
        all_final_rows
    )

    raw_csv = os.path.join(
        RESULTS_DIR,
        f"master_multiseed_final_raw_{timestamp}.csv",
    )

    raw_fields = [
        "seed",
        "stage",
        "model",
        "train_size",
        "cumulative_compute_time_s",
        "eval_time_s",
        "AUC",
        "MAP",
        f"NDCG@{TOP_K}",
        f"Precision@{TOP_K}",
        f"Recall@{TOP_K}",
    ]

    save_csv(
        raw_csv,
        raw_fields,
        all_final_rows,
    )

    summary_csv = os.path.join(
        RESULTS_DIR,
        f"master_multiseed_summary_{timestamp}.csv",
    )

    summary_fields = ["comparison"]

    for metric in [
        "AUC",
        "MAP",
        f"NDCG@{TOP_K}",
        f"Precision@{TOP_K}",
        f"Recall@{TOP_K}",
        "cumulative_compute_time_s",
    ]:
        summary_fields.append(f"{metric}_mean")
        summary_fields.append(f"{metric}_std")

    save_csv(
        summary_csv,
        summary_fields,
        summary_rows,
    )

    print()
    print("=" * 80)
    print("MULTI-SEED EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Raw results : {raw_csv}")
    print(f"Summary     : {summary_csv}")

if __name__ == "__main__":
    main()
