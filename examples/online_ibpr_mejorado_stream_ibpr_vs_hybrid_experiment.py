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

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


# ============================================================
# Configuration
# ============================================================
SEED = 42
RATING_THRESHOLD = 3.0
VARIANT = "1M"   # change to "10M" if desired
BASE_FRAC = 0.60
STREAM_FRAC = 0.20
HOLDOUT_FRAC = 0.20
N_STREAM_CHUNKS = 4
TOP_K = 20
CANDIDATE_MULTIPLIER = 10
MAX_QUERY_USERS_PER_STAGE = 300
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

    ds = Dataset.build(all_rows, fmt="UIRT", seed=SEED)
    return ds.uid_map, ds.iid_map



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


# ============================================================
# Retrieval helpers
# ============================================================
def build_retriever(item_vectors):
    item_vectors = np.asarray(item_vectors, dtype=np.float32)
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    item_vectors_norm = item_vectors / norms

    if NearestNeighbors is None:
        return {"backend": "exact_cosine", "item_vectors_norm": item_vectors_norm}

    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(item_vectors_norm)
    return {
        "backend": "sklearn_knn",
        "nn": nn,
        "item_vectors_norm": item_vectors_norm,
    }



def _normalize_query(user_vector):
    user_vec = np.asarray(user_vector, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(user_vec, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return user_vec / norm



def retrieve_candidates(retriever, user_vector, n_candidates, seen_items=None):
    seen_items = seen_items or set()
    user_vec = _normalize_query(user_vector)
    n_items = retriever["item_vectors_norm"].shape[0]
    fetch_n = min(n_items, max(n_candidates + len(seen_items), n_candidates))

    if retriever["backend"] == "sklearn_knn":
        distances, indices = retriever["nn"].kneighbors(user_vec, n_neighbors=fetch_n)
        candidate_ids = indices[0].astype(np.int64)
        candidate_scores = (1.0 - distances[0]).astype(np.float32)
    else:
        scores = retriever["item_vectors_norm"].dot(user_vec.ravel())
        top_idx = np.argpartition(scores, -fetch_n)[-fetch_n:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        candidate_ids = top_idx.astype(np.int64)
        candidate_scores = scores[top_idx].astype(np.float32)

    if seen_items:
        mask = np.array([int(i) not in seen_items for i in candidate_ids], dtype=bool)
        candidate_ids = candidate_ids[mask]
        candidate_scores = candidate_scores[mask]

    if len(candidate_ids) > n_candidates:
        candidate_ids = candidate_ids[:n_candidates]
        candidate_scores = candidate_scores[:n_candidates]

    return candidate_ids, candidate_scores



def topk_indexed(model, retriever, user_idx, seen_items, k):
    candidate_ids, _ = retrieve_candidates(
        retriever=retriever,
        user_vector=model.U[user_idx],
        n_candidates=k,
        seen_items=seen_items,
    )
    return candidate_ids



def topk_hybrid_rerank(ibpr_model, online_model, retriever, user_idx, seen_items, k, n_candidates):
    candidate_ids, _ = retrieve_candidates(
        retriever=retriever,
        user_vector=online_model.U[user_idx],#ibpr_model.U[user_idx],
        n_candidates=n_candidates,
        seen_items=seen_items,
    )
    if len(candidate_ids) == 0:
        return np.array([], dtype=np.int64)

    candidate_vecs = online_model.V[candidate_ids]
    user_vec = online_model.U[user_idx]
    scores = candidate_vecs.dot(user_vec)
    final_k = min(k, len(candidate_ids))
    top_local = np.argpartition(scores, -final_k)[-final_k:]
    top_local = top_local[np.argsort(-scores[top_local])]
    return candidate_ids[top_local]



def make_seen_lookup(train_set):
    seen = {}
    user_indices = np.asarray(train_set.uir_tuple[0], dtype=np.int64)
    item_indices = np.asarray(train_set.uir_tuple[1], dtype=np.int64)
    for uidx, iidx in zip(user_indices, item_indices):
        seen.setdefault(int(uidx), set()).add(int(iidx))
    return seen



def make_ground_truth_lookup(test_set):
    gt = {}
    user_indices = np.asarray(test_set.uir_tuple[0], dtype=np.int64)
    item_indices = np.asarray(test_set.uir_tuple[1], dtype=np.int64)
    for uidx, iidx in zip(user_indices, item_indices):
        gt.setdefault(int(uidx), set()).add(int(iidx))
    return gt



def get_benchmark_users(test_set, max_users=MAX_QUERY_USERS_PER_STAGE):
    users = sorted({int(u) for u in np.asarray(test_set.uir_tuple[0], dtype=np.int64)})
    return users[:max_users]



def recall_at_k(recommended, relevant):
    if not relevant:
        return np.nan
    return len(set(map(int, recommended)) & set(map(int, relevant))) / len(relevant)



def latency_summary(latencies_ms):
    arr = np.asarray(latencies_ms, dtype=np.float64)
    total_s = arr.sum() / 1000.0
    return {
        "queries": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "qps": float(arr.size / total_s) if total_s > 0 else float("inf"),
    }



def benchmark_retrieval(method_name, users, ranking_fn, seen_lookup, gt_lookup):
    latencies_ms = []
    recalls = []

    for user_idx in users:
        t0 = time.perf_counter()
        recs = ranking_fn(user_idx, seen_lookup.get(user_idx, set()))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        relevant = gt_lookup.get(user_idx, set())
        if relevant:
            recalls.append(recall_at_k(recs, relevant))

    out = latency_summary(latencies_ms)
    out["method"] = method_name
    out["recall@k"] = float(np.nanmean(recalls)) if recalls else float("nan")
    return out


# ============================================================
# Evaluation helpers
# ============================================================
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



def to_quality_row(stage, model_name, train_size, train_time_s, eval_time_s, metrics_dict):
    row = {
        "stage": stage,
        "model": model_name,
        "train_size": train_size,
        "train_or_update_time_s": train_time_s,
        "eval_time_s": eval_time_s,
    }
    row.update(metrics_dict)
    return row



def to_latency_row(stage, method_name, train_size, latency_dict):
    row = {
        "stage": stage,
        "method": method_name,
        "train_size": train_size,
    }
    row.update(latency_dict)
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
def main():
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"stream_ibpr_vs_hybrid_{timestamp}.txt")
    quality_csv = os.path.join(RESULTS_DIR, f"stream_ibpr_vs_hybrid_quality_{timestamp}.csv")
    latency_csv = os.path.join(RESULTS_DIR, f"stream_ibpr_vs_hybrid_latency_{timestamp}.csv")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print(f"Logging results to: {log_path}")
            print(f"Quality CSV      : {quality_csv}")
            print(f"Latency CSV      : {latency_csv}")
            print()

            data = load_positive_chrono_movielens()
            base_raw, stream_raw_all, holdout_raw_all = split_base_stream_holdout(data)
            stream_chunks_all = split_stream_into_chunks(stream_raw_all)
            stream_chunks = filter_chunks_to_base_known(base_raw, stream_chunks_all)
            holdout_raw = filter_to_base_known(base_raw, holdout_raw_all)

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

            uid_map, iid_map = build_global_maps(base_raw, stream_chunks, holdout_raw)
            base_train_set = build_dataset(base_raw, uid_map, iid_map, exclude_unknowns=False)
            metrics = build_metrics()

            print("=" * 80)
            print("STEP 1 - TRAIN BASE IBPR")
            print("=" * 80)
            ibpr = IBPR(name="IBPR_base", **IBPR_CONFIG)
            t0 = time.perf_counter()
            ibpr.fit(base_train_set)
            ibpr_train_time = time.perf_counter() - t0
            print(f"Base IBPR train time: {ibpr_train_time:.4f} s")
            print()

            print("=" * 80)
            print("STEP 2 - INITIALIZE HYBRID ONLINE MODEL")
            print("=" * 80)
            online = OnlineIBPRMejorado(
                name="IBPR_plus_OnlineIBPRMejorado",
                init_params={"U": ibpr.U.copy(), "V": ibpr.V.copy()},
                max_iter=1,
                **ONLINE_CONFIG,
            )
            # metadata del modelo base
            online.num_users = ibpr.num_users
            online.num_items = ibpr.num_items
            online.train_set = base_train_set
            print("Hybrid model initialized from IBPR warm-start.")
            print()

            retriever = build_retriever(ibpr.V)
            print(f"Retriever backend: {retriever['backend']}")
            print()

            current_rows = list(base_raw)
            cumulative_update_time = 0.0
            quality_rows = []
            latency_rows = []
            hybrid_update_times = []

            # Prequential-style evaluation: evaluate current models on next chunk, then update hybrid with that chunk.
            for chunk_idx, chunk_rows in enumerate(stream_chunks, 1):
                stage_name = f"chunk_{chunk_idx}_preupdate_eval"
                current_train_set = build_dataset(current_rows, uid_map, iid_map, exclude_unknowns=False)
                chunk_test_set = build_dataset(chunk_rows, uid_map, iid_map, exclude_unknowns=True)

                print("=" * 80)
                print(f"STAGE {chunk_idx} - EVALUATE CURRENT MODELS ON NEXT CHUNK")
                print("=" * 80)
                print(f"Current train size: {len(current_rows):,}")
                print(f"Chunk test size   : {len(chunk_rows):,}")

                ibpr_metrics, ibpr_eval_time = evaluate_ranking(ibpr, current_train_set, chunk_test_set, metrics)
                online_metrics, online_eval_time = evaluate_ranking(online, current_train_set, chunk_test_set, metrics)

                quality_rows.append(
                    to_quality_row(stage_name, "IBPR_base", len(current_rows), ibpr_train_time, ibpr_eval_time, ibpr_metrics)
                )
                quality_rows.append(
                    to_quality_row(stage_name, "Hybrid_current", len(current_rows), cumulative_update_time, online_eval_time, online_metrics)
                )

                print("IBPR base metrics")
                for k, v in ibpr_metrics.items():
                    print(f"  {k:12s}: {v:.4f}")
                print(f"  eval_time_s  : {ibpr_eval_time:.4f}")
                print()

                print("Hybrid current metrics")
                for k, v in online_metrics.items():
                    print(f"  {k:12s}: {v:.4f}")
                print(f"  eval_time_s  : {online_eval_time:.4f}")
                print()

                seen_lookup = make_seen_lookup(current_train_set)
                gt_lookup = make_ground_truth_lookup(chunk_test_set)
                users = get_benchmark_users(chunk_test_set)
                n_candidates = min(len(ibpr.V), max(TOP_K, TOP_K * CANDIDATE_MULTIPLIER))

                ibpr_latency = benchmark_retrieval(
                    method_name="IBPR_indexed_topk",
                    users=users,
                    ranking_fn=lambda u, seen: topk_indexed(
                        model=ibpr,
                        retriever=retriever,
                        user_idx=u,
                        seen_items=seen,
                        k=TOP_K,
                    ),
                    seen_lookup=seen_lookup,
                    gt_lookup=gt_lookup,
                )
                hybrid_latency = benchmark_retrieval(
                    method_name="IBPR_indexed_plus_online_rerank",
                    users=users,
                    ranking_fn=lambda u, seen: topk_hybrid_rerank(
                        ibpr_model=ibpr,
                        online_model=online,
                        retriever=retriever,
                        user_idx=u,
                        seen_items=seen,
                        k=TOP_K,
                        n_candidates=n_candidates,
                    ),
                    seen_lookup=seen_lookup,
                    gt_lookup=gt_lookup,
                )

                latency_rows.append(to_latency_row(stage_name, ibpr_latency["method"], len(current_rows), ibpr_latency))
                latency_rows.append(to_latency_row(stage_name, hybrid_latency["method"], len(current_rows), hybrid_latency))

                print("Latency benchmark")
                for row in (ibpr_latency, hybrid_latency):
                    print(
                        f"  {row['method']}: mean_ms={row['mean_ms']:.4f}, "
                        f"p95_ms={row['p95_ms']:.4f}, qps={row['qps']:.2f}, recall@k={row['recall@k']:.4f}"
                    )
                print()

                # Update only the hybrid after the chunk arrives.
                post_chunk_rows = current_rows + list(chunk_rows)
                post_chunk_train_set = build_dataset(post_chunk_rows, uid_map, iid_map, exclude_unknowns=False)
                recent_pairs = np.asarray(
                    [[uid_map[u], iid_map[i]] for u, i, _, _ in chunk_rows],
                    dtype=np.int64,
                )

                print(f"Updating hybrid with chunk {chunk_idx}...")
                t0 = time.perf_counter()
                online.partial_fit_recent(
                    recent_pairs=recent_pairs,
                    history_csr=post_chunk_train_set.csr_matrix,
                    max_steps=PARTIAL_UPDATE_CONFIG["max_steps"],
                    n_epochs=PARTIAL_UPDATE_CONFIG["n_epochs"],
                )
                update_time = time.perf_counter() - t0
                hybrid_update_times.append(update_time)
                cumulative_update_time += update_time
                current_rows = post_chunk_rows

                print(f"Hybrid update time for chunk {chunk_idx}: {update_time:.4f} s")
                print(f"Hybrid cumulative update time           : {cumulative_update_time:.4f} s")
                print()

            print("=" * 80)
            print("FINAL HOLDOUT EVALUATION")
            print("=" * 80)
            final_train_set = build_dataset(current_rows, uid_map, iid_map, exclude_unknowns=False)
            holdout_test_set = build_dataset(holdout_raw, uid_map, iid_map, exclude_unknowns=True)

            ibpr_final_metrics, ibpr_final_eval_time = evaluate_ranking(ibpr, final_train_set, holdout_test_set, metrics)
            hybrid_final_metrics, hybrid_final_eval_time = evaluate_ranking(online, final_train_set, holdout_test_set, metrics)

            quality_rows.append(
                to_quality_row("final_holdout", "IBPR_base", len(current_rows), ibpr_train_time, ibpr_final_eval_time, ibpr_final_metrics)
            )
            quality_rows.append(
                to_quality_row("final_holdout", "Hybrid_current", len(current_rows), cumulative_update_time, hybrid_final_eval_time, hybrid_final_metrics)
            )

            print("Final holdout - IBPR base")
            for k, v in ibpr_final_metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"  eval_time_s  : {ibpr_final_eval_time:.4f}")
            print()

            print("Final holdout - Hybrid")
            for k, v in hybrid_final_metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"  eval_time_s  : {hybrid_final_eval_time:.4f}")
            print()

            seen_lookup = make_seen_lookup(final_train_set)
            gt_lookup = make_ground_truth_lookup(holdout_test_set)
            users = get_benchmark_users(holdout_test_set)
            n_candidates = min(len(ibpr.V), max(TOP_K, TOP_K * CANDIDATE_MULTIPLIER))

            ibpr_latency = benchmark_retrieval(
                method_name="IBPR_indexed_topk",
                users=users,
                ranking_fn=lambda u, seen: topk_indexed(
                    model=ibpr,
                    retriever=retriever,
                    user_idx=u,
                    seen_items=seen,
                    k=TOP_K,
                ),
                seen_lookup=seen_lookup,
                gt_lookup=gt_lookup,
            )
            hybrid_latency = benchmark_retrieval(
                method_name="IBPR_indexed_plus_online_rerank",
                users=users,
                ranking_fn=lambda u, seen: topk_hybrid_rerank(
                    ibpr_model=ibpr,
                    online_model=online,
                    retriever=retriever,
                    user_idx=u,
                    seen_items=seen,
                    k=TOP_K,
                    n_candidates=n_candidates,
                ),
                seen_lookup=seen_lookup,
                gt_lookup=gt_lookup,
            )

            latency_rows.append(to_latency_row("final_holdout", ibpr_latency["method"], len(current_rows), ibpr_latency))
            latency_rows.append(to_latency_row("final_holdout", hybrid_latency["method"], len(current_rows), hybrid_latency))

            print("Final holdout latency")
            for row in (ibpr_latency, hybrid_latency):
                print(
                    f"  {row['method']}: mean_ms={row['mean_ms']:.4f}, "
                    f"p95_ms={row['p95_ms']:.4f}, qps={row['qps']:.2f}, recall@k={row['recall@k']:.4f}"
                )
            print()

            print("=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Base train time                 : {ibpr_train_time:.4f} s")
            print(f"Hybrid cumulative update time   : {cumulative_update_time:.4f} s")
            print(f"Hybrid avg chunk update time    : {mean(hybrid_update_times):.4f} s")
            print(
                f"Final holdout ΔRecall@{TOP_K} (Hybrid - Base): "
                f"{hybrid_final_metrics[f'Recall@{TOP_K}'] - ibpr_final_metrics[f'Recall@{TOP_K}']:.4f}"
            )
            print(
                f"Final holdout latency overhead mean_ms (Hybrid - Base): "
                f"{hybrid_latency['mean_ms'] - ibpr_latency['mean_ms']:.4f}"
            )
            print()

            quality_fields = [
                "stage",
                "model",
                "train_size",
                "train_or_update_time_s",
                "eval_time_s",
                "AUC",
                "MAP",
                f"NDCG@{TOP_K}",
                f"Precision@{TOP_K}",
                f"Recall@{TOP_K}",
            ]
            latency_fields = [
                "stage",
                "method",
                "train_size",
                "queries",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "qps",
                "recall@k",
            ]
            save_csv(quality_csv, quality_fields, quality_rows)
            save_csv(latency_csv, latency_fields, latency_rows)


if __name__ == "__main__":
    main()
