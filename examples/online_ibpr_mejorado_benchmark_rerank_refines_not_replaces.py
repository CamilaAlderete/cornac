import csv
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.utils.Tee import Tee

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


SEED = 42
RATING_THRESHOLD = 3.0
VARIANT = "1M"
BASE_FRAC = 0.70
ADAPT_FRAC = 0.10
TEST_FRAC = 0.20
TOP_K = 20
CANDIDATE_MULTIPLIERS = [1, 2, 5, 10, 20]
MAX_QUERY_USERS = 300
RESULTS_DIR = "results"

EXPERIMENT_SCOPE = "warm_start_known_entities"
COLD_START_INCLUDED = False

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



def split_base_adapt_test(data, base_frac=BASE_FRAC, adapt_frac=ADAPT_FRAC):
    n = len(data)
    base_end = int(n * base_frac)
    adapt_end = int(n * (base_frac + adapt_frac))
    base_raw = data[:base_end]
    adapt_raw = data[base_end:adapt_end]
    test_raw = data[adapt_end:]
    return base_raw, adapt_raw, test_raw



def filter_to_base_known(base_raw, rows):
    known_users = {u for u, _, _, _ in base_raw}
    known_items = {i for _, i, _, _ in base_raw}
    return [row for row in rows if row[0] in known_users and row[1] in known_items]



def build_datasets(base_raw, adapt_raw, test_raw):
    all_rows = base_raw + adapt_raw + test_raw
    if not all_rows:
        raise ValueError("No hay datos luego del filtrado para construir el experimento.")

    ds = Dataset.build(all_rows, fmt="UIRT", seed=SEED)
    uid_map = ds.uid_map
    iid_map = ds.iid_map

    base_train_set = Dataset.build(
        base_raw,
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
    return base_train_set, combined_train_set, test_set, uid_map, iid_map


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
    return {"backend": "sklearn_knn", "nn": nn, "item_vectors_norm": item_vectors_norm}



def _normalize_query(user_vector):
    user_vec = np.asarray(user_vector, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(user_vec, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return user_vec / norm



def retrieve_candidates(retriever, user_vector, n_candidates, seen_items=None):
    seen_items = seen_items or set()
    user_vec = _normalize_query(user_vector)
    n_items = retriever["item_vectors_norm"].shape[0]
    #fetch_n = min(n_items, max(n_candidates + len(seen_items), n_candidates))
    extra_margin = max(50, n_candidates)
    fetch_n = min(n_items, max(n_candidates + len(seen_items) + extra_margin, n_candidates * 2))

    if retriever["backend"] == "sklearn_knn":
        distances, indices = retriever["nn"].kneighbors(user_vec, n_neighbors=fetch_n)
        candidate_ids = indices[0].astype(np.int64)
    else:
        scores = retriever["item_vectors_norm"].dot(user_vec.ravel())
        top_idx = np.argpartition(scores, -fetch_n)[-fetch_n:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        candidate_ids = top_idx.astype(np.int64)

    if seen_items:
        candidate_ids = np.array([i for i in candidate_ids if int(i) not in seen_items], dtype=np.int64)

    if len(candidate_ids) > n_candidates:
        candidate_ids = candidate_ids[:n_candidates]
    return candidate_ids



def topk_full_scores(model, user_idx, seen_items, k):
    scores = np.asarray(model.score(user_idx), dtype=np.float32)
    if seen_items:
        scores[list(seen_items)] = -np.inf

    if k >= len(scores):
        ranked = np.argsort(-scores)
        ranked = ranked[np.isfinite(scores[ranked])]
        return ranked[:k]

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx[np.isfinite(scores[top_idx])][:k]



def rerank_candidates(online_model, user_idx, candidate_ids, k):
    if len(candidate_ids) == 0:
        return np.array([], dtype=np.int64)
    scores = online_model.V[candidate_ids].dot(online_model.U[user_idx])
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



"""def get_benchmark_users(test_set, max_users=MAX_QUERY_USERS):
    users = sorted({int(u) for u in np.asarray(test_set.uir_tuple[0], dtype=np.int64)})
    return users[:max_users]"""

def get_benchmark_users(test_set, max_users=MAX_QUERY_USERS, seed=SEED):
    users = sorted({int(u) for u in np.asarray(test_set.uir_tuple[0], dtype=np.int64)})

    if len(users) <= max_users:
        return users

    rng = np.random.default_rng(seed)
    sampled = rng.choice(users, size=max_users, replace=False)
    return sorted(sampled.tolist())


def recall_at_k(recommended, relevant):
    if not relevant:
        return np.nan
    return len(set(map(int, recommended)) & set(map(int, relevant))) / len(relevant)



def overlap_ratio(left, right, denom):
    if denom == 0:
        return np.nan
    return len(set(map(int, left)) & set(map(int, right))) / denom


# ============================================================
# Main benchmark
# ============================================================
def main():
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"rerank_refines_not_replaces_{timestamp}.txt")
    csv_path = os.path.join(RESULTS_DIR, f"rerank_refines_not_replaces_{timestamp}.csv")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print(f"Logging results to: {log_path}")
            print(f"CSV summary will be saved to: {csv_path}")
            print()

            data = load_positive_chrono_movielens()
            base_raw, adapt_raw_all, test_raw_all = split_base_adapt_test(data)
            adapt_raw = filter_to_base_known(base_raw, adapt_raw_all)
            test_raw = filter_to_base_known(base_raw, test_raw_all)

            if not adapt_raw:
                raise ValueError("Adapt quedó vacío luego del filtrado a entidades conocidas.")
            if not test_raw:
                raise ValueError("Test quedó vacío luego del filtrado a entidades conocidas.")

            print("=" * 80)
            print("RERANK PREMISE BENCHMARK")
            print("=" * 80)
            print(f"Variant       : {VARIANT}")
            print(f"Base rows     : {len(base_raw):,}")
            print(f"Adapt rows    : {len(adapt_raw):,}")
            print(f"Test rows     : {len(test_raw):,}")
            print(f"TOP_K         : {TOP_K}")
            print(f"Multipliers   : {CANDIDATE_MULTIPLIERS}")
            print(f"Scope         : {EXPERIMENT_SCOPE}")
            print(f"Cold-start    : {COLD_START_INCLUDED}")
            print("Note          : adapt/test contain only users and items already seen in base")
            print()

            base_train_set, combined_train_set, test_set, uid_map, iid_map = build_datasets(base_raw, adapt_raw, test_raw)

            print("Training base IBPR...")
            ibpr = IBPR(name="IBPR_base", **IBPR_CONFIG)
            t0 = time.perf_counter()
            ibpr.fit(base_train_set)
            ibpr_train_time = time.perf_counter() - t0
            print(f"IBPR train time: {ibpr_train_time:.4f} s")
            print()

            print("Warm-starting and partially adapting OnlineIBPRMejorado...")
            online = OnlineIBPRMejorado(
                name="IBPR_plus_OnlineIBPRMejorado",
                init_params={"U": ibpr.U.copy(), "V": ibpr.V.copy()},
                max_iter=1,
                **ONLINE_CONFIG,
            )
            recent_pairs = np.asarray([[uid_map[u], iid_map[i]] for u, i, _, _ in adapt_raw], dtype=np.int64)
            t0 = time.perf_counter()
            online.partial_fit_recent(
                recent_pairs=recent_pairs,
                history_csr=combined_train_set.csr_matrix,
                max_steps=PARTIAL_UPDATE_CONFIG["max_steps"],
                n_epochs=PARTIAL_UPDATE_CONFIG["n_epochs"],
            )
            online_update_time = time.perf_counter() - t0
            print(f"Online partial update time: {online_update_time:.4f} s")
            print()

            retriever = build_retriever(ibpr.V)
            print(f"Retriever backend: {retriever['backend']}")
            print()

            seen_lookup = make_seen_lookup(combined_train_set)
            gt_lookup = make_ground_truth_lookup(test_set)
            users = get_benchmark_users(test_set)
            print(f"Users benchmarked: {len(users)}")
            print()

            rows = []

            # Exhaustive online acts as the reference 'ideal' final ranking for the adapted model.
            exhaustive_latencies = []
            exhaustive_recalls = []
            exhaustive_topk_by_user = {}
            for user_idx in users:
                seen = seen_lookup.get(user_idx, set())
                t0 = time.perf_counter()
                recs = topk_full_scores(online, user_idx, seen, TOP_K)
                exhaustive_latencies.append((time.perf_counter() - t0) * 1000.0)
                exhaustive_topk_by_user[user_idx] = recs
                relevant = gt_lookup.get(user_idx, set())
                if relevant:
                    exhaustive_recalls.append(recall_at_k(recs, relevant))

            exhaustive_summary = {
                "method": "Online_exhaustive_oracle",#"Online_exhaustive_topk",
                "role": "quality_oracle",
                "scope": EXPERIMENT_SCOPE,
                "cold_start_included": COLD_START_INCLUDED,
                "candidates": len(online.V),
                "queries": len(users),
                "mean_ms": float(np.mean(exhaustive_latencies)),
                "median_ms": float(np.median(exhaustive_latencies)),
                "p95_ms": float(np.percentile(exhaustive_latencies, 95)),
                "recall@k": float(np.nanmean(exhaustive_recalls)) if exhaustive_recalls else float("nan"),
                "agreement_with_online_exhaustive@k": 1.0,
                "candidate_coverage_of_online_exhaustive@k": 1.0,
                "subset_of_candidate_pool_rate": 1.0,
            }
            rows.append(exhaustive_summary)

            #print("Reference method: Online exhaustive top-k")
            print("Quality oracle: Online exhaustive ranking over full catalog")

            print(
                f"  mean_ms={exhaustive_summary['mean_ms']:.4f}, p95_ms={exhaustive_summary['p95_ms']:.4f}, "
                f"recall@k={exhaustive_summary['recall@k']:.4f}"
            )
            print()

            # Also include pure IBPR indexed top-k for context.
            ibpr_latencies = []
            ibpr_recalls = []
            ibpr_agreements = []
            for user_idx in users:
                seen = seen_lookup.get(user_idx, set())
                t0 = time.perf_counter()
                recs = retrieve_candidates(retriever, ibpr.U[user_idx], TOP_K, seen)
                recs = recs[:TOP_K]
                ibpr_latencies.append((time.perf_counter() - t0) * 1000.0)
                relevant = gt_lookup.get(user_idx, set())
                if relevant:
                    ibpr_recalls.append(recall_at_k(recs, relevant))
                ibpr_agreements.append(overlap_ratio(recs, exhaustive_topk_by_user[user_idx], TOP_K))

            ibpr_summary = {
                "method": "IBPR_indexed_topk",
                "role": "serving_baseline",
                "scope": EXPERIMENT_SCOPE,
                "cold_start_included": COLD_START_INCLUDED,
                "candidates": TOP_K,
                "queries": len(users),
                "mean_ms": float(np.mean(ibpr_latencies)),
                "median_ms": float(np.median(ibpr_latencies)),
                "p95_ms": float(np.percentile(ibpr_latencies, 95)),
                "recall@k": float(np.nanmean(ibpr_recalls)) if ibpr_recalls else float("nan"),
                "agreement_with_online_exhaustive@k": float(np.nanmean(ibpr_agreements)),
                "candidate_coverage_of_online_exhaustive@k": float(np.nanmean(ibpr_agreements)),
                "subset_of_candidate_pool_rate": 1.0,
            }
            rows.append(ibpr_summary)

            #print("Context method: IBPR indexed top-k")
            print("Serving baseline: IBPR indexed top-k")

            print(
                f"  mean_ms={ibpr_summary['mean_ms']:.4f}, p95_ms={ibpr_summary['p95_ms']:.4f}, "
                f"recall@k={ibpr_summary['recall@k']:.4f}, agreement={ibpr_summary['agreement_with_online_exhaustive@k']:.4f}"
            )
            print()

            # Main premise evaluation: the hybrid rerank only refines what retrieval already recovered.
            for multiplier in CANDIDATE_MULTIPLIERS:
                n_candidates = min(len(ibpr.V), max(TOP_K, multiplier * TOP_K))
                latencies = []
                recalls = []
                agreements = []
                coverage = []
                subset_ok = []

                for user_idx in users:
                    seen = seen_lookup.get(user_idx, set())
                    t0 = time.perf_counter()
                    #candidate_ids = retrieve_candidates(retriever, ibpr.U[user_idx], n_candidates, seen)
                    candidate_ids = retrieve_candidates(retriever, online.U[user_idx], n_candidates, seen)
                    recs = rerank_candidates(online, user_idx, candidate_ids, TOP_K)
                    latencies.append((time.perf_counter() - t0) * 1000.0)

                    relevant = gt_lookup.get(user_idx, set())
                    if relevant:
                        recalls.append(recall_at_k(recs, relevant))

                    exhaustive_topk = exhaustive_topk_by_user[user_idx]
                    agreements.append(overlap_ratio(recs, exhaustive_topk, TOP_K))
                    coverage.append(overlap_ratio(candidate_ids, exhaustive_topk, TOP_K))
                    subset_ok.append(float(set(map(int, recs)).issubset(set(map(int, candidate_ids)))))

                row = {
                    "method": f"Hybrid_rerank_from_{n_candidates}_candidates",
                    "role": "serving_hybrid",
                    "scope": EXPERIMENT_SCOPE,
                    "cold_start_included": COLD_START_INCLUDED,
                    "candidates": n_candidates,
                    "queries": len(users),
                    "mean_ms": float(np.mean(latencies)),
                    "median_ms": float(np.median(latencies)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "recall@k": float(np.nanmean(recalls)) if recalls else float("nan"),
                    "agreement_with_online_exhaustive@k": float(np.nanmean(agreements)),
                    "candidate_coverage_of_online_exhaustive@k": float(np.nanmean(coverage)),
                    "subset_of_candidate_pool_rate": float(np.mean(subset_ok)),
                }

                row["latency_overhead_vs_ibpr_ms"] = row["mean_ms"] - ibpr_summary["mean_ms"]

                row["latency_overhead_vs_ibpr_pct"] = (
                    100.0 * (row["mean_ms"] - ibpr_summary["mean_ms"]) / ibpr_summary["mean_ms"]
                    if ibpr_summary["mean_ms"] > 0 else np.nan
                )

                row["delta_recall_vs_ibpr"] = row["recall@k"] - ibpr_summary["recall@k"]

                row["delta_agreement_vs_ibpr"] = (
                        row["agreement_with_online_exhaustive@k"]
                        - ibpr_summary["agreement_with_online_exhaustive@k"]
                )

                rows.append(row)

                print(f"Hybrid with rerank over {n_candidates} recovered candidates")
                print(
                    f"  mean_ms={row['mean_ms']:.4f}, p95_ms={row['p95_ms']:.4f}, recall@k={row['recall@k']:.4f}, "
                    f"coverage={row['candidate_coverage_of_online_exhaustive@k']:.4f}, "
                    f"agreement={row['agreement_with_online_exhaustive@k']:.4f}, subset_ok={row['subset_of_candidate_pool_rate']:.4f}"
                )
                print()

            print("=" * 80)
            print("INTERPRETATION")
            print("=" * 80)
            print("- candidate_coverage_of_online_exhaustive@k shows whether retrieval already recovered the items")
            print("  that the adapted online model would ideally want in its final top-k.")
            print("- agreement_with_online_exhaustive@k shows how close the reranked result gets to the exhaustive")
            print("  online ranking when it only sees the recovered candidate pool.")
            print("- subset_of_candidate_pool_rate should be 1.0; if so, rerank is refining the recovered set,")
            print("  not replacing retrieval.")
            print("- This benchmark is warm-start only: adapt/test were filtered to entities already known in base.")
            print("- Therefore, conclusions here do not cover user cold-start or item cold-start.")
            print()

            fieldnames = [
                "method",
                "role",
                "scope",
                "cold_start_included",
                "candidates",
                "queries",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "recall@k",
                "agreement_with_online_exhaustive@k",
                "candidate_coverage_of_online_exhaustive@k",
                "subset_of_candidate_pool_rate",
                "latency_overhead_vs_ibpr_ms",
                "latency_overhead_vs_ibpr_pct",
                "delta_recall_vs_ibpr",
                "delta_agreement_vs_ibpr",
            ]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    main()
