import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime

import cornac
import numpy as np
import torch
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.models import IBPR, OnlineIBPRMejorado, FaissANN
from cornac.models.ibpr.ibpr import ibpr as ibpr_core
from cornac.models.online_ibpr_mejorado.online_ibpr_mejorado import (
    online_ibpr_mejorado as online_core,
)
from cornac.models.recommender import ANNMixin, MEASURE_DOT
from threadpoolctl import threadpool_info, threadpool_limits

import faiss


# ============================================================
# H4 frozen protocol
# ============================================================

RATING_THRESHOLD = 3.0
VARIANT = "1M"
TOP_K = 20
TARGET_BASE_FRAC = 0.60
N_STREAM_CHUNKS = 4
FINAL_SEEDS = [777, 999]

FROZEN_IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}

FROZEN_ONLINE_CONFIG = {
    "learning_rate": 0.005,
    "lamda": 1e-06,
    "batch_size": 1024,
    "n_epochs": 3,
    "loss_mode": "angular",
    "update_V": False,
    "neg_sampling": "uniform",
    "normalize": True,
    "max_steps": None,
}

# Frozen after H4 preflight V3.1.
FROZEN_ANN_CONFIG = {
    "backend": "Cornac FaissANN / FAISS IndexIVFFlat",
    "metric": "inner_product",
    "nlist": 80,
    "nprobe": 40,
    "use_gpu": False,
    "num_threads": 1,
    "seed": 42,
}

LATENCY_REPEATS = 5

PROTOCOL_VERSION = "h4_index_reuse_v1_frozen_20260914"

BASE_SPLIT_RULE = (
    "target floor(N*0.60); if the last included row shares its timestamp "
    "with following rows, move the effective boundary forward to the end "
    "of that timestamp group"
)

CHUNK_SPLIT_RULE = (
    "filter the raw future stream to the fixed warm-start universe first; "
    "then create 4 approximately equal chronological WARM chunks by cumulative "
    "row targets, moving each internal boundary forward to the end of its "
    "timestamp group"
)

PRIMARY_EVAL_RULE = (
    "at eval chunk t, H4 query users are the unique users in PRIMARY rows; "
    "PRIMARY rows retain only users that received at least one warm update "
    "in chunks before the evaluation chunk"
)

SEEN_ITEMS_RULE = (
    "point1=base+chunk1; point2=base+chunk1+chunk2; "
    "point3=base+chunk1+chunk2+chunk3; eval chunk is never added before query"
)

# Exact H1-H3 frozen-data guards.
EXPECTED_DATA_SHA256 = (
    "29da5346c5bcf37dc927771d8ffd7ec3323dc7857ed4b0f6a45278b666954d3e"
)
EXPECTED_TOTAL_POSITIVE = 836_478
EXPECTED_TARGET_BASE = 501_886
EXPECTED_EFFECTIVE_BASE = 501_887
EXPECTED_FUTURE_RAW = 334_591
EXPECTED_FUTURE_WARM = 52_467
EXPECTED_BASE_USERS = 4_037
EXPECTED_BASE_ITEMS = 3_505
EXPECTED_CHUNKS = [13_118, 13_118, 13_114, 13_117]
EXPECTED_PRIMARY_ROWS = [5_300, 8_546, 9_143]
EXPECTED_PRIMARY_USERS = [212, 315, 306]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ============================================================
# Output schemas
# ============================================================

CONDITIONS = (
    "online_reused",
    "full_stale",
    "full_rebuilt",
)

METRIC_SUFFIXES = (
    "mean_recall",
    "median_recall",
    "mean_position_agreement",
    "median_position_agreement",
    "exact_set_match_rate",
    "exact_ordered_match_rate",
    "mean_candidate_shortfall",
    "median_candidate_shortfall",
    "p95_candidate_shortfall",
    "max_candidate_shortfall",
    "candidate_shortfall_rate",
    "mean_ann_latency_ms",
    "median_ann_latency_ms",
    "p95_ann_latency_ms",
    "mean_exhaustive_latency_ms",
    "median_exhaustive_latency_ms",
    "p95_exhaustive_latency_ms",
    "latency_speedup_exhaustive_over_ann",
)

STEP_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "seed",
    "eval_point",
    "update_chunk",
    "eval_chunk",
    "n_observed_before_rows",
    "n_observed_rows",
    "n_update_rows",
    "n_primary_eval_rows",
    "n_query_users",
    "n_query_users_k_eff_zero",
    "n_base_users",
    "n_base_items",
    "online_maps_exact",
    "full_maps_exact",
    "online_v_exact_equal_base",
    "online_v_max_abs_diff_base",
    "online_v_sha256",
    "base_v_sha256",
    "full_v_exact_equal_base",
    "full_v_max_abs_diff_base",
    "full_v_sha256",
    "base_index_object_same",
    "base_index_sha256",
    "base_index_sha256_current",
    "base_index_sha256_unchanged",
    "online_operational_index_build_count",
    "online_operational_index_rebuild_count",
    "full_operational_index_rebuild_count",
    "base_index_build_time_s",
    "full_index_rebuild_time_s",
    "full_rebuilt_index_sha256",
    "mean_norm_base_v",
    "max_abs_norm_error_base_v",
    "mean_norm_online_u",
    "max_abs_norm_error_online_u",
    "mean_norm_online_v",
    "max_abs_norm_error_online_v",
    "mean_norm_full_u",
    "max_abs_norm_error_full_u",
    "mean_norm_full_v",
    "max_abs_norm_error_full_v",
]

for prefix in CONDITIONS:
    STEP_FIELDS.extend(f"{prefix}_{suffix}" for suffix in METRIC_SUFFIXES)

QUERY_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "seed",
    "eval_point",
    "condition",
    "user_idx",
    "raw_user_id",
    "n_seen_items",
    "k_eff",
    "raw_k",
    "n_ann_returned",
    "candidate_shortfall",
    "recall",
    "position_agreement",
    "exact_set_match",
    "exact_ordered_match",
    "ann_latency_median_ms",
    "exhaustive_latency_median_ms",
]

TRIAL_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "seed",
    "n_eval_points",
    "n_query_condition_rows",
    "n_base_rows",
    "n_base_users",
    "n_base_items",
    "base_train_time_s",
    "base_index_build_time_s",
    "online_operational_index_build_count",
    "online_operational_index_rebuild_count",
    "full_operational_index_rebuild_count",
    "total_full_index_rebuild_time_s",
    "all_online_v_exact_equal_base",
    "max_online_v_abs_diff_base",
    "all_online_maps_exact",
    "all_full_maps_exact",
    "n_full_points_v_equal_base",
]

for prefix in CONDITIONS:
    TRIAL_FIELDS.extend(
        [
            f"mean_{prefix}_mean_recall",
            f"mean_{prefix}_mean_position_agreement",
            f"mean_{prefix}_exact_set_match_rate",
            f"mean_{prefix}_exact_ordered_match_rate",
            f"mean_{prefix}_candidate_shortfall_rate",
            f"mean_{prefix}_median_ann_latency_ms",
            f"mean_{prefix}_median_exhaustive_latency_ms",
        ]
    )

SUMMARY_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "n_trials",
    "n_steps",
    "n_query_rows",
    "seeds",
    "ann_nlist",
    "ann_nprobe",
    "ann_num_threads",
    "ann_seed",
    "n_total_positive",
    "n_effective_base",
    "n_future_raw",
    "n_future_warm",
    "n_chunk_1",
    "n_chunk_2",
    "n_chunk_3",
    "n_chunk_4",
    "n_primary_rows_1",
    "n_primary_rows_2",
    "n_primary_rows_3",
    "all_online_v_exact_equal_base",
    "max_online_v_abs_diff_base",
    "all_online_maps_exact",
    "all_full_maps_exact",
    "n_full_points_v_equal_base",
    "mean_base_index_build_time_s",
    "mean_full_index_rebuild_time_s",
]

for prefix in CONDITIONS:
    SUMMARY_FIELDS.extend(
        [
            f"overall_{prefix}_mean_recall",
            f"overall_{prefix}_mean_position_agreement",
            f"overall_{prefix}_exact_set_match_rate",
            f"overall_{prefix}_exact_ordered_match_rate",
            f"overall_{prefix}_candidate_shortfall_rate",
            f"overall_{prefix}_median_ann_latency_ms",
            f"overall_{prefix}_median_exhaustive_latency_ms",
        ]
    )


# ============================================================
# CLI / persistence
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen H4 experiment: reuse of the FAISS item index when "
            "OnlineIBPRMejorado keeps V fixed, compared with stale and rebuilt "
            "indices for IBPR Full Retrain."
        )
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Timestamp YYYYMMDD_HHMMSS. Reusing a timestamp resumes only "
            "fully completed seeds under the exact same protocol fingerprint."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and print the frozen H4 protocol without training MovieLens models.",
    )
    return parser.parse_args()


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_csv_atomic(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(tmp, path)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def remove_seed(rows, seed):
    return [row for row in rows if int(row["seed"]) != int(seed)]


def seed_complete(seed, step_rows, query_rows, trial_rows):
    steps = [r for r in step_rows if int(r["seed"]) == int(seed)]
    trials = [r for r in trial_rows if int(r["seed"]) == int(seed)]
    queries = [r for r in query_rows if int(r["seed"]) == int(seed)]

    if len(steps) != 3 or len(trials) != 1:
        return False

    if sorted(int(r["eval_point"]) for r in steps) != [1, 2, 3]:
        return False

    # Query rows vary by number of PRIMARY users but must contain all 3 conditions.
    if not queries:
        return False
    condition_points = {
        (int(r["eval_point"]), str(r["condition"])) for r in queries
    }
    return condition_points == {
        (point, condition)
        for point in (1, 2, 3)
        for condition in CONDITIONS
    }


# ============================================================
# Generic utilities / fingerprints
# ============================================================

def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_sha256(obj):
    try:
        return _sha256_text(inspect.getsource(obj))
    except (OSError, TypeError):
        return "unavailable"


def _script_sha256():
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unavailable"


def array_sha256(array):
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def dataset_sha256(rows):
    digest = hashlib.sha256()
    for u, i, value, timestamp, original_position in rows:
        payload = (
            f"{u}\t{i}\t{float(value):.1f}\t{int(timestamp)}\t"
            f"{int(original_position)}\n"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def norm_stats(matrix):
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return float("nan"), float("nan")
    norms = np.linalg.norm(arr, axis=1)
    return float(np.mean(norms)), float(np.max(np.abs(norms - 1.0)))


def ts_min(rows):
    return int(min(row[3] for row in rows))


def ts_max(rows):
    return int(max(row[3] for row in rows))


def rows_pairs(rows):
    return {(row[0], row[1]) for row in rows}


def mean(values):
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def percentile(values, q):
    vals = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def bool_from_csv(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_baseann_class():
    for module_name in (
        "cornac.models.ann.recom_ann_base",
        "cornac.models.ann.recom_ann",
    ):
        try:
            module = __import__(module_name, fromlist=["BaseANN"])
            return getattr(module, "BaseANN")
        except Exception:
            pass
    return None


BaseANN = load_baseann_class()


def protocol_payload(data_hash):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "MovieLens 1M",
        "data_sha256": data_hash,
        "variant": VARIANT,
        "rating_threshold": RATING_THRESHOLD,
        "feedback": "rating>=3 -> 1.0; lower ratings absent",
        "target_base_fraction": TARGET_BASE_FRAC,
        "base_split_rule": BASE_SPLIT_RULE,
        "n_stream_chunks": N_STREAM_CHUNKS,
        "chunk_split_rule": CHUNK_SPLIT_RULE,
        "warm_start_rule": "future row eligible iff user and item exist in effective base",
        "primary_eval_rule": PRIMARY_EVAL_RULE,
        "seen_items_rule": SEEN_ITEMS_RULE,
        "prequential_sequence": [[1, 2], [2, 3], [3, 4]],
        "final_seeds": FINAL_SEEDS,
        "top_k": TOP_K,
        "ibpr_config": FROZEN_IBPR_CONFIG,
        "online_config": FROZEN_ONLINE_CONFIG,
        "ann_config": FROZEN_ANN_CONFIG,
        "latency_repeats": LATENCY_REPEATS,
        "retrieval_metrics": [
            "set_recall",
            "position_agreement",
            "exact_set_match",
            "exact_ordered_match",
            "candidate_shortfall",
        ],
        "full_retrain": "fresh IBPR R900 from scratch on accumulated warm history",
        "script_sha256": _script_sha256(),
        "dataset_build_sha256": _source_sha256(Dataset.build),
        "ibpr_wrapper_sha256": _source_sha256(IBPR),
        "ibpr_core_sha256": _source_sha256(ibpr_core),
        "online_wrapper_sha256": _source_sha256(OnlineIBPRMejorado),
        "online_core_sha256": _source_sha256(online_core),
        "faissann_sha256": _source_sha256(FaissANN),
        "baseann_sha256": (
            _source_sha256(BaseANN) if BaseANN is not None else "unavailable"
        ),
    }


def current_protocol_hash(data_hash):
    canonical = json.dumps(
        protocol_payload(data_hash),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_resume_protocol(rows, path, protocol_hash, data_hash):
    if not rows:
        return

    for idx, row in enumerate(rows):
        if (
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("protocol_hash") != protocol_hash
            or row.get("data_sha256") != data_hash
        ):
            raise RuntimeError(
                f"{path}: fila {idx} con fingerprint incompatible. "
                "Usa un timestamp nuevo; no mezcles ejecuciones."
            )


# ============================================================
# Implementation guards
# ============================================================

def validate_implementation_contracts():
    if not issubclass(IBPR, ANNMixin):
        raise RuntimeError("IBPR no hereda ANNMixin.")
    if not issubclass(OnlineIBPRMejorado, ANNMixin):
        raise RuntimeError("OnlineIBPRMejorado no hereda ANNMixin.")

    online_ctor = set(inspect.signature(OnlineIBPRMejorado.__init__).parameters)
    online_partial = set(
        inspect.signature(OnlineIBPRMejorado.partial_fit_recent).parameters
    )

    if "seed" not in online_ctor:
        raise RuntimeError("OnlineIBPRMejorado obsoleto: falta seed.")

    required_partial = {"recent_pairs", "history_csr", "max_steps", "n_epochs"}
    missing = required_partial - online_partial
    if missing:
        raise RuntimeError(
            "OnlineIBPRMejorado.partial_fit_recent incompatible. "
            f"Faltan {sorted(missing)}"
        )

    faiss_params = set(inspect.signature(FaissANN.__init__).parameters)
    required_faiss = {"model", "nlist", "nprobe", "use_gpu", "num_threads"}
    missing_faiss = required_faiss - faiss_params
    if missing_faiss:
        raise RuntimeError(
            "FaissANN incompatible con H4. "
            f"Faltan {sorted(missing_faiss)}"
        )

    online_source = inspect.getsource(online_core)
    expected_online_tokens = [
        "_sample_negatives_uniform",
        "history_csr",
        "recent_pairs",
        "update_V",
        "torch.nn.functional.normalize",
    ]
    missing_online = [
        token for token in expected_online_tokens if token not in online_source
    ]
    if missing_online:
        raise RuntimeError(
            "Core OnlineIBPRMejorado no coincide con versión estabilizada. "
            f"Tokens ausentes: {missing_online}"
        )

    ibpr_source = inspect.getsource(ibpr_core)
    expected_ibpr_tokens = [
        "train_set.uij_iter",
        "torch.optim.Adam([U, V]",
        "torch.nn.functional.normalize(U",
        "torch.nn.functional.normalize(V",
    ]
    missing_ibpr = [token for token in expected_ibpr_tokens if token not in ibpr_source]
    if missing_ibpr:
        raise RuntimeError(
            "Core IBPR no coincide con R900 auditado. "
            f"Tokens ausentes: {missing_ibpr}"
        )

    if FROZEN_ANN_CONFIG["nlist"] != 80 or FROZEN_ANN_CONFIG["nprobe"] != 40:
        raise RuntimeError("Configuración ANN no coincide con preflight V3.1 aprobado.")

    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(FROZEN_ANN_CONFIG["num_threads"])

    print("Implementation contracts: OK")
    print(f"  IBPR wrapper   : {inspect.getsourcefile(IBPR) or 'unknown'}")
    print(f"  IBPR core      : {inspect.getsourcefile(ibpr_core) or 'unknown'}")
    print(f"  Online wrapper : {inspect.getsourcefile(OnlineIBPRMejorado) or 'unknown'}")
    print(f"  Online core    : {inspect.getsourcefile(online_core) or 'unknown'}")
    print(f"  FaissANN       : {inspect.getsourcefile(FaissANN) or 'unknown'}")
    print()


# ============================================================
# Data protocol
# ============================================================

def load_positive_chrono_movielens():
    data = movielens.load_feedback(fmt="UIRT", variant=VARIANT)
    positive = []

    for original_position, (u, i, rating, timestamp) in enumerate(data):
        if float(rating) >= RATING_THRESHOLD:
            positive.append(
                (
                    str(u),
                    str(i),
                    1.0,
                    int(timestamp),
                    int(original_position),
                )
            )

    positive.sort(key=lambda row: (row[3], row[4]))

    if not positive:
        raise ValueError("MovieLens 1M no produjo interacciones positivas.")

    return positive


def split_base_future_strict(rows):
    n = len(rows)
    target = int(n * TARGET_BASE_FRAC)

    if target <= 0 or target >= n:
        raise ValueError(f"Corte 60/40 inválido: target={target}, n={n}")

    effective = target
    last_base_ts = rows[target - 1][3]

    while effective < n and rows[effective][3] == last_base_ts:
        effective += 1

    if effective >= n:
        raise ValueError("El ajuste por timestamp consumió todo el future stream.")

    base_rows = list(rows[:effective])
    future_rows = list(rows[effective:])

    if ts_max(base_rows) >= ts_min(future_rows):
        raise RuntimeError("Base/future no tienen separación temporal estricta.")

    return {
        "target_base_rows": target,
        "effective_base_rows": effective,
        "boundary_tie_rows_added_to_base": effective - target,
        "base_rows": base_rows,
        "future_rows": future_rows,
    }


def split_chrono_chunks_strict(rows, n_chunks=N_STREAM_CHUNKS, label="warm stream"):
    n = len(rows)

    if n < n_chunks:
        raise ValueError(f"{label} demasiado pequeño para {n_chunks} chunks.")

    boundaries = [0]
    adjustments = []

    for k in range(1, n_chunks):
        target = int(np.floor(n * k / n_chunks))
        target = max(target, boundaries[-1] + 1)

        if target >= n:
            raise ValueError(f"No se pudo crear límite interno válido en {label}.")

        effective = target
        boundary_ts = rows[target - 1][3]

        while effective < n and rows[effective][3] == boundary_ts:
            effective += 1

        if effective >= n:
            raise ValueError(f"Empate de timestamp consume el resto de {label}.")

        boundaries.append(effective)
        adjustments.append(
            {
                "boundary": k,
                "target": target,
                "effective": effective,
                "rows_shifted": effective - target,
                "timestamp": int(boundary_ts),
            }
        )

    boundaries.append(n)
    chunks = [
        list(rows[boundaries[idx]: boundaries[idx + 1]])
        for idx in range(n_chunks)
    ]

    for idx in range(n_chunks - 1):
        if ts_max(chunks[idx]) >= ts_min(chunks[idx + 1]):
            raise RuntimeError(
                f"Chunks {idx+1}/{idx+2} no tienen separación temporal estricta."
            )

    return chunks, boundaries, adjustments


def warm_start_filter(base_rows, future_rows):
    base_users = {row[0] for row in base_rows}
    base_items = {row[1] for row in base_rows}

    warm_rows = []
    excluded_user = 0
    excluded_item = 0
    excluded_both = 0

    for row in future_rows:
        known_u = row[0] in base_users
        known_i = row[1] in base_items

        if known_u and known_i:
            warm_rows.append(row)
        elif not known_u and not known_i:
            excluded_both += 1
        elif not known_u:
            excluded_user += 1
        else:
            excluded_item += 1

    if not warm_rows:
        raise ValueError("Future warm quedó vacío.")

    return {
        "base_users": base_users,
        "base_items": base_items,
        "warm_future_rows": warm_rows,
        "n_future_raw": len(future_rows),
        "n_future_warm": len(warm_rows),
        "warm_start_fraction": len(warm_rows) / len(future_rows),
        "n_excluded_unknown_user": excluded_user,
        "n_excluded_unknown_item": excluded_item,
        "n_excluded_unknown_both": excluded_both,
    }


def primary_eval_rows_for_step(known_chunks, step_idx):
    updated_users = {
        row[0]
        for chunk in known_chunks[: step_idx + 1]
        for row in chunk
    }

    allwarm_eval_rows = list(known_chunks[step_idx + 1])
    primary_rows = [
        row for row in allwarm_eval_rows if row[0] in updated_users
    ]

    if not primary_rows:
        raise RuntimeError(
            f"eval_point={step_idx+1}: PRIMARY quedó vacío."
        )

    return primary_rows, allwarm_eval_rows, updated_users


def build_primary_eval_plan(known_chunks):
    plan = []

    for step_idx in range(3):
        primary_rows, allwarm_rows, updated_users = primary_eval_rows_for_step(
            known_chunks,
            step_idx,
        )

        plan.append(
            {
                "eval_point": step_idx + 1,
                "update_chunk": step_idx + 1,
                "eval_chunk": step_idx + 2,
                "n_updated_users": len(updated_users),
                "n_primary_rows": len(primary_rows),
                "n_primary_users": len({r[0] for r in primary_rows}),
                "n_primary_items": len({r[1] for r in primary_rows}),
                "n_allwarm_rows": len(allwarm_rows),
            }
        )

    return plan


def prepare_protocol_data():
    all_rows = load_positive_chrono_movielens()
    data_hash = dataset_sha256(all_rows)

    split = split_base_future_strict(all_rows)
    warm = warm_start_filter(split["base_rows"], split["future_rows"])

    known_chunks, boundaries, adjustments = split_chrono_chunks_strict(
        warm["warm_future_rows"],
        n_chunks=N_STREAM_CHUNKS,
        label="warm-start future stream",
    )

    # Pair-level leakage guard inherited from H1-H3.
    observed = rows_pairs(split["base_rows"])

    for idx, chunk in enumerate(known_chunks, start=1):
        chunk_pairs = rows_pairs(chunk)
        overlap = observed & chunk_pairs

        if overlap:
            sample = next(iter(overlap))
            raise RuntimeError(
                f"Par (u,i) repetido entre historial y chunk {idx}: {sample}"
            )

        observed.update(chunk_pairs)

    primary_plan = build_primary_eval_plan(known_chunks)

    data = {
        "all_rows": all_rows,
        "data_sha256": data_hash,
        "target_base_rows": split["target_base_rows"],
        "effective_base_rows": split["effective_base_rows"],
        "boundary_tie_rows_added_to_base": split[
            "boundary_tie_rows_added_to_base"
        ],
        "base_rows": split["base_rows"],
        "future_rows": split["future_rows"],
        "known_chunks": known_chunks,
        "warm_chunk_boundaries": boundaries,
        "warm_chunk_adjustments": adjustments,
        "primary_eval_plan": primary_plan,
        **warm,
    }

    validate_expected_data(data)
    return data


def validate_expected_data(data):
    actuals = {
        "data_sha256": data["data_sha256"],
        "total_positive": len(data["all_rows"]),
        "target_base": data["target_base_rows"],
        "effective_base": data["effective_base_rows"],
        "future_raw": data["n_future_raw"],
        "future_warm": data["n_future_warm"],
        "base_users": len(data["base_users"]),
        "base_items": len(data["base_items"]),
        "chunks": [len(c) for c in data["known_chunks"]],
        "primary_rows": [
            p["n_primary_rows"] for p in data["primary_eval_plan"]
        ],
        "primary_users": [
            p["n_primary_users"] for p in data["primary_eval_plan"]
        ],
    }

    expected = {
        "data_sha256": EXPECTED_DATA_SHA256,
        "total_positive": EXPECTED_TOTAL_POSITIVE,
        "target_base": EXPECTED_TARGET_BASE,
        "effective_base": EXPECTED_EFFECTIVE_BASE,
        "future_raw": EXPECTED_FUTURE_RAW,
        "future_warm": EXPECTED_FUTURE_WARM,
        "base_users": EXPECTED_BASE_USERS,
        "base_items": EXPECTED_BASE_ITEMS,
        "chunks": EXPECTED_CHUNKS,
        "primary_rows": EXPECTED_PRIMARY_ROWS,
        "primary_users": EXPECTED_PRIMARY_USERS,
    }

    mismatches = {
        key: (actuals[key], expected[key])
        for key in expected
        if actuals[key] != expected[key]
    }

    if mismatches:
        raise RuntimeError(
            "H4 no reproduce exactamente el universo final H1-H3. "
            f"Mismatches={mismatches}"
        )


def print_plan(data, protocol_hash):
    print("=" * 118)
    print("H4 INDEX REUSE - ONLINEIBPRMEJORADO")
    print("=" * 118)
    print(f"Dataset                         : MovieLens 1M")
    print(f"Data SHA256                     : {data['data_sha256']}")
    print(f"Protocol version                : {PROTOCOL_VERSION}")
    print(f"Protocol hash                   : {protocol_hash}")
    print(f"Script SHA256                   : {_script_sha256()}")
    print()

    print("Frozen model configurations:")
    print(f"  R900                          : {FROZEN_IBPR_CONFIG}")
    print(f"  O014                          : {FROZEN_ONLINE_CONFIG}")
    print(f"  Seeds                         : {FINAL_SEEDS}")
    print()

    print("Frozen ANN configuration (preflight V3.1 approved):")
    print(f"  backend                       : {FROZEN_ANN_CONFIG['backend']}")
    print(f"  nlist                         : {FROZEN_ANN_CONFIG['nlist']}")
    print(f"  nprobe                        : {FROZEN_ANN_CONFIG['nprobe']}")
    print(f"  use_gpu                       : {FROZEN_ANN_CONFIG['use_gpu']}")
    print(f"  num_threads                   : {FROZEN_ANN_CONFIG['num_threads']}")
    print(f"  seed                          : {FROZEN_ANN_CONFIG['seed']}")
    print()

    print("Frozen H1-H3 data guards:")
    print(f"  positives                     : {len(data['all_rows']):,}")
    print(f"  target base                   : {data['target_base_rows']:,}")
    print(f"  effective base                : {data['effective_base_rows']:,}")
    print(f"  future raw                    : {data['n_future_raw']:,}")
    print(f"  future warm                   : {data['n_future_warm']:,}")
    print(
        f"  base users/items              : "
        f"{len(data['base_users']):,} / {len(data['base_items']):,}"
    )
    print(
        f"  chunks                        : "
        f"{[len(c) for c in data['known_chunks']]}"
    )
    print(
        f"  PRIMARY rows                  : "
        f"{[p['n_primary_rows'] for p in data['primary_eval_plan']]}"
    )
    print(
        f"  PRIMARY users                 : "
        f"{[p['n_primary_users'] for p in data['primary_eval_plan']]}"
    )
    print("  expected-data validation      : PASS")
    print()

    print("H4 conditions:")
    print("  ONLINE_REUSED_INDEX : U_online -> index(V_base)")
    print("  FULL_STALE_INDEX    : U_full   -> index(V_base)")
    print("  FULL_REBUILT_INDEX  : U_full   -> index(V_full)")
    print()

    print("Seen-items semantics:")
    print(f"  {SEEN_ITEMS_RULE}")
    print()

    print("H4a structural guards:")
    print("  Online V == V_base bitwise")
    print("  Online uid_map/iid_map == Base")
    print("  Full uid_map/iid_map == Base")
    print("  base operational index build count = 1 per seed")
    print("  Online operational rebuild count = 0")
    print("  same base ANN object reused through all 3 points")
    print()

    print("H4b retrieval metrics:")
    print("  Recall@k_eff, Position Agreement@k_eff")
    print("  Exact Set Match, Exact Ordered Match")
    print("  candidate shortfall")
    print(f"  latency: warm-up + {LATENCY_REPEATS} repeats/user, median per user")
    print()

    print("Plan-only expected action:")
    print("  no MovieLens model training")
    print("  no final-result file written")
    print()


# ============================================================
# Dataset / model construction
# ============================================================

def cornac_rows(rows):
    return [
        (u, i, float(value), int(timestamp))
        for u, i, value, timestamp, _ in rows
    ]


def build_dataset(rows, uid_map=None, iid_map=None, seed=42, exclude_unknowns=False):
    kwargs = {
        "fmt": "UIRT",
        "seed": int(seed),
        "exclude_unknowns": bool(exclude_unknowns),
    }

    if uid_map is not None:
        kwargs["global_uid_map"] = uid_map
    if iid_map is not None:
        kwargs["global_iid_map"] = iid_map

    return Dataset.build(cornac_rows(rows), **kwargs)


def assert_dataset_contract(dataset, uid_map, iid_map, n_users, n_items, label):
    if dataset.num_users != n_users or dataset.num_items != n_items:
        raise RuntimeError(
            f"{label}: dimensiones incompatibles "
            f"({dataset.num_users},{dataset.num_items}) != "
            f"({n_users},{n_items})"
        )

    if dataset.uid_map != uid_map or dataset.iid_map != iid_map:
        raise RuntimeError(
            f"{label}: uid_map/iid_map no coinciden con Base."
        )


def rows_to_pairs(rows, uid_map, iid_map):
    pairs = np.asarray(
        [[uid_map[row[0]], iid_map[row[1]]] for row in rows],
        dtype=np.int64,
    )

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise RuntimeError("recent_pairs debe tener shape (n,2).")

    return pairs


def train_base_model(base_rows, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_train_set = build_dataset(
        base_rows,
        seed=seed,
        exclude_unknowns=False,
    )

    model = IBPR(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=FROZEN_IBPR_CONFIG["max_iter"],
        learning_rate=FROZEN_IBPR_CONFIG["learning_rate"],
        lamda=FROZEN_IBPR_CONFIG["lamda"],
        batch_size=FROZEN_IBPR_CONFIG["batch_size"],
        verbose=False,
        name=f"H4_IBPR_R900_BASE_seed{seed}",
    )

    start = time.perf_counter()
    model.fit(base_train_set)
    elapsed = time.perf_counter() - start

    if model.get_vector_measure() != MEASURE_DOT:
        raise RuntimeError("IBPR base no declara MEASURE_DOT.")

    return model, base_train_set, elapsed


def initialize_online(base_train_set, base_u, base_v, seed):
    model = OnlineIBPRMejorado(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=FROZEN_ONLINE_CONFIG["n_epochs"],
        learning_rate=FROZEN_ONLINE_CONFIG["learning_rate"],
        lamda=FROZEN_ONLINE_CONFIG["lamda"],
        batch_size=FROZEN_ONLINE_CONFIG["batch_size"],
        trainable=False,
        init_params={
            "U": base_u.copy(),
            "V": base_v.copy(),
        },
        update_V=FROZEN_ONLINE_CONFIG["update_V"],
        neg_sampling=FROZEN_ONLINE_CONFIG["neg_sampling"],
        normalize=FROZEN_ONLINE_CONFIG["normalize"],
        loss_mode=FROZEN_ONLINE_CONFIG["loss_mode"],
        seed=seed,
        verbose=False,
        name=f"H4_OnlineIBPRMejorado_O014_seed{seed}",
    )

    model.fit(base_train_set)
    model.trainable = True

    if not np.array_equal(model.U, base_u):
        raise RuntimeError("Inicialización Online modificó U_base.")
    if not np.array_equal(model.V, base_v):
        raise RuntimeError("Inicialización Online modificó V_base.")
    if (
        model.uid_map != base_train_set.uid_map
        or model.iid_map != base_train_set.iid_map
    ):
        raise RuntimeError("Inicialización Online perdió mappings Base.")

    return model


def train_full_retrain(accumulated_rows, uid_map, iid_map, n_users, n_items, seed):
    train_set = build_dataset(
        accumulated_rows,
        uid_map=uid_map,
        iid_map=iid_map,
        seed=seed,
        exclude_unknowns=False,
    )

    assert_dataset_contract(
        train_set,
        uid_map,
        iid_map,
        n_users,
        n_items,
        "full_train_set",
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = IBPR(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=FROZEN_IBPR_CONFIG["max_iter"],
        learning_rate=FROZEN_IBPR_CONFIG["learning_rate"],
        lamda=FROZEN_IBPR_CONFIG["lamda"],
        batch_size=FROZEN_IBPR_CONFIG["batch_size"],
        verbose=False,
        name=f"H4_IBPR_FULL_RETRAIN_seed{seed}",
    )

    model.fit(train_set)

    if model.U.shape != (n_users, FROZEN_IBPR_CONFIG["k"]):
        raise RuntimeError("Full U shape incompatible.")
    if model.V.shape != (n_items, FROZEN_IBPR_CONFIG["k"]):
        raise RuntimeError("Full V shape incompatible.")
    if model.uid_map != uid_map or model.iid_map != iid_map:
        raise RuntimeError("Full Retrain mappings incompatibles.")
    if model.get_vector_measure() != MEASURE_DOT:
        raise RuntimeError("Full IBPR no declara MEASURE_DOT.")

    return model


# ============================================================
# FAISS index helpers
# ============================================================

def faiss_constructor_kwargs(model):
    params = set(inspect.signature(FaissANN.__init__).parameters)

    kwargs = {
        "model": model,
        "nlist": FROZEN_ANN_CONFIG["nlist"],
        "nprobe": FROZEN_ANN_CONFIG["nprobe"],
        "use_gpu": FROZEN_ANN_CONFIG["use_gpu"],
        "num_threads": FROZEN_ANN_CONFIG["num_threads"],
    }

    if "seed" in params:
        kwargs["seed"] = FROZEN_ANN_CONFIG["seed"]

    return kwargs


def get_faiss_index_object(ann):
    for name in ("index", "_index", "ann_index", "_ann_index"):
        obj = getattr(ann, name, None)
        if obj is not None:
            return obj

    for value in vars(ann).values():
        if value is None:
            continue
        module = getattr(value.__class__, "__module__", "")
        if module.startswith("faiss"):
            return value

    return None


def serialized_index_sha256(ann):
    index = get_faiss_index_object(ann)

    if index is None:
        return ""

    try:
        raw = faiss.serialize_index(index)
        raw = np.asarray(raw, dtype=np.uint8).tobytes()
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        return ""


def call_knn_query(ann, query, k):
    result = ann.knn_query(
        np.asarray(query, dtype=np.float32).reshape(1, -1),
        int(k),
    )

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("FaissANN.knn_query() retornó formato inesperado.")

    a, b = result
    a = np.asarray(a)
    b = np.asarray(b)

    if np.issubdtype(a.dtype, np.integer):
        neighbors, distances = a, b
    elif np.issubdtype(b.dtype, np.integer):
        neighbors, distances = b, a
    else:
        raise RuntimeError("No se pudo identificar salida de IDs de FaissANN.")

    return neighbors.reshape(-1), distances.reshape(-1)


def build_operational_index(model, tracker, tracker_key):
    tracker[tracker_key] += 1

    ann = FaissANN(**faiss_constructor_kwargs(model))

    start_ns = time.perf_counter_ns()
    with threadpool_limits(limits=1):
        ann.build_index()
    elapsed_s = (time.perf_counter_ns() - start_ns) / 1e9

    return ann, elapsed_s


# ============================================================
# Retrieval / timing
# ============================================================

def build_seen_by_user(rows, uid_map, iid_map):
    seen = defaultdict(set)

    for raw_u, raw_i, _, _, _ in rows:
        seen[uid_map[raw_u]].add(iid_map[raw_i])

    return seen


def exact_topk(V, user_vector, seen_items, k_eff):
    V = np.asarray(V, dtype=np.float32)
    u = np.asarray(user_vector, dtype=np.float32).reshape(-1)

    scores = V.dot(u)

    eligible_mask = np.ones(V.shape[0], dtype=bool)
    if seen_items:
        eligible_mask[np.fromiter(seen_items, dtype=np.int64)] = False

    eligible_ids = np.flatnonzero(eligible_mask)
    if len(eligible_ids) < k_eff:
        raise RuntimeError("Menos elegibles que k_eff.")

    eligible_scores = scores[eligible_ids]

    if len(eligible_ids) == k_eff:
        candidate_ids = eligible_ids
    else:
        part = np.argpartition(-eligible_scores, k_eff - 1)[:k_eff]
        cutoff = float(np.min(eligible_scores[part]))

        strict_ids = eligible_ids[eligible_scores > cutoff]
        tied_ids = eligible_ids[eligible_scores == cutoff]
        candidate_ids = np.concatenate((strict_ids, tied_ids))

    candidate_scores = scores[candidate_ids]
    order = np.lexsort((candidate_ids, -candidate_scores))
    return candidate_ids[order[:k_eff]].astype(np.int64)


def ann_topk(ann, user_vector, seen_items, k_eff, n_items):
    raw_k = min(n_items, k_eff + len(seen_items))

    neighbors, _ = call_knn_query(
        ann,
        user_vector,
        raw_k,
    )

    out = []
    used = set()

    for value in neighbors:
        item_idx = int(value)

        if item_idx < 0 or item_idx >= n_items:
            continue
        if item_idx in used:
            continue
        used.add(item_idx)

        if item_idx in seen_items:
            continue

        out.append(item_idx)

        if len(out) >= k_eff:
            break

    return np.asarray(out, dtype=np.int64), raw_k


def retrieval_metrics(ann_ids, exact_ids, k_eff):
    ann_list = [int(x) for x in ann_ids]
    exact_list = [int(x) for x in exact_ids]

    ann_set = set(ann_list)
    exact_set = set(exact_list)

    recall = len(ann_set & exact_set) / k_eff

    position_matches = sum(
        1
        for pos in range(min(len(ann_list), k_eff))
        if ann_list[pos] == exact_list[pos]
    )
    position_agreement = position_matches / k_eff

    exact_set_match = (
        len(ann_list) == k_eff and ann_set == exact_set
    )
    exact_ordered_match = (
        len(ann_list) == k_eff and ann_list == exact_list
    )

    shortfall = k_eff - len(ann_list)

    return {
        "recall": float(recall),
        "position_agreement": float(position_agreement),
        "exact_set_match": bool(exact_set_match),
        "exact_ordered_match": bool(exact_ordered_match),
        "candidate_shortfall": int(shortfall),
    }


def benchmark_exhaustive(V, user_vector, seen_items, k_eff):
    # One warm-up.
    with threadpool_limits(limits=1):
        exact_topk(V, user_vector, seen_items, k_eff)

    samples = []

    for _ in range(LATENCY_REPEATS):
        start_ns = time.perf_counter_ns()
        with threadpool_limits(limits=1):
            exact_topk(V, user_vector, seen_items, k_eff)
        samples.append((time.perf_counter_ns() - start_ns) / 1e6)

    return float(np.median(samples))


def benchmark_ann(ann, user_vector, seen_items, k_eff, n_items):
    # One warm-up.
    with threadpool_limits(limits=1):
        ann_topk(ann, user_vector, seen_items, k_eff, n_items)

    samples = []

    for _ in range(LATENCY_REPEATS):
        start_ns = time.perf_counter_ns()
        with threadpool_limits(limits=1):
            ann_topk(ann, user_vector, seen_items, k_eff, n_items)
        samples.append((time.perf_counter_ns() - start_ns) / 1e6)

    return float(np.median(samples))


def evaluate_condition(
    *,
    condition,
    seed,
    eval_point,
    ann,
    U,
    V_ground_truth,
    query_user_ids,
    inverse_uid_map,
    seen_by_user,
    n_items,
    protocol_hash,
    data_hash,
):
    rows = []

    for user_idx in query_user_ids:
        seen_items = seen_by_user.get(user_idx, set())
        k_eff = min(TOP_K, n_items - len(seen_items))

        if k_eff <= 0:
            continue

        user_vector = np.asarray(U[user_idx], dtype=np.float32)

        with threadpool_limits(limits=1):
            exact_ids = exact_topk(
                V_ground_truth,
                user_vector,
                seen_items,
                k_eff,
            )
            ann_ids, raw_k = ann_topk(
                ann,
                user_vector,
                seen_items,
                k_eff,
                n_items,
            )

        metrics = retrieval_metrics(ann_ids, exact_ids, k_eff)

        exhaustive_latency = benchmark_exhaustive(
            V_ground_truth,
            user_vector,
            seen_items,
            k_eff,
        )
        ann_latency = benchmark_ann(
            ann,
            user_vector,
            seen_items,
            k_eff,
            n_items,
        )

        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash,
                "data_sha256": data_hash,
                "seed": seed,
                "eval_point": eval_point,
                "condition": condition,
                "user_idx": user_idx,
                "raw_user_id": inverse_uid_map[user_idx],
                "n_seen_items": len(seen_items),
                "k_eff": k_eff,
                "raw_k": raw_k,
                "n_ann_returned": len(ann_ids),
                "candidate_shortfall": metrics["candidate_shortfall"],
                "recall": metrics["recall"],
                "position_agreement": metrics["position_agreement"],
                "exact_set_match": metrics["exact_set_match"],
                "exact_ordered_match": metrics["exact_ordered_match"],
                "ann_latency_median_ms": ann_latency,
                "exhaustive_latency_median_ms": exhaustive_latency,
            }
        )

    return rows


def aggregate_query_rows(rows):
    if not rows:
        raise RuntimeError("No hay query rows para agregar.")

    recall = [float(r["recall"]) for r in rows]
    pos = [float(r["position_agreement"]) for r in rows]
    shortfall = [int(r["candidate_shortfall"]) for r in rows]
    ann_ms = [float(r["ann_latency_median_ms"]) for r in rows]
    ex_ms = [float(r["exhaustive_latency_median_ms"]) for r in rows]

    return {
        "mean_recall": mean(recall),
        "median_recall": percentile(recall, 50),
        "mean_position_agreement": mean(pos),
        "median_position_agreement": percentile(pos, 50),
        "exact_set_match_rate": mean(
            1.0 if bool(r["exact_set_match"]) else 0.0 for r in rows
        ),
        "exact_ordered_match_rate": mean(
            1.0 if bool(r["exact_ordered_match"]) else 0.0 for r in rows
        ),
        "mean_candidate_shortfall": mean(shortfall),
        "median_candidate_shortfall": percentile(shortfall, 50),
        "p95_candidate_shortfall": percentile(shortfall, 95),
        "max_candidate_shortfall": max(shortfall),
        "candidate_shortfall_rate": mean(
            1.0 if int(v) > 0 else 0.0 for v in shortfall
        ),
        "mean_ann_latency_ms": mean(ann_ms),
        "median_ann_latency_ms": percentile(ann_ms, 50),
        "p95_ann_latency_ms": percentile(ann_ms, 95),
        "mean_exhaustive_latency_ms": mean(ex_ms),
        "median_exhaustive_latency_ms": percentile(ex_ms, 50),
        "p95_exhaustive_latency_ms": percentile(ex_ms, 95),
        "latency_speedup_exhaustive_over_ann": (
            percentile(ex_ms, 50) / percentile(ann_ms, 50)
            if percentile(ann_ms, 50) > 0
            else float("inf")
        ),
    }


# ============================================================
# Trial execution
# ============================================================

def run_seed_trial(seed, data, protocol_hash):
    print()
    print("=" * 118)
    print(f"RUN H4 | seed={seed}")
    print("=" * 118)

    base_model, base_train_set, base_train_time = train_base_model(
        data["base_rows"],
        seed,
    )

    uid_map = dict(base_train_set.uid_map)
    iid_map = dict(base_train_set.iid_map)
    inverse_uid_map = {idx: raw for raw, idx in uid_map.items()}

    n_users = base_train_set.num_users
    n_items = base_train_set.num_items

    if n_users != EXPECTED_BASE_USERS or n_items != EXPECTED_BASE_ITEMS:
        raise RuntimeError(
            f"Universo base inesperado: users/items={n_users}/{n_items}"
        )

    base_u = np.asarray(base_model.U).copy()
    base_v = np.asarray(base_model.V).copy()
    base_v_hash = array_sha256(base_v)

    online_model = initialize_online(
        base_train_set,
        base_u,
        base_v,
        seed,
    )

    tracker = defaultdict(int)

    base_ann, base_index_build_time = build_operational_index(
        base_model,
        tracker,
        "online_build",
    )

    if tracker["online_build"] != 1:
        raise RuntimeError("El índice base no fue construido exactamente una vez.")

    base_ann_object_id = id(base_ann)
    base_index_hash = serialized_index_sha256(base_ann)

    print(
        f"BASE: rows={len(data['base_rows']):,} | users={n_users:,} | "
        f"items={n_items:,} | train={base_train_time:.2f}s | "
        f"index_build={base_index_build_time:.4f}s"
    )
    print(f"BASE index SHA256: {base_index_hash or 'unavailable'}")

    observed_rows = list(data["base_rows"])

    step_rows = []
    query_rows = []
    total_full_rebuild_time = 0.0

    for step_idx in range(3):
        eval_point = step_idx + 1
        update_chunk_number = step_idx + 1
        eval_chunk_number = step_idx + 2

        update_rows = list(data["known_chunks"][step_idx])
        primary_rows, allwarm_eval_rows, updated_users = primary_eval_rows_for_step(
            data["known_chunks"],
            step_idx,
        )

        if ts_max(update_rows) >= ts_min(allwarm_eval_rows):
            raise RuntimeError(
                f"step {eval_point}: update/eval no estrictamente temporales."
            )

        eval_pairs = rows_pairs(allwarm_eval_rows)
        post_update_rows = observed_rows + update_rows

        if eval_pairs & rows_pairs(post_update_rows):
            sample = next(iter(eval_pairs & rows_pairs(post_update_rows)))
            raise RuntimeError(
                f"step {eval_point}: leakage de pair eval/post-update: {sample}"
            )

        eval_train_set = build_dataset(
            post_update_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=False,
        )

        assert_dataset_contract(
            eval_train_set,
            uid_map,
            iid_map,
            n_users,
            n_items,
            "eval_train_set",
        )

        recent_pairs = rows_to_pairs(
            update_rows,
            uid_map,
            iid_map,
        )

        online_v_before = np.asarray(online_model.V).copy()

        online_model.partial_fit_recent(
            recent_pairs=recent_pairs,
            history_csr=eval_train_set.csr_matrix,
            max_steps=FROZEN_ONLINE_CONFIG["max_steps"],
            n_epochs=FROZEN_ONLINE_CONFIG["n_epochs"],
        )

        if not np.array_equal(online_model.V, online_v_before):
            raise RuntimeError(
                f"step {eval_point}: Online modificó V respecto del estado previo."
            )

        online_v_exact = np.array_equal(online_model.V, base_v)
        online_v_diff = max_abs_diff(online_model.V, base_v)

        if not online_v_exact or online_v_diff != 0.0:
            raise RuntimeError(
                f"step {eval_point}: V_online != V_base, diff={online_v_diff}"
            )

        online_maps_ok = (
            online_model.uid_map == uid_map
            and online_model.iid_map == iid_map
            and online_model.num_users == n_users
            and online_model.num_items == n_items
        )

        if not online_maps_ok:
            raise RuntimeError(
                f"step {eval_point}: Online mappings incompatibles."
            )

        # Full Retrain on the exact same accumulated post-update history.
        full_model = train_full_retrain(
            post_update_rows,
            uid_map,
            iid_map,
            n_users,
            n_items,
            seed,
        )

        full_maps_ok = (
            full_model.uid_map == uid_map
            and full_model.iid_map == iid_map
            and full_model.num_users == n_users
            and full_model.num_items == n_items
        )

        if not full_maps_ok:
            raise RuntimeError(
                f"step {eval_point}: Full mappings incompatibles."
            )

        full_v_exact = np.array_equal(full_model.V, base_v)
        full_v_diff = max_abs_diff(full_model.V, base_v)

        # Faithful current Full index: operational rebuild.
        full_ann, full_rebuild_time = build_operational_index(
            full_model,
            tracker,
            "full_rebuild",
        )
        total_full_rebuild_time += full_rebuild_time

        # Structural proof that base index remained the same operational object/state.
        if id(base_ann) != base_ann_object_id:
            raise RuntimeError("El objeto base ANN cambió.")

        current_base_index_hash = serialized_index_sha256(base_ann)
        hash_unchanged = (
            True
            if not base_index_hash or not current_base_index_hash
            else base_index_hash == current_base_index_hash
        )

        if not hash_unchanged:
            raise RuntimeError(
                f"step {eval_point}: el índice base cambió sin rebuild."
            )

        # PRIMARY query population = unique PRIMARY users.
        query_user_ids = sorted(
            {uid_map[row[0]] for row in primary_rows}
        )

        if len(query_user_ids) != EXPECTED_PRIMARY_USERS[step_idx]:
            raise RuntimeError(
                f"step {eval_point}: query users={len(query_user_ids)} "
                f"!= esperado {EXPECTED_PRIMARY_USERS[step_idx]}"
            )

        seen_by_user = build_seen_by_user(
            post_update_rows,
            uid_map,
            iid_map,
        )

        n_k_eff_zero = sum(
            1
            for u in query_user_ids
            if min(TOP_K, n_items - len(seen_by_user.get(u, set()))) <= 0
        )

        # Explicit-query retrieval; FaissANN.rank() is intentionally not used.
        online_query_rows = evaluate_condition(
            condition="online_reused",
            seed=seed,
            eval_point=eval_point,
            ann=base_ann,
            U=np.asarray(online_model.U),
            V_ground_truth=np.asarray(online_model.V),
            query_user_ids=query_user_ids,
            inverse_uid_map=inverse_uid_map,
            seen_by_user=seen_by_user,
            n_items=n_items,
            protocol_hash=protocol_hash,
            data_hash=data["data_sha256"],
        )

        full_stale_query_rows = evaluate_condition(
            condition="full_stale",
            seed=seed,
            eval_point=eval_point,
            ann=base_ann,
            U=np.asarray(full_model.U),
            V_ground_truth=np.asarray(full_model.V),
            query_user_ids=query_user_ids,
            inverse_uid_map=inverse_uid_map,
            seen_by_user=seen_by_user,
            n_items=n_items,
            protocol_hash=protocol_hash,
            data_hash=data["data_sha256"],
        )

        full_rebuilt_query_rows = evaluate_condition(
            condition="full_rebuilt",
            seed=seed,
            eval_point=eval_point,
            ann=full_ann,
            U=np.asarray(full_model.U),
            V_ground_truth=np.asarray(full_model.V),
            query_user_ids=query_user_ids,
            inverse_uid_map=inverse_uid_map,
            seen_by_user=seen_by_user,
            n_items=n_items,
            protocol_hash=protocol_hash,
            data_hash=data["data_sha256"],
        )

        condition_rows = {
            "online_reused": online_query_rows,
            "full_stale": full_stale_query_rows,
            "full_rebuilt": full_rebuilt_query_rows,
        }

        if any(len(rows) != len(query_user_ids) - n_k_eff_zero for rows in condition_rows.values()):
            raise RuntimeError(
                f"step {eval_point}: conteo de query rows inconsistente."
            )

        query_rows.extend(
            online_query_rows
            + full_stale_query_rows
            + full_rebuilt_query_rows
        )

        aggregates = {
            condition: aggregate_query_rows(rows)
            for condition, rows in condition_rows.items()
        }

        mean_base_v_norm, max_base_v_norm_err = norm_stats(base_v)
        mean_online_u_norm, max_online_u_norm_err = norm_stats(online_model.U)
        mean_online_v_norm, max_online_v_norm_err = norm_stats(online_model.V)
        mean_full_u_norm, max_full_u_norm_err = norm_stats(full_model.U)
        mean_full_v_norm, max_full_v_norm_err = norm_stats(full_model.V)

        step_row = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash,
            "data_sha256": data["data_sha256"],
            "seed": seed,
            "eval_point": eval_point,
            "update_chunk": update_chunk_number,
            "eval_chunk": eval_chunk_number,
            "n_observed_before_rows": len(observed_rows),
            "n_observed_rows": len(post_update_rows),
            "n_update_rows": len(update_rows),
            "n_primary_eval_rows": len(primary_rows),
            "n_query_users": len(query_user_ids),
            "n_query_users_k_eff_zero": n_k_eff_zero,
            "n_base_users": n_users,
            "n_base_items": n_items,
            "online_maps_exact": online_maps_ok,
            "full_maps_exact": full_maps_ok,
            "online_v_exact_equal_base": online_v_exact,
            "online_v_max_abs_diff_base": online_v_diff,
            "online_v_sha256": array_sha256(online_model.V),
            "base_v_sha256": base_v_hash,
            "full_v_exact_equal_base": full_v_exact,
            "full_v_max_abs_diff_base": full_v_diff,
            "full_v_sha256": array_sha256(full_model.V),
            "base_index_object_same": id(base_ann) == base_ann_object_id,
            "base_index_sha256": base_index_hash,
            "base_index_sha256_current": current_base_index_hash,
            "base_index_sha256_unchanged": hash_unchanged,
            "online_operational_index_build_count": tracker["online_build"],
            "online_operational_index_rebuild_count": 0,
            "full_operational_index_rebuild_count": tracker["full_rebuild"],
            "base_index_build_time_s": base_index_build_time,
            "full_index_rebuild_time_s": full_rebuild_time,
            "full_rebuilt_index_sha256": serialized_index_sha256(full_ann),
            "mean_norm_base_v": mean_base_v_norm,
            "max_abs_norm_error_base_v": max_base_v_norm_err,
            "mean_norm_online_u": mean_online_u_norm,
            "max_abs_norm_error_online_u": max_online_u_norm_err,
            "mean_norm_online_v": mean_online_v_norm,
            "max_abs_norm_error_online_v": max_online_v_norm_err,
            "mean_norm_full_u": mean_full_u_norm,
            "max_abs_norm_error_full_u": max_full_u_norm_err,
            "mean_norm_full_v": mean_full_v_norm,
            "max_abs_norm_error_full_v": max_full_v_norm_err,
        }

        for condition in CONDITIONS:
            for suffix in METRIC_SUFFIXES:
                step_row[f"{condition}_{suffix}"] = aggregates[condition][suffix]

        step_rows.append(step_row)

        print(
            f"  point {eval_point} | PRIMARY rows={len(primary_rows):,} | "
            f"query_users={len(query_user_ids):,} | "
            f"Online reused Recall={aggregates['online_reused']['mean_recall']:.6f} | "
            f"Full stale Recall={aggregates['full_stale']['mean_recall']:.6f} | "
            f"Full rebuilt Recall={aggregates['full_rebuilt']['mean_recall']:.6f} | "
            f"Full rebuild={full_rebuild_time:.4f}s | "
            f"V_online exact={online_v_exact} | V_full exact={full_v_exact}"
        )

        observed_rows = post_update_rows

    if tracker["online_build"] != 1:
        raise RuntimeError(
            f"seed={seed}: base build count={tracker['online_build']} != 1"
        )

    if tracker["full_rebuild"] != 3:
        raise RuntimeError(
            f"seed={seed}: full rebuild count={tracker['full_rebuild']} != 3"
        )

    if len(step_rows) != 3:
        raise RuntimeError(f"seed={seed}: no produjo 3 steps.")

    trial = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "data_sha256": data["data_sha256"],
        "seed": seed,
        "n_eval_points": 3,
        "n_query_condition_rows": len(query_rows),
        "n_base_rows": len(data["base_rows"]),
        "n_base_users": n_users,
        "n_base_items": n_items,
        "base_train_time_s": base_train_time,
        "base_index_build_time_s": base_index_build_time,
        "online_operational_index_build_count": tracker["online_build"],
        "online_operational_index_rebuild_count": 0,
        "full_operational_index_rebuild_count": tracker["full_rebuild"],
        "total_full_index_rebuild_time_s": total_full_rebuild_time,
        "all_online_v_exact_equal_base": all(
            bool(row["online_v_exact_equal_base"]) for row in step_rows
        ),
        "max_online_v_abs_diff_base": max(
            float(row["online_v_max_abs_diff_base"]) for row in step_rows
        ),
        "all_online_maps_exact": all(
            bool(row["online_maps_exact"]) for row in step_rows
        ),
        "all_full_maps_exact": all(
            bool(row["full_maps_exact"]) for row in step_rows
        ),
        "n_full_points_v_equal_base": sum(
            1 for row in step_rows if bool(row["full_v_exact_equal_base"])
        ),
    }

    for condition in CONDITIONS:
        trial[f"mean_{condition}_mean_recall"] = mean(
            row[f"{condition}_mean_recall"] for row in step_rows
        )
        trial[f"mean_{condition}_mean_position_agreement"] = mean(
            row[f"{condition}_mean_position_agreement"] for row in step_rows
        )
        trial[f"mean_{condition}_exact_set_match_rate"] = mean(
            row[f"{condition}_exact_set_match_rate"] for row in step_rows
        )
        trial[f"mean_{condition}_exact_ordered_match_rate"] = mean(
            row[f"{condition}_exact_ordered_match_rate"] for row in step_rows
        )
        trial[f"mean_{condition}_candidate_shortfall_rate"] = mean(
            row[f"{condition}_candidate_shortfall_rate"] for row in step_rows
        )
        trial[f"mean_{condition}_median_ann_latency_ms"] = mean(
            row[f"{condition}_median_ann_latency_ms"] for row in step_rows
        )
        trial[f"mean_{condition}_median_exhaustive_latency_ms"] = mean(
            row[f"{condition}_median_exhaustive_latency_ms"] for row in step_rows
        )

    return step_rows, query_rows, trial


# ============================================================
# Summary
# ============================================================

def build_summary(step_rows, query_rows, trial_rows, data, protocol_hash):
    if len(step_rows) != 6 or len(trial_rows) != 2:
        raise RuntimeError(
            f"Summary requiere 6 steps y 2 trials; "
            f"recibió {len(step_rows)} / {len(trial_rows)}"
        )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "data_sha256": data["data_sha256"],
        "n_trials": len(trial_rows),
        "n_steps": len(step_rows),
        "n_query_rows": len(query_rows),
        "seeds": ",".join(str(s) for s in FINAL_SEEDS),
        "ann_nlist": FROZEN_ANN_CONFIG["nlist"],
        "ann_nprobe": FROZEN_ANN_CONFIG["nprobe"],
        "ann_num_threads": FROZEN_ANN_CONFIG["num_threads"],
        "ann_seed": FROZEN_ANN_CONFIG["seed"],
        "n_total_positive": len(data["all_rows"]),
        "n_effective_base": len(data["base_rows"]),
        "n_future_raw": data["n_future_raw"],
        "n_future_warm": data["n_future_warm"],
        "n_chunk_1": len(data["known_chunks"][0]),
        "n_chunk_2": len(data["known_chunks"][1]),
        "n_chunk_3": len(data["known_chunks"][2]),
        "n_chunk_4": len(data["known_chunks"][3]),
        "n_primary_rows_1": data["primary_eval_plan"][0]["n_primary_rows"],
        "n_primary_rows_2": data["primary_eval_plan"][1]["n_primary_rows"],
        "n_primary_rows_3": data["primary_eval_plan"][2]["n_primary_rows"],
        "all_online_v_exact_equal_base": all(
            bool_from_csv(row["online_v_exact_equal_base"])
            if isinstance(row["online_v_exact_equal_base"], str)
            else bool(row["online_v_exact_equal_base"])
            for row in step_rows
        ),
        "max_online_v_abs_diff_base": max(
            float(row["online_v_max_abs_diff_base"]) for row in step_rows
        ),
        "all_online_maps_exact": all(
            bool_from_csv(row["online_maps_exact"])
            if isinstance(row["online_maps_exact"], str)
            else bool(row["online_maps_exact"])
            for row in step_rows
        ),
        "all_full_maps_exact": all(
            bool_from_csv(row["full_maps_exact"])
            if isinstance(row["full_maps_exact"], str)
            else bool(row["full_maps_exact"])
            for row in step_rows
        ),
        "n_full_points_v_equal_base": sum(
            1
            for row in step_rows
            if (
                bool_from_csv(row["full_v_exact_equal_base"])
                if isinstance(row["full_v_exact_equal_base"], str)
                else bool(row["full_v_exact_equal_base"])
            )
        ),
        "mean_base_index_build_time_s": mean(
            row["base_index_build_time_s"] for row in trial_rows
        ),
        "mean_full_index_rebuild_time_s": mean(
            row["total_full_index_rebuild_time_s"] for row in trial_rows
        ) / 3.0,
    }

    for condition in CONDITIONS:
        cond_rows = [
            row for row in query_rows if str(row["condition"]) == condition
        ]

        summary[f"overall_{condition}_mean_recall"] = mean(
            row["recall"] for row in cond_rows
        )
        summary[f"overall_{condition}_mean_position_agreement"] = mean(
            row["position_agreement"] for row in cond_rows
        )
        summary[f"overall_{condition}_exact_set_match_rate"] = mean(
            1.0
            if (
                bool_from_csv(row["exact_set_match"])
                if isinstance(row["exact_set_match"], str)
                else bool(row["exact_set_match"])
            )
            else 0.0
            for row in cond_rows
        )
        summary[f"overall_{condition}_exact_ordered_match_rate"] = mean(
            1.0
            if (
                bool_from_csv(row["exact_ordered_match"])
                if isinstance(row["exact_ordered_match"], str)
                else bool(row["exact_ordered_match"])
            )
            else 0.0
            for row in cond_rows
        )
        summary[f"overall_{condition}_candidate_shortfall_rate"] = mean(
            1.0 if int(row["candidate_shortfall"]) > 0 else 0.0
            for row in cond_rows
        )
        summary[f"overall_{condition}_median_ann_latency_ms"] = percentile(
            (float(row["ann_latency_median_ms"]) for row in cond_rows),
            50,
        )
        summary[f"overall_{condition}_median_exhaustive_latency_ms"] = percentile(
            (float(row["exhaustive_latency_median_ms"]) for row in cond_rows),
            50,
        )

    return summary


def print_summary(summary):
    print()
    print("=" * 118)
    print("H4 SUMMARY")
    print("=" * 118)
    print(
        f"Online V exact all points       : "
        f"{summary['all_online_v_exact_equal_base']}"
    )
    print(
        f"Max |V_online - V_base|         : "
        f"{float(summary['max_online_v_abs_diff_base']):.12g}"
    )
    print(
        f"Full points V exactly base      : "
        f"{summary['n_full_points_v_equal_base']} / 6"
    )
    print()

    for condition in CONDITIONS:
        print(
            f"{condition:16s} | "
            f"Recall={float(summary[f'overall_{condition}_mean_recall']):.6f} | "
            f"PosAgree={float(summary[f'overall_{condition}_mean_position_agreement']):.6f} | "
            f"ExactSet={float(summary[f'overall_{condition}_exact_set_match_rate']):.6f} | "
            f"ShortfallRate={float(summary[f'overall_{condition}_candidate_shortfall_rate']):.6f} | "
            f"ANN median={float(summary[f'overall_{condition}_median_ann_latency_ms']):.4f} ms | "
            f"Exhaustive median={float(summary[f'overall_{condition}_median_exhaustive_latency_ms']):.4f} ms"
        )

    print()
    print("Interpretation guard:")
    print("  - H4a is structural: unchanged V means unchanged indexed item corpus.")
    print("  - FAISS IVF retrieval is approximate; Recall need not equal 1.0.")
    print("  - FULL_STALE_INDEX is a cross-representation operational control.")
    print("  - No ANN retuning is permitted after these results.")
    print()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    validate_implementation_contracts()
    data = prepare_protocol_data()
    protocol_hash = current_protocol_hash(data["data_sha256"])

    print_plan(data, protocol_hash)

    if args.plan_only:
        print(
            "PLAN-ONLY complete. No MovieLens model was trained "
            "and no H4 result file was written."
        )
        return

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_path = os.path.join(
        RESULTS_DIR,
        f"h4_index_reuse_online_ibpr_mejorado_{timestamp}.txt",
    )
    steps_path = os.path.join(
        RESULTS_DIR,
        f"h4_index_reuse_online_ibpr_mejorado_steps_{timestamp}.csv",
    )
    queries_path = os.path.join(
        RESULTS_DIR,
        f"h4_index_reuse_online_ibpr_mejorado_queries_{timestamp}.csv",
    )
    trials_path = os.path.join(
        RESULTS_DIR,
        f"h4_index_reuse_online_ibpr_mejorado_trials_{timestamp}.csv",
    )
    summary_path = os.path.join(
        RESULTS_DIR,
        f"h4_index_reuse_online_ibpr_mejorado_summary_{timestamp}.csv",
    )

    step_rows = load_csv(steps_path)
    query_rows = load_csv(queries_path)
    trial_rows = load_csv(trials_path)

    validate_resume_protocol(
        step_rows,
        steps_path,
        protocol_hash,
        data["data_sha256"],
    )
    validate_resume_protocol(
        query_rows,
        queries_path,
        protocol_hash,
        data["data_sha256"],
    )
    validate_resume_protocol(
        trial_rows,
        trials_path,
        protocol_hash,
        data["data_sha256"],
    )

    log_mode = "a" if os.path.exists(log_path) else "w"

    with open(log_path, log_mode, encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)

        with redirect_stdout(tee):
            print()
            print("#" * 118)
            print(f"H4 TIMESTAMP: {timestamp}")
            print("#" * 118)
            print(f"Log     : {log_path}")
            print(f"Steps   : {steps_path}")
            print(f"Queries : {queries_path}")
            print(f"Trials  : {trials_path}")
            print(f"Summary : {summary_path}")
            print(f"Protocol: {protocol_hash}")
            print()

            print("Environment:")
            print(f"  Python : {sys.version.split()[0]}")
            print(f"  OS     : {platform.platform()}")
            print(f"  Cornac : {getattr(cornac, '__version__', 'unavailable')}")
            print(f"  NumPy  : {np.__version__}")
            print(f"  PyTorch: {torch.__version__}")
            print(f"  FAISS  : {getattr(faiss, '__version__', 'unavailable')}")
            print(f"  Threadpools: {threadpool_info()}")
            print()

            for seed in FINAL_SEEDS:
                if seed_complete(
                    seed,
                    step_rows,
                    query_rows,
                    trial_rows,
                ):
                    print(f"REUSE seed={seed}: complete.")
                    continue

                step_rows = remove_seed(step_rows, seed)
                query_rows = remove_seed(query_rows, seed)
                trial_rows = remove_seed(trial_rows, seed)

                save_csv_atomic(steps_path, STEP_FIELDS, step_rows)
                save_csv_atomic(queries_path, QUERY_FIELDS, query_rows)
                save_csv_atomic(trials_path, TRIAL_FIELDS, trial_rows)

                new_steps, new_queries, new_trial = run_seed_trial(
                    seed,
                    data,
                    protocol_hash,
                )

                step_rows.extend(new_steps)
                query_rows.extend(new_queries)
                trial_rows.append(new_trial)

                # Persist atomically only after the full seed completed.
                save_csv_atomic(steps_path, STEP_FIELDS, step_rows)
                save_csv_atomic(queries_path, QUERY_FIELDS, query_rows)
                save_csv_atomic(trials_path, TRIAL_FIELDS, trial_rows)

            for seed in FINAL_SEEDS:
                if not seed_complete(
                    seed,
                    step_rows,
                    query_rows,
                    trial_rows,
                ):
                    raise RuntimeError(
                        f"seed={seed} no quedó completa al cierre."
                    )

            if len(step_rows) != 6 or len(trial_rows) != 2:
                raise RuntimeError(
                    f"Conteos finales inválidos: "
                    f"steps={len(step_rows)}, trials={len(trial_rows)}"
                )

            step_keys = [
                (int(row["seed"]), int(row["eval_point"]))
                for row in step_rows
            ]
            if len(step_keys) != len(set(step_keys)):
                raise RuntimeError("Steps duplicados.")

            query_keys = [
                (
                    int(row["seed"]),
                    int(row["eval_point"]),
                    str(row["condition"]),
                    int(row["user_idx"]),
                )
                for row in query_rows
            ]
            if len(query_keys) != len(set(query_keys)):
                raise RuntimeError("Query rows duplicadas.")

            summary = build_summary(
                step_rows,
                query_rows,
                trial_rows,
                data,
                protocol_hash,
            )

            save_csv_atomic(
                summary_path,
                SUMMARY_FIELDS,
                [summary],
            )

            print_summary(summary)

            print("Files:")
            print(f"  log     : {log_path}")
            print(f"  steps   : {steps_path}")
            print(f"  queries : {queries_path}")
            print(f"  trials  : {trials_path}")
            print(f"  summary : {summary_path}")


if __name__ == "__main__":
    main()
