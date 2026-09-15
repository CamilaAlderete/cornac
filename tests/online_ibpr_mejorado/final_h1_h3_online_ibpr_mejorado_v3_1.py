import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import torch
import cornac
import scipy
from scipy.sparse import csr_matrix
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.models.ibpr.ibpr import ibpr as ibpr_core
from cornac.models.online_ibpr_mejorado.online_ibpr_mejorado import (
    online_ibpr_mejorado as online_core,
)


# ============================================================
# Frozen final H1-H3 protocol
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

QUALITY_METRICS = [
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
]

PROTOCOL_VERSION = "final_h1_h3_v3_1_hardened_20260914"
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
    "at eval chunk t, primary H1-H3 population contains only rows whose user "
    "received at least one warm update in chunks < t; all warm rows in the eval "
    "chunk are retained as a supplementary diagnostic population"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ============================================================
# CSV schemas
# ============================================================

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
    "n_eval_rows",
    "n_eval_users",
    "n_eval_items",
    "n_allwarm_eval_rows",
    "n_allwarm_eval_users",
    "n_allwarm_eval_items",
    "n_allwarm_eval_unadapted_users",
    "update_min_timestamp",
    "update_max_timestamp",
    "eval_min_timestamp",
    "eval_max_timestamp",
    "history_dataset_build_time_s",
    "test_datasets_build_time_s",
    "full_dataset_build_time_s",
    "online_update_time_s",
    "full_retrain_time_s",
    "online_construction_inclusive_time_s",
    "full_construction_inclusive_time_s",
    "cumulative_online_time_s",
    "cumulative_full_retrain_time_s",
    "cumulative_online_construction_inclusive_time_s",
    "cumulative_full_construction_inclusive_time_s",
    "step_speedup_full_over_online",
    "step_online_full_cost_fraction",
    "cumulative_speedup_full_over_online",
    "cumulative_online_full_cost_fraction",
    "step_construction_inclusive_speedup_full_over_online",
    "step_construction_inclusive_online_full_cost_fraction",
    "cumulative_construction_inclusive_speedup_full_over_online",
    "cumulative_construction_inclusive_online_full_cost_fraction",
    "adaptation_recovery_NDCG@20",
    "allwarm_adaptation_recovery_NDCG@20",
    "stale_u_exact_equal_base",
    "stale_u_max_abs_diff_base",
    "stale_v_exact_equal_base",
    "stale_v_max_abs_diff_base",
    "online_v_exact_equal_base",
    "online_v_max_abs_diff_base",
    "full_v_exact_equal_base",
    "full_v_max_abs_diff_base",
    "online_maps_exact",
    "full_maps_exact",
]

for metric in QUALITY_METRICS:
    STEP_FIELDS.extend(
        [
            f"stale_{metric}",
            f"online_{metric}",
            f"full_{metric}",
            f"online_minus_stale_{metric}",
            f"online_minus_full_{metric}",
            f"full_minus_stale_{metric}",
            f"allwarm_stale_{metric}",
            f"allwarm_online_{metric}",
            f"allwarm_full_{metric}",
            f"allwarm_online_minus_stale_{metric}",
            f"allwarm_online_minus_full_{metric}",
            f"allwarm_full_minus_stale_{metric}",
        ]
    )


TRIAL_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "seed",
    "n_eval_points",
    "n_base_rows",
    "n_base_users",
    "n_base_items",
    "base_train_time_s",
    "total_online_update_time_s",
    "total_full_retrain_time_s",
    "total_online_construction_inclusive_time_s",
    "total_full_construction_inclusive_time_s",
    "speedup_full_over_online",
    "online_full_cost_fraction",
    "construction_inclusive_speedup_full_over_online",
    "construction_inclusive_online_full_cost_fraction",
    "mean_adaptation_recovery_NDCG@20",
    "n_valid_adaptation_recovery_points",
    "mean_allwarm_adaptation_recovery_NDCG@20",
    "n_valid_allwarm_adaptation_recovery_points",
    "all_stale_u_exact_equal_base",
    "max_stale_u_abs_diff_base",
    "all_stale_v_exact_equal_base",
    "max_stale_v_abs_diff_base",
    "all_online_v_exact_equal_base",
    "max_online_v_abs_diff_base",
]

for metric in QUALITY_METRICS:
    TRIAL_FIELDS.extend(
        [
            f"mean_stale_{metric}",
            f"mean_online_{metric}",
            f"mean_full_{metric}",
            f"mean_online_minus_stale_{metric}",
            f"mean_online_minus_full_{metric}",
            f"mean_full_minus_stale_{metric}",
            f"positive_online_minus_stale_{metric}",
            f"positive_online_minus_full_{metric}",
            f"positive_full_minus_stale_{metric}",
            f"mean_allwarm_stale_{metric}",
            f"mean_allwarm_online_{metric}",
            f"mean_allwarm_full_{metric}",
            f"mean_allwarm_online_minus_stale_{metric}",
            f"mean_allwarm_online_minus_full_{metric}",
            f"mean_allwarm_full_minus_stale_{metric}",
        ]
    )


SUMMARY_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "n_trials",
    "n_paired_points",
    "seeds",
    "target_base_fraction",
    "effective_base_fraction",
    "target_base_rows",
    "effective_base_rows",
    "boundary_tie_rows_added_to_base",
    "n_future_raw",
    "n_future_warm",
    "warm_start_fraction",
    "n_warm_chunk_1",
    "n_warm_chunk_2",
    "n_warm_chunk_3",
    "n_warm_chunk_4",
    "n_primary_eval_rows_point_1",
    "n_primary_eval_rows_point_2",
    "n_primary_eval_rows_point_3",
    "n_excluded_unknown_user",
    "n_excluded_unknown_item",
    "n_excluded_unknown_both",
    "mean_base_train_time_s",
    "std_base_train_time_s",
    "mean_total_online_update_time_s",
    "std_total_online_update_time_s",
    "mean_total_full_retrain_time_s",
    "std_total_full_retrain_time_s",
    "mean_total_online_construction_inclusive_time_s",
    "std_total_online_construction_inclusive_time_s",
    "mean_total_full_construction_inclusive_time_s",
    "std_total_full_construction_inclusive_time_s",
    "mean_speedup_full_over_online",
    "std_speedup_full_over_online",
    "mean_online_full_cost_fraction",
    "std_online_full_cost_fraction",
    "mean_construction_inclusive_speedup_full_over_online",
    "std_construction_inclusive_speedup_full_over_online",
    "mean_construction_inclusive_online_full_cost_fraction",
    "std_construction_inclusive_online_full_cost_fraction",
    "mean_adaptation_recovery_NDCG@20",
    "std_adaptation_recovery_NDCG@20",
    "n_valid_adaptation_recovery_points",
    "mean_allwarm_adaptation_recovery_NDCG@20",
    "std_allwarm_adaptation_recovery_NDCG@20",
    "n_valid_allwarm_adaptation_recovery_points",
    "all_stale_u_exact_equal_base",
    "max_stale_u_abs_diff_base",
    "all_stale_v_exact_equal_base",
    "max_stale_v_abs_diff_base",
    "all_online_v_exact_equal_base",
    "max_online_v_abs_diff_base",
    "mean_online_minus_stale_NDCG@20_eval_point_1",
    "mean_online_minus_stale_NDCG@20_eval_point_2",
    "mean_online_minus_stale_NDCG@20_eval_point_3",
    "mean_online_minus_full_NDCG@20_eval_point_1",
    "mean_online_minus_full_NDCG@20_eval_point_2",
    "mean_online_minus_full_NDCG@20_eval_point_3",
    "mean_full_minus_stale_NDCG@20_eval_point_1",
    "mean_full_minus_stale_NDCG@20_eval_point_2",
    "mean_full_minus_stale_NDCG@20_eval_point_3",
]

for metric in QUALITY_METRICS:
    SUMMARY_FIELDS.extend(
        [
            f"mean_stale_{metric}",
            f"std_stale_{metric}",
            f"mean_online_{metric}",
            f"std_online_{metric}",
            f"mean_full_{metric}",
            f"std_full_{metric}",
            f"mean_online_minus_stale_{metric}",
            f"std_online_minus_stale_{metric}",
            f"median_online_minus_stale_{metric}",
            f"min_online_minus_stale_{metric}",
            f"max_online_minus_stale_{metric}",
            f"positive_online_minus_stale_{metric}",
            f"mean_online_minus_full_{metric}",
            f"std_online_minus_full_{metric}",
            f"positive_online_minus_full_{metric}",
            f"mean_full_minus_stale_{metric}",
            f"std_full_minus_stale_{metric}",
            f"positive_full_minus_stale_{metric}",
            f"mean_allwarm_stale_{metric}",
            f"std_allwarm_stale_{metric}",
            f"mean_allwarm_online_{metric}",
            f"std_allwarm_online_{metric}",
            f"mean_allwarm_full_{metric}",
            f"std_allwarm_full_{metric}",
            f"mean_allwarm_online_minus_stale_{metric}",
            f"std_allwarm_online_minus_stale_{metric}",
            f"positive_allwarm_online_minus_stale_{metric}",
            f"mean_allwarm_online_minus_full_{metric}",
            f"std_allwarm_online_minus_full_{metric}",
            f"positive_allwarm_online_minus_full_{metric}",
            f"mean_allwarm_full_minus_stale_{metric}",
            f"std_allwarm_full_minus_stale_{metric}",
            f"positive_allwarm_full_minus_stale_{metric}",
        ]
    )


# ============================================================
# CLI / logging
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen final H1-H3 experiment: IBPR_STALE vs "
            "OnlineIBPRMejorado O014 vs IBPR_FULL_RETRAIN on the globally "
            "chronological post-development MovieLens 1M stream."
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
        help="Validate and print the frozen protocol without training any model.",
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


# ============================================================
# Generic utilities
# ============================================================


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


def mean_std(values):
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        raise ValueError("No se puede agregar una colección vacía.")
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_value, std_value


def finite_values(values):
    out = []
    for value in values:
        if value in (None, ""):
            continue
        parsed = float(value)
        if np.isfinite(parsed):
            out.append(parsed)
    return out


def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def maps_exact(dataset_or_model, uid_map, iid_map):
    return bool(
        getattr(dataset_or_model, "uid_map", None) == uid_map
        and getattr(dataset_or_model, "iid_map", None) == iid_map
    )


def rows_pairs(rows):
    return {(row[0], row[1]) for row in rows}


def assert_unique_user_item_pairs(rows, label):
    """Reject silent duplicate (user,item) observations before Cornac can collapse them."""
    n_rows = len(rows)
    n_pairs = len(rows_pairs(rows))
    if n_pairs != n_rows:
        raise RuntimeError(
            f"{label}: contiene pares (u,i) duplicados: rows={n_rows}, unique_pairs={n_pairs}."
        )


def assert_dataset_row_count(dataset, expected_rows, label):
    """
    Ensure Dataset.build did not silently change the effective interaction count.
    For this implicit protocol every row must correspond to one unique nonzero (u,i).
    """
    expected_rows = int(expected_rows)
    actual_nnz = int(dataset.csr_matrix.nnz)
    if actual_nnz != expected_rows:
        raise RuntimeError(
            f"{label}: Dataset.build cambió el número efectivo de interacciones: "
            f"expected={expected_rows}, csr_nnz={actual_nnz}."
        )

    try:
        actual_uir = len(dataset.uir_tuple[0])
    except Exception:
        actual_uir = actual_nnz

    if int(actual_uir) != expected_rows:
        raise RuntimeError(
            f"{label}: uir_tuple no conserva el número esperado de filas: "
            f"expected={expected_rows}, uir_rows={actual_uir}."
        )


def ts_min(rows):
    return int(min(row[3] for row in rows))


def ts_max(rows):
    return int(max(row[3] for row in rows))


# ============================================================
# Protocol fingerprint / implementation guards
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


def dataset_sha256(rows):
    digest = hashlib.sha256()
    for u, i, value, timestamp, original_position in rows:
        payload = (
            f"{u}\t{i}\t{float(value):.1f}\t{int(timestamp)}\t"
            f"{int(original_position)}\n"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def protocol_payload(data_hash):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "MovieLens 1M",
        "variant": VARIANT,
        "rating_threshold": RATING_THRESHOLD,
        "feedback": "rating>=3 -> 1.0; lower ratings absent",
        "target_base_fraction": TARGET_BASE_FRAC,
        "base_split_rule": BASE_SPLIT_RULE,
        "n_stream_chunks": N_STREAM_CHUNKS,
        "chunk_split_rule": CHUNK_SPLIT_RULE,
        "warm_start_rule": "future row eligible iff user and item exist in effective base",
        "primary_eval_rule": PRIMARY_EVAL_RULE,
        "prequential_sequence": [[1, 2], [2, 3], [3, 4]],
        "final_seeds": FINAL_SEEDS,
        "ibpr_config": FROZEN_IBPR_CONFIG,
        "online_config": FROZEN_ONLINE_CONFIG,
        "full_retrain": "fresh IBPR R900 from scratch on accumulated warm history",
        "full_retrain_seed_policy": "same trial seed reset before each full fit",
        "online_seed_policy": "seed + partial_update_count",
        "quality_metrics": QUALITY_METRICS,
        "ranking_rating_threshold": 1.0,
        "ranking_exclude_unknowns": True,
        "primary_metric": f"NDCG@{TOP_K}",
        "data_sha256": data_hash,
        "script_sha256": _script_sha256(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cornac_version": getattr(cornac, "__version__", "unavailable"),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "scipy_version": scipy.__version__,
        "dataset_build_sha256": _source_sha256(Dataset.build),
        "ranking_eval_sha256": _source_sha256(ranking_eval),
        "ibpr_wrapper_sha256": _source_sha256(IBPR),
        "ibpr_core_sha256": _source_sha256(ibpr_core),
        "online_wrapper_sha256": _source_sha256(OnlineIBPRMejorado),
        "online_core_sha256": _source_sha256(online_core),
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
    missing = [
        idx
        for idx, row in enumerate(rows)
        if not row.get("protocol_version")
        or not row.get("protocol_hash")
        or not row.get("data_sha256")
    ]
    if missing:
        raise RuntimeError(
            f"{path}: contiene filas sin fingerprint. Usa un timestamp nuevo."
        )

    versions = {str(row["protocol_version"]) for row in rows}
    hashes = {str(row["protocol_hash"]) for row in rows}
    data_hashes = {str(row["data_sha256"]) for row in rows}

    if versions != {PROTOCOL_VERSION} or hashes != {protocol_hash} or data_hashes != {data_hash}:
        raise RuntimeError(
            f"{path}: fingerprint incompatible. versions={sorted(versions)}, "
            f"protocol_hashes={sorted(hashes)}, data_hashes={sorted(data_hashes)}. "
            "No mezcles ejecuciones; usa un timestamp nuevo."
        )


def validate_implementation_contracts():
    online_ctor = set(inspect.signature(OnlineIBPRMejorado.__init__).parameters)
    online_partial = set(inspect.signature(OnlineIBPRMejorado.partial_fit_recent).parameters)

    if "seed" not in online_ctor:
        raise RuntimeError("OnlineIBPRMejorado obsoleto: falta seed en constructor.")

    required_partial = {"recent_pairs", "history_csr", "max_steps", "n_epochs"}
    missing = required_partial - online_partial
    if missing:
        raise RuntimeError(
            "OnlineIBPRMejorado obsoleto: contrato partial_fit_recent incompleto. "
            f"Faltan {sorted(missing)}"
        )

    online_source = inspect.getsource(online_core)
    expected_online_tokens = [
        "_sample_negatives_uniform",
        "history_csr",
        "recent_pairs",
        "update_V",
        "torch.nn.functional.normalize",
    ]
    missing_online = [token for token in expected_online_tokens if token not in online_source]
    if missing_online:
        raise RuntimeError(
            "El core OnlineIBPRMejorado no coincide con la versión estabilizada. "
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
            "El core IBPR no coincide con la implementación R900 auditada. "
            f"Tokens ausentes: {missing_ibpr}"
        )

    # Tiny stabilized Online contract smoke test.
    U0 = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    V0 = np.asarray([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    history = csr_matrix(
        np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    )
    recent = np.asarray([[0, 1]], dtype=np.int64)

    result = online_core(
        train_set=None,
        k=2,
        lamda=1e-4,
        n_epochs=1,
        learning_rate=0.001,
        batch_size=1,
        init_params={"U": U0.copy(), "V": V0.copy()},
        update_V=False,
        neg_sampling="uniform",
        normalize=True,
        verbose=False,
        recent_pairs=recent,
        history_csr=history,
        max_steps=None,
        random_seed=17,
        loss_mode="cosine_bpr",
    )

    if not np.array_equal(np.asarray(result["V"]), V0):
        raise RuntimeError("Contrato Online inválido: update_V=False modificó V.")

    try:
        online_core(
            train_set=None,
            k=2,
            init_params={"U": U0.copy(), "V": V0.copy()},
            update_V=False,
            recent_pairs=recent,
            history_csr=history,
            max_steps=0,
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("Contrato Online inválido: max_steps=0 no fue rechazado.")

    wrapper = OnlineIBPRMejorado(
        k=2,
        max_iter=1,
        learning_rate=0.001,
        lamda=1e-4,
        batch_size=1,
        init_params={"U": U0.copy(), "V": V0.copy()},
        update_V=False,
        neg_sampling="uniform",
        normalize=True,
        loss_mode="cosine_bpr",
        seed=17,
        trainable=True,
        verbose=False,
        name="final_h1_h3_contract_check",
    )
    wrapper.num_users, wrapper.num_items = history.shape

    count_before = int(wrapper._partial_update_count)
    U_before = np.asarray(wrapper.U).copy()
    V_before = np.asarray(wrapper.V).copy()

    wrapper.partial_fit_recent(
        recent_pairs=np.empty((0, 2), dtype=np.int64),
        history_csr=history,
        max_steps=None,
        n_epochs=1,
    )
    if int(wrapper._partial_update_count) != count_before:
        raise RuntimeError("Contrato Online inválido: update vacío consumió seed.")
    if not np.array_equal(wrapper.U, U_before) or not np.array_equal(wrapper.V, V_before):
        raise RuntimeError("Contrato Online inválido: update vacío modificó factores.")

    wrapper.partial_fit_recent(
        recent_pairs=recent,
        history_csr=history,
        max_steps=None,
        n_epochs=1,
    )
    if int(wrapper._partial_update_count) != count_before + 1:
        raise RuntimeError("Contrato Online inválido: progresión de seed incorrecta.")
    if not np.array_equal(wrapper.V, V_before):
        raise RuntimeError("Contrato Online inválido: wrapper modificó V congelada.")

    print("Implementation contracts: OK")
    print(f"  IBPR wrapper     : {inspect.getsourcefile(IBPR) or 'unknown'}")
    print(f"  IBPR core        : {inspect.getsourcefile(ibpr_core) or 'unknown'}")
    print(f"  Online wrapper   : {inspect.getsourcefile(OnlineIBPRMejorado) or 'unknown'}")
    print(f"  Online core      : {inspect.getsourcefile(online_core) or 'unknown'}")
    print("  Online V invariant / seed progression / empty update: OK")
    print()


# ============================================================
# Data preparation
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
        raise ValueError("El ajuste por empate de timestamp consumió todo el future stream.")

    base_rows = list(rows[:effective])
    future_rows = list(rows[effective:])

    if ts_max(base_rows) >= ts_min(future_rows):
        raise RuntimeError(
            "El corte base/future no es estrictamente cronológico después del ajuste."
        )

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
            raise ValueError(f"No se pudo crear un límite interno válido en {label}.")

        effective = target
        boundary_ts = rows[target - 1][3]
        while effective < n and rows[effective][3] == boundary_ts:
            effective += 1

        if effective >= n:
            raise ValueError(
                f"Un empate de timestamp en {label} consumiría todo el resto del stream."
            )
        if effective <= boundaries[-1]:
            raise RuntimeError(f"Límite de chunk no creciente en {label}.")

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
        list(rows[boundaries[idx] : boundaries[idx + 1]])
        for idx in range(n_chunks)
    ]

    for idx in range(n_chunks - 1):
        if ts_max(chunks[idx]) >= ts_min(chunks[idx + 1]):
            raise RuntimeError(
                f"Chunks {idx+1}/{idx+2} de {label} no tienen separación temporal estricta."
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
        raise ValueError("El future stream quedó vacío después del filtrado warm-start.")

    n_future_raw = len(future_rows)
    n_future_warm = len(warm_rows)
    excluded_total = excluded_user + excluded_item + excluded_both

    if n_future_raw != n_future_warm + excluded_total:
        raise RuntimeError("Conteo warm-start inconsistente.")

    return {
        "base_users": base_users,
        "base_items": base_items,
        "warm_future_rows": warm_rows,
        "n_future_raw": n_future_raw,
        "n_future_warm": n_future_warm,
        "warm_start_fraction": n_future_warm / n_future_raw,
        "n_excluded_unknown_user": excluded_user,
        "n_excluded_unknown_item": excluded_item,
        "n_excluded_unknown_both": excluded_both,
    }


def primary_eval_rows_for_step(known_chunks, step_idx):
    """
    Primary H1-H3 population: eval rows whose user has already received at
    least one online update in chunks 0..step_idx.
    """
    users_exposed_to_update = {
        row[0]
        for chunk in known_chunks[: step_idx + 1]
        for row in chunk
    }
    allwarm_eval_rows = list(known_chunks[step_idx + 1])
    primary_rows = [row for row in allwarm_eval_rows if row[0] in users_exposed_to_update]
    if not primary_rows:
        raise ValueError(
            f"eval_point={step_idx+1}: población primaria adaptada quedó vacía."
        )
    return primary_rows, allwarm_eval_rows, users_exposed_to_update


def build_primary_eval_plan(known_chunks):
    plan = []
    for step_idx in range(3):
        primary_rows, allwarm_rows, users_exposed_to_update = primary_eval_rows_for_step(
            known_chunks, step_idx
        )
        primary_users = {row[0] for row in primary_rows}
        allwarm_users = {row[0] for row in allwarm_rows}
        plan.append(
            {
                "eval_point": step_idx + 1,
                "update_chunk": step_idx + 1,
                "eval_chunk": step_idx + 2,
                "n_users_exposed_to_update": len(users_exposed_to_update),
                "n_primary_rows": len(primary_rows),
                "n_primary_users": len(primary_users),
                "n_primary_items": len({row[1] for row in primary_rows}),
                "n_allwarm_rows": len(allwarm_rows),
                "n_allwarm_users": len(allwarm_users),
                "n_allwarm_items": len({row[1] for row in allwarm_rows}),
                "n_allwarm_unadapted_users": len(allwarm_users - users_exposed_to_update),
                "primary_fraction_of_allwarm_rows": len(primary_rows) / len(allwarm_rows),
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

    # Dataset-level pair uniqueness/leakage guard.
    assert_unique_user_item_pairs(all_rows, "all_positive_rows")
    assert_unique_user_item_pairs(split["base_rows"], "base_rows")
    for idx, chunk in enumerate(known_chunks, start=1):
        assert_unique_user_item_pairs(chunk, f"warm_chunk_{idx}")

    base_pairs = rows_pairs(split["base_rows"])
    observed = set(base_pairs)
    for idx, chunk in enumerate(known_chunks, start=1):
        chunk_pairs = rows_pairs(chunk)
        overlap = observed & chunk_pairs
        if overlap:
            sample = next(iter(overlap))
            raise RuntimeError(
                f"Par (u,i) repetido entre historial y chunk {idx}: {sample}. "
                "El protocolo final no permite evaluar pares ya observados."
            )
        observed.update(chunk_pairs)

    primary_plan = build_primary_eval_plan(known_chunks)

    return {
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


def print_plan(data, protocol_hash):
    all_rows = data["all_rows"]
    base_rows = data["base_rows"]
    future_rows = data["future_rows"]

    print("=" * 118)
    print("FINAL H1-H3 - ONLINEIBPRMEJORADO")
    print("=" * 118)
    print(f"Dataset                         : MovieLens 1M")
    print(f"Positive threshold              : rating >= {RATING_THRESHOLD}")
    print("Feedback                        : implicit positive = 1.0")
    print(f"Total implicit-positive rows    : {len(all_rows):,}")
    print(f"Data SHA256                     : {data['data_sha256']}")
    print(f"Protocol version                : {PROTOCOL_VERSION}")
    print(f"Protocol hash                   : {protocol_hash}")
    print()
    print("Environment fingerprint:")
    print(f"  Python                        : {sys.version.split()[0]}")
    print(f"  Platform                      : {platform.platform()}")
    print(f"  Cornac                        : {getattr(cornac, '__version__', 'unavailable')}")
    print(f"  NumPy                         : {np.__version__}")
    print(f"  PyTorch                       : {torch.__version__}")
    print(f"  SciPy                         : {scipy.__version__}")
    print()

    print("Frozen configurations:")
    print(f"  R900: {FROZEN_IBPR_CONFIG}")
    print(f"  O014: {FROZEN_ONLINE_CONFIG}")
    print(f"  Final seeds: {FINAL_SEEDS}")
    print()

    print("Base/future split:")
    print(f"  Target base fraction          : {TARGET_BASE_FRAC:.2%}")
    print(f"  Target base rows              : {data['target_base_rows']:,}")
    print(f"  Effective base rows           : {data['effective_base_rows']:,}")
    print(
        "  Effective base fraction       : "
        f"{data['effective_base_rows']/len(all_rows):.6%}"
    )
    print(
        "  Tie rows moved into base      : "
        f"{data['boundary_tie_rows_added_to_base']:,}"
    )
    print(f"  Base timestamp range          : {ts_min(base_rows)} .. {ts_max(base_rows)}")
    print(f"  Future timestamp range        : {ts_min(future_rows)} .. {ts_max(future_rows)}")
    print(
        "  Strict base < future          : "
        f"{ts_max(base_rows) < ts_min(future_rows)}"
    )
    print(f"  Base users/items              : {len(data['base_users']):,} / {len(data['base_items']):,}")
    print()

    print("Warm-start future universe:")
    print(f"  Future raw rows               : {data['n_future_raw']:,}")
    print(f"  Future warm rows              : {data['n_future_warm']:,}")
    print(f"  Warm-start fraction           : {data['warm_start_fraction']:.4%}")
    print(f"  Excluded unknown user only    : {data['n_excluded_unknown_user']:,}")
    print(f"  Excluded unknown item only    : {data['n_excluded_unknown_item']:,}")
    print(f"  Excluded unknown both         : {data['n_excluded_unknown_both']:,}")
    print()

    print("Global chronological WARM chunks (filter first, split second):")
    for idx, warm_chunk in enumerate(data["known_chunks"], start=1):
        print(
            f"  chunk_{idx}: warm={len(warm_chunk):,} | "
            f"users={len({r[0] for r in warm_chunk}):,} | "
            f"items={len({r[1] for r in warm_chunk}):,} | "
            f"ts={ts_min(warm_chunk)}..{ts_max(warm_chunk)}"
        )

    print("  Internal WARM boundary adjustments:")
    for adj in data["warm_chunk_adjustments"]:
        print(
            f"    boundary {adj['boundary']}: target={adj['target']:,} -> "
            f"effective={adj['effective']:,} | shifted={adj['rows_shifted']:,} "
            f"| timestamp={adj['timestamp']}"
        )
    print()

    print("Prequential sequence and primary adaptation population:")
    for item in data["primary_eval_plan"]:
        print(
            f"  update chunk{item['update_chunk']} -> eval chunk{item['eval_chunk']} | "
            f"users_exposed_to_update={item['n_users_exposed_to_update']:,} | "
            f"PRIMARY rows={item['n_primary_rows']:,}, users={item['n_primary_users']:,}, "
            f"items={item['n_primary_items']:,} "
            f"({item['primary_fraction_of_allwarm_rows']:.2%} of all-warm eval rows) | "
            f"ALL-WARM rows={item['n_allwarm_rows']:,}, users={item['n_allwarm_users']:,} | "
            f"unadapted_eval_users={item['n_allwarm_unadapted_users']:,}"
        )
    print("  PRIMARY metrics drive H1/H3; ALL-WARM metrics are supplementary diagnostics.")
    print()

    print("Branches:")
    print("  IBPR_STALE")
    print("  OnlineIBPRMejorado O014")
    print("  IBPR_FULL_RETRAIN R900 from scratch on accumulated history")
    print()

    print(f"Primary metric                  : NDCG@{TOP_K}")
    print(f"All metrics                     : {QUALITY_METRICS}")
    print("H2 primary timing               : partial_fit_recent vs IBPR.fit only")
    print("H2 supplementary timing         : history-build+partial vs full-build+fit")
    print("Expected paired points          : 2 seeds x 3 eval points = 6")
    print("Retuning after this point       : PROHIBITED")
    print()


# ============================================================
# Datasets / evaluation
# ============================================================


def cornac_rows(rows):
    return [(u, i, float(value), int(timestamp)) for u, i, value, timestamp, _ in rows]


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
            f"({dataset.num_users},{dataset.num_items}) != ({n_users},{n_items})"
        )
    if dataset.uid_map != uid_map or dataset.iid_map != iid_map:
        raise RuntimeError(f"{label}: uid_map/iid_map no coinciden con el base.")


def rows_to_pairs(rows, uid_map, iid_map):
    pairs = np.asarray(
        [[uid_map[row[0]], iid_map[row[1]]] for row in rows],
        dtype=np.int64,
    )
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("recent_pairs debe tener shape (n,2).")
    return pairs


def build_metrics():
    return [
        cornac.metrics.AUC(),
        cornac.metrics.MAP(),
        cornac.metrics.NDCG(k=TOP_K),
        cornac.metrics.Precision(k=TOP_K),
        cornac.metrics.Recall(k=TOP_K),
    ]


def evaluate_model(model, train_set, test_set):
    metrics = build_metrics()
    avg_results, _ = ranking_eval(
        model=model,
        metrics=metrics,
        train_set=train_set,
        test_set=test_set,
        val_set=None,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
    )
    return OrderedDict(
        (metric.name, float(value))
        for metric, value in zip(metrics, avg_results)
    )


# ============================================================
# Model construction / invariants
# ============================================================


def train_base_model(base_rows, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    base_train_set = build_dataset(base_rows, seed=seed, exclude_unknowns=False)
    assert_dataset_row_count(base_train_set, len(base_rows), "base_train_set")

    model = IBPR(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=FROZEN_IBPR_CONFIG["max_iter"],
        learning_rate=FROZEN_IBPR_CONFIG["learning_rate"],
        lamda=FROZEN_IBPR_CONFIG["lamda"],
        batch_size=FROZEN_IBPR_CONFIG["batch_size"],
        verbose=False,
        name=f"IBPR_R900_base_seed{seed}",
    )

    start = time.perf_counter()
    model.fit(base_train_set)
    elapsed = time.perf_counter() - start

    if model.U.shape != (base_train_set.num_users, FROZEN_IBPR_CONFIG["k"]):
        raise RuntimeError("U_base shape inválida.")
    if model.V.shape != (base_train_set.num_items, FROZEN_IBPR_CONFIG["k"]):
        raise RuntimeError("V_base shape inválida.")

    return model, base_train_set, elapsed


def initialize_stale(base_train_set, base_u, base_v, seed):
    model = IBPR(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=FROZEN_IBPR_CONFIG["max_iter"],
        learning_rate=FROZEN_IBPR_CONFIG["learning_rate"],
        lamda=FROZEN_IBPR_CONFIG["lamda"],
        batch_size=FROZEN_IBPR_CONFIG["batch_size"],
        trainable=False,
        init_params={"U": base_u.copy(), "V": base_v.copy()},
        verbose=False,
        name=f"IBPR_STALE_seed{seed}",
    )
    model.fit(base_train_set)
    if not np.array_equal(model.U, base_u) or not np.array_equal(model.V, base_v):
        raise RuntimeError("Inicialización Stale modificó los factores base.")
    return model


def initialize_online(base_train_set, base_u, base_v, seed):
    model = OnlineIBPRMejorado(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=FROZEN_ONLINE_CONFIG["n_epochs"],
        learning_rate=FROZEN_ONLINE_CONFIG["learning_rate"],
        lamda=FROZEN_ONLINE_CONFIG["lamda"],
        batch_size=FROZEN_ONLINE_CONFIG["batch_size"],
        trainable=False,
        init_params={"U": base_u.copy(), "V": base_v.copy()},
        update_V=FROZEN_ONLINE_CONFIG["update_V"],
        neg_sampling=FROZEN_ONLINE_CONFIG["neg_sampling"],
        normalize=FROZEN_ONLINE_CONFIG["normalize"],
        loss_mode=FROZEN_ONLINE_CONFIG["loss_mode"],
        seed=seed,
        verbose=False,
        name=f"OnlineIBPRMejorado_O014_seed{seed}",
    )
    # Populate recommender metadata without retraining.
    model.fit(base_train_set)
    model.trainable = True

    if not np.array_equal(model.U, base_u) or not np.array_equal(model.V, base_v):
        raise RuntimeError("Inicialización Online modificó los factores base.")
    if model.uid_map != base_train_set.uid_map or model.iid_map != base_train_set.iid_map:
        raise RuntimeError("Inicialización Online perdió los mapas del base.")
    return model


def train_full_retrain(accumulated_rows, uid_map, iid_map, n_users, n_items, seed):
    start_build = time.perf_counter()
    full_train_set = build_dataset(
        accumulated_rows,
        uid_map=uid_map,
        iid_map=iid_map,
        seed=seed,
        exclude_unknowns=False,
    )
    build_time = time.perf_counter() - start_build
    assert_dataset_contract(
        full_train_set, uid_map, iid_map, n_users, n_items, "full_train_set"
    )
    assert_dataset_row_count(
        full_train_set, len(accumulated_rows), "full_train_set"
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
        name=f"IBPR_FULL_RETRAIN_seed{seed}",
    )

    start_fit = time.perf_counter()
    model.fit(full_train_set)
    fit_time = time.perf_counter() - start_fit

    if model.U.shape != (n_users, FROZEN_IBPR_CONFIG["k"]):
        raise RuntimeError("Full Retrain U shape incompatible con universo base.")
    if model.V.shape != (n_items, FROZEN_IBPR_CONFIG["k"]):
        raise RuntimeError("Full Retrain V shape incompatible con universo base.")
    if model.uid_map != uid_map or model.iid_map != iid_map:
        raise RuntimeError("Full Retrain reconstruyó mapas incompatibles.")

    return model, build_time, fit_time


# ============================================================
# Trial execution
# ============================================================


def run_seed_trial(seed, data, protocol_hash):
    print()
    print("=" * 118)
    print(f"RUN FINAL H1-H3 | seed={seed}")
    print("=" * 118)

    base_model, base_train_set, base_train_time = train_base_model(
        data["base_rows"], seed
    )
    uid_map = dict(base_train_set.uid_map)
    iid_map = dict(base_train_set.iid_map)
    n_users = base_train_set.num_users
    n_items = base_train_set.num_items

    base_u = np.asarray(base_model.U).copy()
    base_v = np.asarray(base_model.V).copy()

    stale_model = initialize_stale(base_train_set, base_u, base_v, seed)
    online_model = initialize_online(base_train_set, base_u, base_v, seed)

    print(
        f"BASE trained: rows={len(data['base_rows']):,} | users={n_users:,} | "
        f"items={n_items:,} | time={base_train_time:.2f}s"
    )

    observed_rows = list(data["base_rows"])
    cumulative_online = 0.0
    cumulative_full = 0.0
    cumulative_online_construction_inclusive = 0.0
    cumulative_full_construction_inclusive = 0.0
    step_rows = []

    for step_idx in range(3):
        update_chunk_number = step_idx + 1
        eval_chunk_number = step_idx + 2
        eval_point = step_idx + 1

        update_rows = list(data["known_chunks"][step_idx])
        eval_rows, allwarm_eval_rows, users_exposed_to_update = primary_eval_rows_for_step(
            data["known_chunks"], step_idx
        )

        if not update_rows or not eval_rows or not allwarm_eval_rows:
            raise RuntimeError("Update/eval chunk vacío durante ejecución.")

        # Strict temporal and pair-level leakage guards.  The all-warm eval
        # population spans the entire evaluation chunk; the primary population
        # is a subset selected only by prior user adaptation.
        if ts_max(update_rows) >= ts_min(allwarm_eval_rows):
            raise RuntimeError(
                f"step {eval_point}: update/eval no son estrictamente temporales."
            )

        eval_pairs = rows_pairs(allwarm_eval_rows)
        observed_before_pairs = rows_pairs(observed_rows)
        if eval_pairs & observed_before_pairs:
            sample = next(iter(eval_pairs & observed_before_pairs))
            raise RuntimeError(
                f"step {eval_point}: eval pair ya estaba observado antes del update: {sample}"
            )

        post_update_rows = observed_rows + update_rows
        post_update_pairs = rows_pairs(post_update_rows)
        if eval_pairs & post_update_pairs:
            sample = next(iter(eval_pairs & post_update_pairs))
            raise RuntimeError(
                f"step {eval_point}: eval pair aparece en historial post-update: {sample}"
            )

        # Common post-update history. This construction is required by Online
        # for history_csr negative sampling and also reused by all branches in evaluation.
        start_history_build = time.perf_counter()
        eval_train_set = build_dataset(
            post_update_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=False,
        )
        history_build_time = time.perf_counter() - start_history_build

        # Evaluation datasets are not part of either branch's update cost.
        start_test_build = time.perf_counter()
        test_set = build_dataset(
            eval_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=True,
        )
        allwarm_test_set = build_dataset(
            allwarm_eval_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=True,
        )
        test_build_time = time.perf_counter() - start_test_build

        assert_dataset_contract(
            eval_train_set, uid_map, iid_map, n_users, n_items, "eval_train_set"
        )
        assert_dataset_contract(test_set, uid_map, iid_map, n_users, n_items, "test_set")
        assert_dataset_contract(
            allwarm_test_set, uid_map, iid_map, n_users, n_items, "allwarm_test_set"
        )
        assert_dataset_row_count(
            eval_train_set, len(post_update_rows), "eval_train_set"
        )
        assert_dataset_row_count(test_set, len(eval_rows), "test_set")
        assert_dataset_row_count(
            allwarm_test_set, len(allwarm_eval_rows), "allwarm_test_set"
        )

        recent_pairs = rows_to_pairs(update_rows, uid_map, iid_map)

        # Online partial update: only this call is timed for H2.
        online_v_before = np.asarray(online_model.V).copy()
        start_online = time.perf_counter()
        online_model.partial_fit_recent(
            recent_pairs=recent_pairs,
            history_csr=eval_train_set.csr_matrix,
            max_steps=FROZEN_ONLINE_CONFIG["max_steps"],
            n_epochs=FROZEN_ONLINE_CONFIG["n_epochs"],
        )
        online_time = time.perf_counter() - start_online
        cumulative_online += online_time

        if not np.array_equal(online_model.V, online_v_before):
            raise RuntimeError(
                f"step {eval_point}: Online modificó V respecto del estado anterior."
            )
        online_v_exact = np.array_equal(online_model.V, base_v)
        online_v_diff = max_abs_diff(online_model.V, base_v)
        if not online_v_exact or online_v_diff != 0.0:
            raise RuntimeError(
                f"step {eval_point}: invariante V_online==V_base violado, "
                f"max_diff={online_v_diff}"
            )
        if online_model.num_users != n_users or online_model.num_items != n_items:
            raise RuntimeError("Online cambió dimensiones del universo.")
        online_maps_ok = online_model.uid_map == uid_map and online_model.iid_map == iid_map
        if not online_maps_ok:
            raise RuntimeError("Online perdió uid_map/iid_map congelados.")

        # Stale must remain bit-exact from base.
        stale_u_exact = np.array_equal(stale_model.U, base_u)
        stale_v_exact = np.array_equal(stale_model.V, base_v)
        stale_u_diff = max_abs_diff(stale_model.U, base_u)
        stale_v_diff = max_abs_diff(stale_model.V, base_v)
        if not stale_u_exact or not stale_v_exact:
            raise RuntimeError(f"step {eval_point}: rama Stale modificó factores.")

        # Full retrain from scratch: only fit() is timed for primary H2.
        full_model, full_build_time, full_time = train_full_retrain(
            post_update_rows,
            uid_map,
            iid_map,
            n_users,
            n_items,
            seed,
        )
        cumulative_full += full_time

        online_construction_inclusive = history_build_time + online_time
        full_construction_inclusive = full_build_time + full_time
        cumulative_online_construction_inclusive += online_construction_inclusive
        cumulative_full_construction_inclusive += full_construction_inclusive

        full_maps_ok = full_model.uid_map == uid_map and full_model.iid_map == iid_map
        if not full_maps_ok:
            raise RuntimeError("Full Retrain no preservó los mapas congelados.")

        # Same PRIMARY evaluation set for all branches.
        stale_metrics = evaluate_model(stale_model, eval_train_set, test_set)
        online_metrics = evaluate_model(online_model, eval_train_set, test_set)
        full_metrics = evaluate_model(full_model, eval_train_set, test_set)

        # Supplementary diagnostic over all warm-start rows in the eval chunk,
        # including users that have not yet received a new online interaction.
        allwarm_stale_metrics = evaluate_model(
            stale_model, eval_train_set, allwarm_test_set
        )
        allwarm_online_metrics = evaluate_model(
            online_model, eval_train_set, allwarm_test_set
        )
        allwarm_full_metrics = evaluate_model(
            full_model, eval_train_set, allwarm_test_set
        )

        # Re-check invariants after evaluation.
        if not np.array_equal(stale_model.U, base_u) or not np.array_equal(stale_model.V, base_v):
            raise RuntimeError("Evaluación modificó la rama Stale.")
        if not np.array_equal(online_model.V, base_v):
            raise RuntimeError("Evaluación modificó V_online.")

        ndcg_name = f"NDCG@{TOP_K}"
        denom = full_metrics[ndcg_name] - stale_metrics[ndcg_name]
        if denom > 0:
            recovery = (online_metrics[ndcg_name] - stale_metrics[ndcg_name]) / denom
        else:
            recovery = None

        allwarm_denom = (
            allwarm_full_metrics[ndcg_name] - allwarm_stale_metrics[ndcg_name]
        )
        if allwarm_denom > 0:
            allwarm_recovery = (
                allwarm_online_metrics[ndcg_name] - allwarm_stale_metrics[ndcg_name]
            ) / allwarm_denom
        else:
            allwarm_recovery = None

        step_speedup = full_time / online_time if online_time > 0 else float("inf")
        step_fraction = online_time / full_time if full_time > 0 else float("inf")
        cumulative_speedup = (
            cumulative_full / cumulative_online if cumulative_online > 0 else float("inf")
        )
        cumulative_fraction = (
            cumulative_online / cumulative_full if cumulative_full > 0 else float("inf")
        )

        step_construction_inclusive_speedup = (
            full_construction_inclusive / online_construction_inclusive
            if online_construction_inclusive > 0
            else float("inf")
        )
        step_construction_inclusive_fraction = (
            online_construction_inclusive / full_construction_inclusive
            if full_construction_inclusive > 0
            else float("inf")
        )
        cumulative_construction_inclusive_speedup = (
            cumulative_full_construction_inclusive
            / cumulative_online_construction_inclusive
            if cumulative_online_construction_inclusive > 0
            else float("inf")
        )
        cumulative_construction_inclusive_fraction = (
            cumulative_online_construction_inclusive
            / cumulative_full_construction_inclusive
            if cumulative_full_construction_inclusive > 0
            else float("inf")
        )

        full_v_exact = np.array_equal(full_model.V, base_v)
        full_v_diff = max_abs_diff(full_model.V, base_v)

        row = {
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
            "n_eval_rows": len(eval_rows),
            "n_eval_users": len({r[0] for r in eval_rows}),
            "n_eval_items": len({r[1] for r in eval_rows}),
            "n_allwarm_eval_rows": len(allwarm_eval_rows),
            "n_allwarm_eval_users": len({r[0] for r in allwarm_eval_rows}),
            "n_allwarm_eval_items": len({r[1] for r in allwarm_eval_rows}),
            "n_allwarm_eval_unadapted_users": len(
                {r[0] for r in allwarm_eval_rows} - users_exposed_to_update
            ),
            "update_min_timestamp": ts_min(update_rows),
            "update_max_timestamp": ts_max(update_rows),
            "eval_min_timestamp": ts_min(allwarm_eval_rows),
            "eval_max_timestamp": ts_max(allwarm_eval_rows),
            "history_dataset_build_time_s": history_build_time,
            "test_datasets_build_time_s": test_build_time,
            "full_dataset_build_time_s": full_build_time,
            "online_update_time_s": online_time,
            "full_retrain_time_s": full_time,
            "online_construction_inclusive_time_s": online_construction_inclusive,
            "full_construction_inclusive_time_s": full_construction_inclusive,
            "cumulative_online_time_s": cumulative_online,
            "cumulative_full_retrain_time_s": cumulative_full,
            "cumulative_online_construction_inclusive_time_s": (
                cumulative_online_construction_inclusive
            ),
            "cumulative_full_construction_inclusive_time_s": (
                cumulative_full_construction_inclusive
            ),
            "step_speedup_full_over_online": step_speedup,
            "step_online_full_cost_fraction": step_fraction,
            "cumulative_speedup_full_over_online": cumulative_speedup,
            "cumulative_online_full_cost_fraction": cumulative_fraction,
            "step_construction_inclusive_speedup_full_over_online": (
                step_construction_inclusive_speedup
            ),
            "step_construction_inclusive_online_full_cost_fraction": (
                step_construction_inclusive_fraction
            ),
            "cumulative_construction_inclusive_speedup_full_over_online": (
                cumulative_construction_inclusive_speedup
            ),
            "cumulative_construction_inclusive_online_full_cost_fraction": (
                cumulative_construction_inclusive_fraction
            ),
            "adaptation_recovery_NDCG@20": "" if recovery is None else recovery,
            "allwarm_adaptation_recovery_NDCG@20": (
                "" if allwarm_recovery is None else allwarm_recovery
            ),
            "stale_u_exact_equal_base": stale_u_exact,
            "stale_u_max_abs_diff_base": stale_u_diff,
            "stale_v_exact_equal_base": stale_v_exact,
            "stale_v_max_abs_diff_base": stale_v_diff,
            "online_v_exact_equal_base": online_v_exact,
            "online_v_max_abs_diff_base": online_v_diff,
            "full_v_exact_equal_base": full_v_exact,
            "full_v_max_abs_diff_base": full_v_diff,
            "online_maps_exact": online_maps_ok,
            "full_maps_exact": full_maps_ok,
        }

        for metric in QUALITY_METRICS:
            stale_value = stale_metrics[metric]
            online_value = online_metrics[metric]
            full_value = full_metrics[metric]
            row[f"stale_{metric}"] = stale_value
            row[f"online_{metric}"] = online_value
            row[f"full_{metric}"] = full_value
            row[f"online_minus_stale_{metric}"] = online_value - stale_value
            row[f"online_minus_full_{metric}"] = online_value - full_value
            row[f"full_minus_stale_{metric}"] = full_value - stale_value

            allwarm_stale_value = allwarm_stale_metrics[metric]
            allwarm_online_value = allwarm_online_metrics[metric]
            allwarm_full_value = allwarm_full_metrics[metric]
            row[f"allwarm_stale_{metric}"] = allwarm_stale_value
            row[f"allwarm_online_{metric}"] = allwarm_online_value
            row[f"allwarm_full_{metric}"] = allwarm_full_value
            row[f"allwarm_online_minus_stale_{metric}"] = (
                allwarm_online_value - allwarm_stale_value
            )
            row[f"allwarm_online_minus_full_{metric}"] = (
                allwarm_online_value - allwarm_full_value
            )
            row[f"allwarm_full_minus_stale_{metric}"] = (
                allwarm_full_value - allwarm_stale_value
            )

        step_rows.append(row)

        print(
            f"  update {update_chunk_number} -> eval {eval_chunk_number} | "
            f"PRIMARY rows={len(eval_rows):,}/{len(allwarm_eval_rows):,} | "
            f"NDCG stale={stale_metrics[ndcg_name]:.6f} | "
            f"online={online_metrics[ndcg_name]:.6f} | "
            f"full={full_metrics[ndcg_name]:.6f} | "
            f"ΔH1={row[f'online_minus_stale_{ndcg_name}']:+.6f} | "
            f"ΔH3={row[f'online_minus_full_{ndcg_name}']:+.6f} | "
            f"online={online_time:.4f}s | full={full_time:.2f}s | "
            f"speedup={step_speedup:.2f}x | "
            f"ALL-WARM ΔH1={allwarm_online_metrics[ndcg_name]-allwarm_stale_metrics[ndcg_name]:+.6f}"
        )

        observed_rows = post_update_rows

    if len(step_rows) != 3:
        raise RuntimeError("Trial no produjo exactamente 3 puntos.")

    recovery_values = finite_values(
        row["adaptation_recovery_NDCG@20"] for row in step_rows
    )
    allwarm_recovery_values = finite_values(
        row["allwarm_adaptation_recovery_NDCG@20"] for row in step_rows
    )

    trial = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "data_sha256": data["data_sha256"],
        "seed": seed,
        "n_eval_points": len(step_rows),
        "n_base_rows": len(data["base_rows"]),
        "n_base_users": n_users,
        "n_base_items": n_items,
        "base_train_time_s": base_train_time,
        "total_online_update_time_s": cumulative_online,
        "total_full_retrain_time_s": cumulative_full,
        "total_online_construction_inclusive_time_s": (
            cumulative_online_construction_inclusive
        ),
        "total_full_construction_inclusive_time_s": (
            cumulative_full_construction_inclusive
        ),
        "speedup_full_over_online": cumulative_full / cumulative_online,
        "online_full_cost_fraction": cumulative_online / cumulative_full,
        "construction_inclusive_speedup_full_over_online": (
            cumulative_full_construction_inclusive
            / cumulative_online_construction_inclusive
        ),
        "construction_inclusive_online_full_cost_fraction": (
            cumulative_online_construction_inclusive
            / cumulative_full_construction_inclusive
        ),
        "mean_adaptation_recovery_NDCG@20": (
            float(np.mean(recovery_values)) if recovery_values else ""
        ),
        "n_valid_adaptation_recovery_points": len(recovery_values),
        "mean_allwarm_adaptation_recovery_NDCG@20": (
            float(np.mean(allwarm_recovery_values)) if allwarm_recovery_values else ""
        ),
        "n_valid_allwarm_adaptation_recovery_points": len(
            allwarm_recovery_values
        ),
        "all_stale_u_exact_equal_base": all(
            bool(row["stale_u_exact_equal_base"]) for row in step_rows
        ),
        "max_stale_u_abs_diff_base": max(
            float(row["stale_u_max_abs_diff_base"]) for row in step_rows
        ),
        "all_stale_v_exact_equal_base": all(
            bool(row["stale_v_exact_equal_base"]) for row in step_rows
        ),
        "max_stale_v_abs_diff_base": max(
            float(row["stale_v_max_abs_diff_base"]) for row in step_rows
        ),
        "all_online_v_exact_equal_base": all(
            bool(row["online_v_exact_equal_base"]) for row in step_rows
        ),
        "max_online_v_abs_diff_base": max(
            float(row["online_v_max_abs_diff_base"]) for row in step_rows
        ),
    }

    for metric in QUALITY_METRICS:
        trial[f"mean_stale_{metric}"] = float(
            np.mean([row[f"stale_{metric}"] for row in step_rows])
        )
        trial[f"mean_online_{metric}"] = float(
            np.mean([row[f"online_{metric}"] for row in step_rows])
        )
        trial[f"mean_full_{metric}"] = float(
            np.mean([row[f"full_{metric}"] for row in step_rows])
        )
        for prefix in ["online_minus_stale", "online_minus_full", "full_minus_stale"]:
            values = [float(row[f"{prefix}_{metric}"]) for row in step_rows]
            trial[f"mean_{prefix}_{metric}"] = float(np.mean(values))
            trial[f"positive_{prefix}_{metric}"] = int(sum(value > 0 for value in values))

        trial[f"mean_allwarm_stale_{metric}"] = float(
            np.mean([row[f"allwarm_stale_{metric}"] for row in step_rows])
        )
        trial[f"mean_allwarm_online_{metric}"] = float(
            np.mean([row[f"allwarm_online_{metric}"] for row in step_rows])
        )
        trial[f"mean_allwarm_full_{metric}"] = float(
            np.mean([row[f"allwarm_full_{metric}"] for row in step_rows])
        )
        for prefix in [
            "allwarm_online_minus_stale",
            "allwarm_online_minus_full",
            "allwarm_full_minus_stale",
        ]:
            values = [float(row[f"{prefix}_{metric}"]) for row in step_rows]
            trial[f"mean_{prefix}_{metric}"] = float(np.mean(values))

    print(
        f"DONE seed={seed}: mean ΔH1 NDCG={trial['mean_online_minus_stale_NDCG@20']:+.6f} | "
        f"mean ΔH3 NDCG={trial['mean_online_minus_full_NDCG@20']:+.6f} | "
        f"total speedup={trial['speedup_full_over_online']:.2f}x | "
        f"construction-inclusive={trial['construction_inclusive_speedup_full_over_online']:.2f}x"
    )

    return step_rows, trial


# ============================================================
# Resume / summary
# ============================================================


def seed_complete(seed, steps, trials):
    seed_steps = [row for row in steps if int(row["seed"]) == int(seed)]
    seed_trials = [row for row in trials if int(row["seed"]) == int(seed)]
    if len(seed_trials) != 1 or len(seed_steps) != 3:
        return False

    eval_points = {int(row["eval_point"]) for row in seed_steps}
    update_chunks = {int(row["update_chunk"]) for row in seed_steps}
    eval_chunks = {int(row["eval_chunk"]) for row in seed_steps}
    return (
        eval_points == {1, 2, 3}
        and update_chunks == {1, 2, 3}
        and eval_chunks == {2, 3, 4}
        and int(seed_trials[0]["n_eval_points"]) == 3
    )


def remove_seed(rows, seed):
    return [row for row in rows if int(row["seed"]) != int(seed)]


def build_summary(step_rows, trial_rows, data, protocol_hash):
    if len(step_rows) != 6 or len(trial_rows) != 2:
        raise RuntimeError(
            f"Resumen final requiere 6 steps y 2 trials; recibidos "
            f"{len(step_rows)} y {len(trial_rows)}."
        )

    recovery_values = finite_values(
        row["adaptation_recovery_NDCG@20"] for row in step_rows
    )
    recovery_mean, recovery_std = (
        mean_std(recovery_values) if recovery_values else (float("nan"), float("nan"))
    )
    allwarm_recovery_values = finite_values(
        row["allwarm_adaptation_recovery_NDCG@20"] for row in step_rows
    )
    allwarm_recovery_mean, allwarm_recovery_std = (
        mean_std(allwarm_recovery_values)
        if allwarm_recovery_values
        else (float("nan"), float("nan"))
    )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "data_sha256": data["data_sha256"],
        "n_trials": len(trial_rows),
        "n_paired_points": len(step_rows),
        "seeds": ";".join(str(seed) for seed in FINAL_SEEDS),
        "target_base_fraction": TARGET_BASE_FRAC,
        "effective_base_fraction": len(data["base_rows"]) / len(data["all_rows"]),
        "target_base_rows": data["target_base_rows"],
        "effective_base_rows": data["effective_base_rows"],
        "boundary_tie_rows_added_to_base": data["boundary_tie_rows_added_to_base"],
        "n_future_raw": data["n_future_raw"],
        "n_future_warm": data["n_future_warm"],
        "warm_start_fraction": data["warm_start_fraction"],
        "n_warm_chunk_1": len(data["known_chunks"][0]),
        "n_warm_chunk_2": len(data["known_chunks"][1]),
        "n_warm_chunk_3": len(data["known_chunks"][2]),
        "n_warm_chunk_4": len(data["known_chunks"][3]),
        "n_primary_eval_rows_point_1": data["primary_eval_plan"][0]["n_primary_rows"],
        "n_primary_eval_rows_point_2": data["primary_eval_plan"][1]["n_primary_rows"],
        "n_primary_eval_rows_point_3": data["primary_eval_plan"][2]["n_primary_rows"],
        "n_excluded_unknown_user": data["n_excluded_unknown_user"],
        "n_excluded_unknown_item": data["n_excluded_unknown_item"],
        "n_excluded_unknown_both": data["n_excluded_unknown_both"],
        "n_valid_adaptation_recovery_points": len(recovery_values),
        "mean_adaptation_recovery_NDCG@20": recovery_mean,
        "std_adaptation_recovery_NDCG@20": recovery_std,
        "n_valid_allwarm_adaptation_recovery_points": len(allwarm_recovery_values),
        "mean_allwarm_adaptation_recovery_NDCG@20": allwarm_recovery_mean,
        "std_allwarm_adaptation_recovery_NDCG@20": allwarm_recovery_std,
        "all_stale_u_exact_equal_base": all(
            str(row["all_stale_u_exact_equal_base"]).lower() == "true"
            for row in trial_rows
        ),
        "max_stale_u_abs_diff_base": max(
            float(row["max_stale_u_abs_diff_base"]) for row in trial_rows
        ),
        "all_stale_v_exact_equal_base": all(
            str(row["all_stale_v_exact_equal_base"]).lower() == "true"
            for row in trial_rows
        ),
        "max_stale_v_abs_diff_base": max(
            float(row["max_stale_v_abs_diff_base"]) for row in trial_rows
        ),
        "all_online_v_exact_equal_base": all(
            str(row["all_online_v_exact_equal_base"]).lower() == "true"
            for row in trial_rows
        ),
        "max_online_v_abs_diff_base": max(
            float(row["max_online_v_abs_diff_base"]) for row in trial_rows
        ),
    }

    aggregate_trial_fields = [
        "base_train_time_s",
        "total_online_update_time_s",
        "total_full_retrain_time_s",
        "total_online_construction_inclusive_time_s",
        "total_full_construction_inclusive_time_s",
        "speedup_full_over_online",
        "online_full_cost_fraction",
        "construction_inclusive_speedup_full_over_online",
        "construction_inclusive_online_full_cost_fraction",
    ]
    output_names = {
        "base_train_time_s": "base_train_time_s",
        "total_online_update_time_s": "total_online_update_time_s",
        "total_full_retrain_time_s": "total_full_retrain_time_s",
        "total_online_construction_inclusive_time_s": (
            "total_online_construction_inclusive_time_s"
        ),
        "total_full_construction_inclusive_time_s": (
            "total_full_construction_inclusive_time_s"
        ),
        "speedup_full_over_online": "speedup_full_over_online",
        "online_full_cost_fraction": "online_full_cost_fraction",
        "construction_inclusive_speedup_full_over_online": (
            "construction_inclusive_speedup_full_over_online"
        ),
        "construction_inclusive_online_full_cost_fraction": (
            "construction_inclusive_online_full_cost_fraction"
        ),
    }
    for field in aggregate_trial_fields:
        mean_value, std_value = mean_std(float(row[field]) for row in trial_rows)
        name = output_names[field]
        summary[f"mean_{name}"] = mean_value
        summary[f"std_{name}"] = std_value

    # Explicit temporal aggregates required by the preregistered H1/H3 report.
    for eval_point in (1, 2, 3):
        point_rows = [
            row for row in step_rows if int(row["eval_point"]) == eval_point
        ]
        if len(point_rows) != len(FINAL_SEEDS):
            raise RuntimeError(
                f"eval_point={eval_point}: se esperaban {len(FINAL_SEEDS)} seeds, "
                f"recibidas={len(point_rows)}"
            )
        for prefix in [
            "online_minus_stale",
            "online_minus_full",
            "full_minus_stale",
        ]:
            values = [
                float(row[f"{prefix}_NDCG@{TOP_K}"]) for row in point_rows
            ]
            summary[f"mean_{prefix}_NDCG@{TOP_K}_eval_point_{eval_point}"] = (
                float(np.mean(values))
            )

    for metric in QUALITY_METRICS:
        for branch in ["stale", "online", "full"]:
            values = [float(row[f"{branch}_{metric}"]) for row in step_rows]
            mean_value, std_value = mean_std(values)
            summary[f"mean_{branch}_{metric}"] = mean_value
            summary[f"std_{branch}_{metric}"] = std_value

        for prefix in ["online_minus_stale", "online_minus_full", "full_minus_stale"]:
            values = [float(row[f"{prefix}_{metric}"]) for row in step_rows]
            mean_value, std_value = mean_std(values)
            summary[f"mean_{prefix}_{metric}"] = mean_value
            summary[f"std_{prefix}_{metric}"] = std_value
            summary[f"positive_{prefix}_{metric}"] = int(sum(value > 0 for value in values))

            if prefix == "online_minus_stale":
                summary[f"median_{prefix}_{metric}"] = float(np.median(values))
                summary[f"min_{prefix}_{metric}"] = float(np.min(values))
                summary[f"max_{prefix}_{metric}"] = float(np.max(values))

        for branch in ["stale", "online", "full"]:
            values = [float(row[f"allwarm_{branch}_{metric}"]) for row in step_rows]
            mean_value, std_value = mean_std(values)
            summary[f"mean_allwarm_{branch}_{metric}"] = mean_value
            summary[f"std_allwarm_{branch}_{metric}"] = std_value

        for prefix in [
            "allwarm_online_minus_stale",
            "allwarm_online_minus_full",
            "allwarm_full_minus_stale",
        ]:
            values = [float(row[f"{prefix}_{metric}"]) for row in step_rows]
            mean_value, std_value = mean_std(values)
            summary[f"mean_{prefix}_{metric}"] = mean_value
            summary[f"std_{prefix}_{metric}"] = std_value
            summary[f"positive_{prefix}_{metric}"] = int(
                sum(value > 0 for value in values)
            )

    return summary


def print_final_summary(summary, trial_rows):
    print()
    print("=" * 118)
    print("FINAL H1-H3 SUMMARY")
    print("=" * 118)
    print(f"Trials                         : {summary['n_trials']}")
    print(f"Paired evaluation points       : {summary['n_paired_points']}")
    print()
    print(f"Stale mean NDCG@20             : {summary['mean_stale_NDCG@20']:.6f}")
    print(f"Online mean NDCG@20            : {summary['mean_online_NDCG@20']:.6f}")
    print(f"Full mean NDCG@20              : {summary['mean_full_NDCG@20']:.6f}")
    print()
    print(
        "Online - Stale NDCG@20        : "
        f"{summary['mean_online_minus_stale_NDCG@20']:+.6f} ± "
        f"{summary['std_online_minus_stale_NDCG@20']:.6f} | "
        f"positive={summary['positive_online_minus_stale_NDCG@20']}/6"
    )
    print(
        "Online - Full NDCG@20         : "
        f"{summary['mean_online_minus_full_NDCG@20']:+.6f} ± "
        f"{summary['std_online_minus_full_NDCG@20']:.6f} | "
        f"positive={summary['positive_online_minus_full_NDCG@20']}/6"
    )
    print(
        "Full - Stale NDCG@20          : "
        f"{summary['mean_full_minus_stale_NDCG@20']:+.6f} ± "
        f"{summary['std_full_minus_stale_NDCG@20']:.6f} | "
        f"positive={summary['positive_full_minus_stale_NDCG@20']}/6"
    )
    print()
    print("Supplementary ALL-WARM diagnostic:")
    print(
        "  Online - Stale NDCG@20      : "
        f"{summary['mean_allwarm_online_minus_stale_NDCG@20']:+.6f} ± "
        f"{summary['std_allwarm_online_minus_stale_NDCG@20']:.6f} | "
        f"positive={summary['positive_allwarm_online_minus_stale_NDCG@20']}/6"
    )
    print(
        "  Online - Full NDCG@20       : "
        f"{summary['mean_allwarm_online_minus_full_NDCG@20']:+.6f} ± "
        f"{summary['std_allwarm_online_minus_full_NDCG@20']:.6f}"
    )
    print()
    print("H1/H2 interpretation            : DESCRIPTIVE ONLY; no automatic support label")
    print(
        "Mean total online update time  : "
        f"{summary['mean_total_online_update_time_s']:.4f}s"
    )
    print(
        "Mean total full retrain time   : "
        f"{summary['mean_total_full_retrain_time_s']:.4f}s"
    )
    print(
        "Mean full/online speedup        : "
        f"{summary['mean_speedup_full_over_online']:.2f}x"
    )
    print(
        "Mean online/full cost fraction  : "
        f"{summary['mean_online_full_cost_fraction']:.4f}"
    )
    print(
        "Supplementary construction-inclusive speedup: "
        f"{summary['mean_construction_inclusive_speedup_full_over_online']:.2f}x"
    )
    print(
        "Supplementary construction-inclusive cost fraction: "
        f"{summary['mean_construction_inclusive_online_full_cost_fraction']:.4f}"
    )
    if summary["n_valid_adaptation_recovery_points"] > 0:
        print(
            "Mean adaptation recovery       : "
            f"{summary['mean_adaptation_recovery_NDCG@20']:.4f} "
            f"(valid points={summary['n_valid_adaptation_recovery_points']}/6)"
        )
    else:
        print("Mean adaptation recovery       : NA (Full-Stale <= 0 in all points)")
    print()
    for row in sorted(trial_rows, key=lambda r: int(r["seed"])):
        print(
            f"seed={row['seed']} | ΔH1={float(row['mean_online_minus_stale_NDCG@20']):+.6f} | "
            f"ΔH3={float(row['mean_online_minus_full_NDCG@20']):+.6f} | "
            f"speedup={float(row['speedup_full_over_online']):.2f}x | "
            f"cost_fraction={float(row['online_full_cost_fraction']):.4f}"
        )
    print("Temporal paired means (across final seeds):")
    for eval_point in (1, 2, 3):
        print(
            f"  eval_point={eval_point} | "
            f"ΔH1={summary[f'mean_online_minus_stale_NDCG@{TOP_K}_eval_point_{eval_point}']:+.6f} | "
            f"ΔH3={summary[f'mean_online_minus_full_NDCG@{TOP_K}_eval_point_{eval_point}']:+.6f} | "
            f"Full-Stale={summary[f'mean_full_minus_stale_NDCG@{TOP_K}_eval_point_{eval_point}']:+.6f}"
        )
    print()
    print("Interpretation guard:")
    print("  - no retuning after these results")
    print("  - H1/H2/H3 summaries are descriptive; no automatic inferential support label")
    print("  - H2 primary timing is model-update compute only (partial_fit_recent vs IBPR.fit)")
    print("  - construction-inclusive H2 is supplementary and excludes evaluation-test builds")
    print("  - H3 is descriptive; no equivalence/non-inferiority claim")
    print("  - H4 index reuse remains a separate experiment")
    print()


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    validate_implementation_contracts()
    data = prepare_protocol_data()
    protocol_hash = current_protocol_hash(data["data_sha256"])

    if args.plan_only:
        print_plan(data, protocol_hash)
        print(
            "PLAN-ONLY complete. No MovieLens final experimental model was trained "
            "and no final-result file was written. Only synthetic implementation "
            "contract smoke tests were executed."
        )
        return

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_path = os.path.join(
        RESULTS_DIR, f"final_h1_h3_online_ibpr_mejorado_{timestamp}.txt"
    )
    steps_path = os.path.join(
        RESULTS_DIR, f"final_h1_h3_online_ibpr_mejorado_steps_{timestamp}.csv"
    )
    trials_path = os.path.join(
        RESULTS_DIR, f"final_h1_h3_online_ibpr_mejorado_trials_{timestamp}.csv"
    )
    summary_path = os.path.join(
        RESULTS_DIR, f"final_h1_h3_online_ibpr_mejorado_summary_{timestamp}.csv"
    )

    step_rows = load_csv(steps_path)
    trial_rows = load_csv(trials_path)
    validate_resume_protocol(step_rows, steps_path, protocol_hash, data["data_sha256"])
    validate_resume_protocol(trial_rows, trials_path, protocol_hash, data["data_sha256"])

    log_mode = "a" if os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with redirect_stdout(tee):
            # Persist the complete frozen plan/fingerprint inside the final log.
            print_plan(data, protocol_hash)
            print()
            print("#" * 118)
            print(f"FINAL H1-H3 TIMESTAMP: {timestamp}")
            print("#" * 118)
            print(f"Log     : {log_path}")
            print(f"Steps   : {steps_path}")
            print(f"Trials  : {trials_path}")
            print(f"Summary : {summary_path}")
            print(f"Protocol: {protocol_hash}")
            print()

            for seed in FINAL_SEEDS:
                # Reuse only structurally complete seeds.
                if seed_complete(seed, step_rows, trial_rows):
                    print(f"REUSE seed={seed}: complete (3/3 paired points).")
                    continue

                # Any partial rows for this seed are invalidated atomically.
                old_step_count = len(step_rows)
                old_trial_count = len(trial_rows)
                step_rows = remove_seed(step_rows, seed)
                trial_rows = remove_seed(trial_rows, seed)
                if len(step_rows) != old_step_count or len(trial_rows) != old_trial_count:
                    print(f"RESET seed={seed}: incomplete prior state removed; rerunning full seed.")
                    save_csv_atomic(steps_path, STEP_FIELDS, step_rows)
                    save_csv_atomic(trials_path, TRIAL_FIELDS, trial_rows)

                new_steps, new_trial = run_seed_trial(seed, data, protocol_hash)

                # Persist only after the whole seed completed successfully.
                step_rows.extend(new_steps)
                trial_rows.append(new_trial)
                save_csv_atomic(steps_path, STEP_FIELDS, step_rows)
                save_csv_atomic(trials_path, TRIAL_FIELDS, trial_rows)

            # Final structural integrity.
            for seed in FINAL_SEEDS:
                if not seed_complete(seed, step_rows, trial_rows):
                    raise RuntimeError(f"seed={seed} no quedó completa al cierre.")

            if len(step_rows) != 6 or len(trial_rows) != 2:
                raise RuntimeError(
                    f"Conteos finales inválidos: steps={len(step_rows)}, trials={len(trial_rows)}"
                )

            # Refuse duplicates.
            step_keys = [
                (int(row["seed"]), int(row["eval_point"])) for row in step_rows
            ]
            if len(step_keys) != len(set(step_keys)):
                raise RuntimeError("Steps duplicados detectados.")
            trial_keys = [int(row["seed"]) for row in trial_rows]
            if len(trial_keys) != len(set(trial_keys)):
                raise RuntimeError("Trials duplicados detectados.")

            summary = build_summary(step_rows, trial_rows, data, protocol_hash)
            save_csv_atomic(summary_path, SUMMARY_FIELDS, [summary])
            print_final_summary(summary, trial_rows)

            print("Files:")
            print(f"  log     : {log_path}")
            print(f"  steps   : {steps_path}")
            print(f"  trials  : {trials_path}")
            print(f"  summary : {summary_path}")


if __name__ == "__main__":
    main()
