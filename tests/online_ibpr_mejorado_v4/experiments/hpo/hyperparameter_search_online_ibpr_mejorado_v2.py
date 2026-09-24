import argparse
import csv
import hashlib
import itertools
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
from cornac.utils.Tee import Tee

from cornac.models.online_ibpr_mejorado.online_ibpr_mejorado import (
    online_ibpr_mejorado as _online_core_contract_check,
)



# ============================================================
# Frozen IBPR base
# ============================================================

RATING_THRESHOLD = 3.0
VARIANT = "1M"
TOP_K = 20
HPO_END_FRAC = 0.60
PROTOCOL_VERSION = "online_hpo_global_prequential_v2_20260924"

FROZEN_IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}

# ============================================================
# Online HPO protocol
# ============================================================

N_STREAM_CHUNKS = 4

STREAM_SCENARIOS = [
    {"name": "S50", "base_ratio": 0.50},
    {"name": "S65", "base_ratio": 0.65},
    {"name": "S80", "base_ratio": 0.80},
]

ONLINE_LR_VALUES = [0.001, 0.0025, 0.005, 0.01, 0.02]
ONLINE_LAMDA_VALUES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
ONLINE_BATCH_VALUES = [128, 256, 512, 1024]
ONLINE_EPOCH_VALUES = [1, 2, 3]
LOSS_MODES = ["cosine_bpr", "angular"]

FIXED_UPDATE_V = False
FIXED_NEG_SAMPLING = "uniform"
FIXED_NORMALIZE = True
FIXED_MAX_STEPS = None

DEFAULT_SCREENING_CONFIGS = 20
SEARCH_RANDOM_SEED = 2026
SCREENING_SEED = 42
CONFIRMATION_SEEDS = [42, 123, 2024]

TOP_STAGE_A = 5
TOP_STAGE_B = 3

PRIMARY_DELTA_METRIC = f"delta_NDCG@{TOP_K}"
PRIMARY_ABSOLUTE_METRIC = f"NDCG@{TOP_K}"

QUALITY_METRICS = [
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
]

PROTECTED_SOURCES = {
    "pilot_online_anchor",
    "frozen_base_like_cosine_anchor",
    "frozen_base_like_angular_anchor",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RUN_CONTEXT = {"protocol_hash": None, "data_sha256": None}


def run_meta():
    if not RUN_CONTEXT["protocol_hash"] or not RUN_CONTEXT["data_sha256"]:
        raise RuntimeError("RUN_CONTEXT no inicializado.")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": RUN_CONTEXT["protocol_hash"],
        "data_sha256": RUN_CONTEXT["data_sha256"],
    }


TRIAL_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "origin_stage",
    "config_id",
    "config_source",
    "seed",
    "scenario",
    "base_ratio",
    "online_learning_rate",
    "online_lamda",
    "online_batch_size",
    "n_epochs",
    "loss_mode",
    "update_V",
    "neg_sampling",
    "normalize",
    "max_steps",
    "n_base",
    "n_base_users",
    "n_base_items",
    "n_stream_candidate",
    "n_stream_known",
    "stream_known_fraction",
    "n_stream_users",
    "n_stream_items",
    "n_eval_points",
    "total_update_time_s",
    "mean_update_time_s",
    "max_update_time_s",
    "v_exact_equal",
    "v_max_abs_diff",
    "mean_AUC",
    "mean_MAP",
    f"mean_NDCG@{TOP_K}",
    f"mean_Precision@{TOP_K}",
    f"mean_Recall@{TOP_K}",
    "mean_stale_AUC",
    "mean_stale_MAP",
    f"mean_stale_NDCG@{TOP_K}",
    f"mean_stale_Precision@{TOP_K}",
    f"mean_stale_Recall@{TOP_K}",
    "mean_delta_AUC",
    "mean_delta_MAP",
    f"mean_delta_NDCG@{TOP_K}",
    f"mean_delta_Precision@{TOP_K}",
    f"mean_delta_Recall@{TOP_K}",
]


CHUNK_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "origin_stage",
    "config_id",
    "seed",
    "scenario",
    "eval_point",
    "eval_chunk",
    "trained_on_chunks",
    "online_learning_rate",
    "online_lamda",
    "online_batch_size",
    "n_epochs",
    "loss_mode",
    "n_eval_rows",
    "n_eval_users",
    "n_eval_items",
    "n_allwarm_eval_rows",
    "n_allwarm_eval_users",
    "n_users_exposed_to_update",
    "update_time_s_before_eval",
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
    "stale_AUC",
    "stale_MAP",
    f"stale_NDCG@{TOP_K}",
    f"stale_Precision@{TOP_K}",
    f"stale_Recall@{TOP_K}",
    "delta_AUC",
    "delta_MAP",
    f"delta_NDCG@{TOP_K}",
    f"delta_Precision@{TOP_K}",
    f"delta_Recall@{TOP_K}",
]


SUMMARY_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "data_sha256",
    "stage",
    "rank",
    "config_id",
    "config_source",
    "online_learning_rate",
    "online_lamda",
    "online_batch_size",
    "n_epochs",
    "loss_mode",
    "seeds",
    "scenarios",
    "n_runs",
    "mean_delta_AUC",
    "std_delta_AUC",
    "mean_delta_MAP",
    "std_delta_MAP",
    f"mean_delta_NDCG@{TOP_K}",
    f"std_delta_NDCG@{TOP_K}",
    f"mean_delta_Precision@{TOP_K}",
    f"std_delta_Precision@{TOP_K}",
    f"mean_delta_Recall@{TOP_K}",
    f"std_delta_Recall@{TOP_K}",
    "mean_AUC",
    "std_AUC",
    "mean_MAP",
    "std_MAP",
    f"mean_NDCG@{TOP_K}",
    f"std_NDCG@{TOP_K}",
    f"mean_Precision@{TOP_K}",
    f"std_Precision@{TOP_K}",
    f"mean_Recall@{TOP_K}",
    f"std_Recall@{TOP_K}",
    "mean_total_update_time_s",
    "std_total_update_time_s",
    "all_v_exact_equal",
    "max_v_abs_diff",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter selection for OnlineIBPRMejorado incremental "
            "adaptation using global chronological prequential development streams."
        )
    )

    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Timestamp YYYYMMDD_HHMMSS. Reusing an existing timestamp "
            "resumes completed configuration/scenario/seed trials."
        ),
    )

    parser.add_argument(
        "--screening-configs",
        type=int,
        default=DEFAULT_SCREENING_CONFIGS,
        help="Number of Stage O-A configurations. Default: 20.",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print protocol, stream sizes and candidate configurations only.",
    )

    return parser.parse_args()


# ============================================================
# Stabilized implementation contract
# ============================================================

def validate_stabilized_online_contract():
    """
    Refuse to run HPO against an obsolete OnlineIBPRMejorado implementation.

    This HPO assumes the stabilized implementation already validated by the
    project invariants:
      - wrapper exposes seed;
      - successive non-empty partial updates consume seed + update_count;
      - update_V=False + normalize=True leaves V bit-for-bit unchanged;
      - empty partial update is an identity;
      - max_steps=0 is rejected.
    """
    ctor_params = set(
        inspect.signature(OnlineIBPRMejorado.__init__).parameters
    )

    if "seed" not in ctor_params:
        raise RuntimeError(
            "OnlineIBPRMejorado obsoleto detectado: el wrapper no expone "
            "`seed`. Usa la versión estabilizada antes de ejecutar este HPO."
        )

    partial_params = set(
        inspect.signature(
            OnlineIBPRMejorado.partial_fit_recent
        ).parameters
    )

    required_partial = {
        "recent_pairs",
        "history_csr",
        "max_steps",
        "n_epochs",
    }

    missing = required_partial - partial_params

    if missing:
        raise RuntimeError(
            "partial_fit_recent(...) no cumple el contrato estabilizado. "
            f"Faltan: {sorted(missing)}"
        )

    # Tiny experiments smoke test: normalize=True must not alter V when V is frozen.
    U0 = np.asarray(
        [[2.0, 0.0], [0.0, 3.0]],
        dtype=np.float32,
    )
    V0 = np.asarray(
        [[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    history = csr_matrix(
        np.asarray(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )
    recent = np.asarray([[0, 1]], dtype=np.int64)

    result = _online_core_contract_check(
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

    if not np.array_equal(
        np.asarray(result["V"]),
        V0,
    ):
        max_diff = float(
            np.max(
                np.abs(
                    np.asarray(result["V"]) - V0
                )
            )
        )
        raise RuntimeError(
            "Core obsoleto detectado: normalize=True modificó V con "
            "update_V=False. Se requiere la versión estabilizada. "
            f"max_abs_diff={max_diff}"
        )

    # max_steps=0 must be rejected.
    try:
        _online_core_contract_check(
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
            max_steps=0,
            random_seed=17,
            loss_mode="cosine_bpr",
        )
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Core obsoleto detectado: max_steps=0 no fue rechazado."
        )

    # Wrapper seed progression / empty-update identity.
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
        verbose=False,
        name="contract_check",
    )
    wrapper.num_users = history.shape[0]
    wrapper.num_items = history.shape[1]

    if not hasattr(wrapper, "_partial_update_count"):
        raise RuntimeError(
            "Wrapper obsoleto detectado: falta _partial_update_count."
        )

    before_u = np.asarray(wrapper.U).copy()
    before_v = np.asarray(wrapper.V).copy()
    before_count = int(wrapper._partial_update_count)

    wrapper.partial_fit_recent(
        recent_pairs=np.empty((0, 2), dtype=np.int64),
        history_csr=history,
        max_steps=None,
        n_epochs=1,
    )

    if int(wrapper._partial_update_count) != before_count:
        raise RuntimeError(
            "Wrapper incorrecto: un update vacío consumió el seed."
        )

    if not np.array_equal(wrapper.U, before_u):
        raise RuntimeError(
            "Wrapper incorrecto: un update vacío modificó U."
        )

    if not np.array_equal(wrapper.V, before_v):
        raise RuntimeError(
            "Wrapper incorrecto: un update vacío modificó V."
        )

    wrapper.partial_fit_recent(
        recent_pairs=recent,
        history_csr=history,
        max_steps=None,
        n_epochs=1,
    )

    if int(wrapper._partial_update_count) != before_count + 1:
        raise RuntimeError(
            "Wrapper incorrecto: un update no vacío no incrementó "
            "_partial_update_count exactamente una vez."
        )

    if not np.array_equal(wrapper.V, before_v):
        raise RuntimeError(
            "Wrapper incorrecto: update_V=False modificó V."
        )

    print("Stabilized OnlineIBPRMejorado contract: OK")
    print("  wrapper seed support               : OK")
    print("  empty partial update identity       : OK")
    print("  non-empty seed progression          : OK")
    print("  V exact with update_V=False         : OK")
    print("  V exact with normalize=True         : OK")
    print("  max_steps=0 rejection               : OK")
    print()


# ============================================================
# Generic utilities
# ============================================================

def append_csv(path, fieldnames, row):
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_csv(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        raise ValueError("No se puede calcular media de una colección vacía.")

    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return mean_value, std_value


def unique_preserving_order(values):
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


# ============================================================
# Data preparation
# ============================================================

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _source_sha256(obj):
    try:
        return sha256_bytes(inspect.getsource(obj).encode("utf-8"))
    except (OSError, TypeError):
        return "unavailable"


def _script_sha256():
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return sha256_bytes(f.read())
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


def ts_min(rows):
    return int(min(row[3] for row in rows))


def ts_max(rows):
    return int(max(row[3] for row in rows))


def rows_pairs(rows):
    return {(row[0], row[1]) for row in rows}


def assert_unique_user_item_pairs(rows, label):
    n_rows = len(rows)
    n_pairs = len(rows_pairs(rows))
    if n_rows != n_pairs:
        raise RuntimeError(
            f"{label}: pares (u,i) duplicados: rows={n_rows}, unique_pairs={n_pairs}."
        )


def cornac_rows(rows):
    return [(u, i, float(value), int(timestamp)) for u, i, value, timestamp, _ in rows]


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

    assert_unique_user_item_pairs(positive, "all_positive_rows")
    return positive


def build_hpo_pool(all_positive_rows):
    """Reserve the globally latest ~40% for final H1-H4.

    The development boundary moves forward through an equal-timestamp group so
    no timestamp is split between development and untouched final evaluation.
    """
    n = len(all_positive_rows)
    target = int(n * HPO_END_FRAC)

    if target <= 0 or target >= n:
        raise ValueError(f"Corte HPO inválido: target={target}, n={n}")

    effective = target
    boundary_ts = all_positive_rows[target - 1][3]
    while effective < n and all_positive_rows[effective][3] == boundary_ts:
        effective += 1

    if effective >= n:
        raise ValueError("El ajuste temporal del HPO consumió todo el holdout final.")

    hpo_pool = list(all_positive_rows[:effective])
    untouched_final = list(all_positive_rows[effective:])

    if ts_max(hpo_pool) >= ts_min(untouched_final):
        raise RuntimeError("Development/final no tienen separación temporal estricta.")

    return {
        "target_rows": target,
        "effective_rows": effective,
        "boundary_tie_rows_added": effective - target,
        "boundary_timestamp": int(boundary_ts),
        "hpo_pool": hpo_pool,
        "untouched_final": untouched_final,
    }


def split_global_base_future_strict(rows, base_ratio, label):
    n = len(rows)
    target = int(np.floor(n * float(base_ratio)))

    if target <= 0 or target >= n:
        raise ValueError(f"{label}: corte base/future inválido: target={target}, n={n}")

    effective = target
    boundary_ts = rows[target - 1][3]
    while effective < n and rows[effective][3] == boundary_ts:
        effective += 1

    if effective >= n:
        raise ValueError(f"{label}: empate temporal consumió todo el future stream.")

    base_rows = list(rows[:effective])
    future_rows = list(rows[effective:])

    if ts_max(base_rows) >= ts_min(future_rows):
        raise RuntimeError(f"{label}: base/future no son estrictamente cronológicos.")

    return {
        "target_base_rows": target,
        "effective_base_rows": effective,
        "boundary_tie_rows_added": effective - target,
        "boundary_timestamp": int(boundary_ts),
        "base_rows": base_rows,
        "future_rows": future_rows,
    }


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
        raise ValueError("El stream quedó vacío después del filtrado warm-start.")

    return {
        "warm_rows": warm_rows,
        "excluded_user": excluded_user,
        "excluded_item": excluded_item,
        "excluded_both": excluded_both,
        "warm_fraction": len(warm_rows) / len(future_rows),
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
            raise ValueError(f"Un empate temporal consumiría el resto de {label}.")
        if effective <= boundaries[-1]:
            raise RuntimeError(f"Límite no creciente en {label}.")

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
                f"Chunks {idx+1}/{idx+2} de {label} no son estrictamente temporales."
            )

    return chunks, boundaries, adjustments


def primary_eval_rows_for_step(known_chunks, step_idx):
    """Rows whose user has received at least one prior/current update chunk."""
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


def build_stream_scenario(hpo_pool, scenario):
    """Build one globally chronological development scenario.

    S50/S65/S80 refer to a GLOBAL base fraction inside the development horizon,
    not to per-user history fractions.
    """
    name = str(scenario["name"])
    base_ratio = float(scenario["base_ratio"])

    if not (0.0 < base_ratio < 1.0):
        raise ValueError(f"{name}: base_ratio inválido: {base_ratio}")

    split = split_global_base_future_strict(hpo_pool, base_ratio, name)
    warm = warm_start_filter(split["base_rows"], split["future_rows"])
    known_chunks, boundaries, adjustments = split_chrono_chunks_strict(
        warm["warm_rows"],
        n_chunks=N_STREAM_CHUNKS,
        label=f"{name} warm-start stream",
    )

    assert_unique_user_item_pairs(split["base_rows"], f"{name}_base")
    observed = rows_pairs(split["base_rows"])
    for idx, chunk in enumerate(known_chunks, start=1):
        assert_unique_user_item_pairs(chunk, f"{name}_chunk_{idx}")
        overlap = observed & rows_pairs(chunk)
        if overlap:
            sample = next(iter(overlap))
            raise RuntimeError(
                f"{name}: par (u,i) repetido entre historial y chunk {idx}: {sample}"
            )
        observed.update(rows_pairs(chunk))

    primary_plan = []
    for step_idx in range(N_STREAM_CHUNKS - 1):
        primary, allwarm, exposed = primary_eval_rows_for_step(known_chunks, step_idx)
        if ts_max(known_chunks[step_idx]) >= ts_min(allwarm):
            raise RuntimeError(f"{name}: update/eval no son estrictamente temporales.")
        primary_plan.append(
            {
                "eval_point": step_idx + 1,
                "update_chunk": step_idx + 1,
                "eval_chunk": step_idx + 2,
                "n_primary_rows": len(primary),
                "n_primary_users": len({row[0] for row in primary}),
                "n_allwarm_rows": len(allwarm),
                "n_allwarm_users": len({row[0] for row in allwarm}),
                "n_users_exposed": len(exposed),
            }
        )

    return {
        "name": name,
        "base_ratio": base_ratio,
        "target_base_rows": split["target_base_rows"],
        "effective_base_rows": split["effective_base_rows"],
        "boundary_tie_rows_added": split["boundary_tie_rows_added"],
        "base_rows": split["base_rows"],
        "future_rows": split["future_rows"],
        "known_chunks": known_chunks,
        "chunk_boundaries": boundaries,
        "chunk_adjustments": adjustments,
        "primary_eval_plan": primary_plan,
        "n_stream_candidate": len(split["future_rows"]),
        "n_stream_known": len(warm["warm_rows"]),
        "stream_known_fraction": warm["warm_fraction"],
        "n_excluded_unknown_user": warm["excluded_user"],
        "n_excluded_unknown_item": warm["excluded_item"],
        "n_excluded_unknown_both": warm["excluded_both"],
    }


def protocol_payload(data_hash, configs):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "MovieLens 1M",
        "rating_threshold": RATING_THRESHOLD,
        "outer_development_fraction": HPO_END_FRAC,
        "outer_boundary_rule": "move forward to end of equal-timestamp group",
        "scenario_base_ratios": STREAM_SCENARIOS,
        "scenario_rule": "global chronological base/future split within development",
        "warm_start_rule": "future row retained iff user and item exist in scenario base",
        "chunk_rule": "four global chronological warm chunks; move boundaries forward through timestamp ties",
        "primary_eval_rule": "evaluate next chunk only for users exposed to an earlier update chunk",
        "prequential_sequence": [[1, 2], [2, 3], [3, 4]],
        "frozen_ibpr_config": FROZEN_IBPR_CONFIG,
        "fixed_online_constraints": {
            "update_V": FIXED_UPDATE_V,
            "neg_sampling": FIXED_NEG_SAMPLING,
            "normalize": FIXED_NORMALIZE,
            "max_steps": FIXED_MAX_STEPS,
        },
        "search_space": {
            "learning_rate": ONLINE_LR_VALUES,
            "lamda": ONLINE_LAMDA_VALUES,
            "batch_size": ONLINE_BATCH_VALUES,
            "n_epochs": ONLINE_EPOCH_VALUES,
            "loss_mode": LOSS_MODES,
        },
        "screening_configs": [
            {
                "config_id": c["config_id"],
                "signature": config_signature(c),
                "config_source": c["config_source"],
            }
            for c in configs
        ],
        "search_random_seed": SEARCH_RANDOM_SEED,
        "screening_seed": SCREENING_SEED,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "top_stage_a": TOP_STAGE_A,
        "top_stage_b": TOP_STAGE_B,
        "primary_selection_metric": PRIMARY_DELTA_METRIC,
        "secondary_selection_metric": PRIMARY_ABSOLUTE_METRIC,
        "data_sha256": data_hash,
        "script_sha256": _script_sha256(),
        "python": platform.python_version(),
        "cornac": getattr(cornac, "__version__", "unavailable"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "ibpr_wrapper_sha256": _source_sha256(IBPR),
        "online_wrapper_sha256": _source_sha256(OnlineIBPRMejorado),
        "online_core_sha256": _source_sha256(_online_core_contract_check),
    }


def current_protocol_hash(data_hash, configs):
    canonical = json.dumps(
        protocol_payload(data_hash, configs),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=list,
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
                f"{path}: fila {idx} pertenece a otro protocolo/dataset. "
                "Usa un timestamp nuevo; no mezcles ejecuciones."
            )


def print_data_plan(all_rows, hpo_info, scenario_data, data_hash, protocol_hash):
    print("=" * 110)
    print("ONLINE IBPR HPO V2 - GLOBAL PREQUENTIAL DEVELOPMENT PROTOCOL")
    print("=" * 110)
    print(f"Protocol version                 : {PROTOCOL_VERSION}")
    print(f"Protocol hash                    : {protocol_hash}")
    print(f"Dataset SHA256                   : {data_hash}")
    print(f"Dataset                          : MovieLens {VARIANT}")
    print(f"Positive threshold               : rating >= {RATING_THRESHOLD}")
    print("Feedback                         : implicit positive = 1.0")
    print(f"Target development horizon       : first {HPO_END_FRAC:.0%} globally")
    print(f"Target development rows          : {hpo_info['target_rows']:,}")
    print(f"Effective development rows       : {hpo_info['effective_rows']:,}")
    print(f"Tie rows added at outer boundary : {hpo_info['boundary_tie_rows_added']:,}")
    print(f"Untouched final rows             : {len(hpo_info['untouched_final']):,}")
    print("Later global data used by HPO    : NO")
    print(f"Total implicit-positive rows     : {len(all_rows):,}")
    print()

    for scenario in scenario_data.values():
        base_rows = scenario["base_rows"]
        print(
            f"{scenario['name']}: GLOBAL base target={scenario['base_ratio']:.0%} | "
            f"base={len(base_rows):,} "
            f"({len({r[0] for r in base_rows}):,} users, "
            f"{len({r[1] for r in base_rows}):,} items) | "
            f"future={scenario['n_stream_candidate']:,} | "
            f"warm={scenario['n_stream_known']:,} "
            f"({scenario['stream_known_fraction']:.2%})"
        )
        print(
            f"    base tie adjustment={scenario['boundary_tie_rows_added']:,} | "
            f"excluded user/item/both="
            f"{scenario['n_excluded_unknown_user']:,}/"
            f"{scenario['n_excluded_unknown_item']:,}/"
            f"{scenario['n_excluded_unknown_both']:,}"
        )

        for idx, (chunk, plan) in enumerate(
            zip(scenario["known_chunks"], [None] + scenario["primary_eval_plan"]),
            start=1,
        ):
            print(
                f"    chunk_{idx}: rows={len(chunk):,} | "
                f"users={len({r[0] for r in chunk}):,} | "
                f"items={len({r[1] for r in chunk}):,}"
            )

        for plan in scenario["primary_eval_plan"]:
            print(
                f"    eval_point_{plan['eval_point']}: update=C{plan['update_chunk']} -> "
                f"eval=C{plan['eval_chunk']} | primary={plan['n_primary_rows']:,} rows / "
                f"{plan['n_primary_users']:,} users | allwarm={plan['n_allwarm_rows']:,} rows"
            )

    print()
    print(
        "Interpretation: all scenario boundaries and chunks are GLOBAL chronological. "
        "The globally later holdout is untouched by HPO and reserved for final H1-H4."
    )
    print()


# ============================================================
# Candidate construction
# ============================================================

def config_signature(config):
    return (
        float(config["online_learning_rate"]),
        float(config["online_lamda"]),
        int(config["online_batch_size"]),
        int(config["n_epochs"]),
        str(config["loss_mode"]),
    )


def all_grid_configs():
    return [
        {
            "online_learning_rate": float(lr),
            "online_lamda": float(lamda),
            "online_batch_size": int(batch),
            "n_epochs": int(epochs),
            "loss_mode": str(loss_mode),
        }
        for lr, lamda, batch, epochs, loss_mode in itertools.product(
            ONLINE_LR_VALUES,
            ONLINE_LAMDA_VALUES,
            ONLINE_BATCH_VALUES,
            ONLINE_EPOCH_VALUES,
            LOSS_MODES,
        )
    ]


def anchor_configs():
    return [
        {
            "config_source": "pilot_online_anchor",
            "online_learning_rate": 0.01,
            "online_lamda": 0.001,
            "online_batch_size": 512,
            "n_epochs": 1,
            "loss_mode": "cosine_bpr",
        },
        {
            "config_source": "frozen_base_like_cosine_anchor",
            "online_learning_rate": 0.0025,
            "online_lamda": 1e-05,
            "online_batch_size": 512,
            "n_epochs": 1,
            "loss_mode": "cosine_bpr",
        },
        {
            "config_source": "frozen_base_like_angular_anchor",
            "online_learning_rate": 0.0025,
            "online_lamda": 1e-05,
            "online_batch_size": 512,
            "n_epochs": 1,
            "loss_mode": "angular",
        },
    ]


def parameter_levels():
    return {
        "online_learning_rate": set(map(float, ONLINE_LR_VALUES)),
        "online_lamda": set(map(float, ONLINE_LAMDA_VALUES)),
        "online_batch_size": set(map(int, ONLINE_BATCH_VALUES)),
        "n_epochs": set(map(int, ONLINE_EPOCH_VALUES)),
        "loss_mode": set(map(str, LOSS_MODES)),
    }


def build_screening_configs(total_configs):
    anchors = anchor_configs()

    if total_configs < len(anchors):
        raise ValueError(
            f"screening-configs debe ser >= {len(anchors)}."
        )

    full_grid = all_grid_configs()
    anchor_signatures = {config_signature(config) for config in anchors}

    remaining = [
        config
        for config in full_grid
        if config_signature(config) not in anchor_signatures
    ]

    rng = np.random.default_rng(SEARCH_RANDOM_SEED)
    order = rng.permutation(len(remaining))
    shuffled = [remaining[int(index)] for index in order]

    selected = [dict(config) for config in anchors]

    levels = parameter_levels()

    def observed_levels(configs):
        return {
            parameter: {config[parameter] for config in configs}
            for parameter in levels
        }

    # Coverage-aware deterministic sampling.
    for candidate in shuffled:
        if len(selected) >= total_configs:
            break

        observed = observed_levels(selected)

        introduces_missing_level = any(
            candidate[parameter] not in observed[parameter]
            for parameter in levels
        )

        if introduces_missing_level:
            item = dict(candidate)
            item["config_source"] = "coverage_random_search"
            selected.append(item)

    # Fill the remaining budget reproducibly.
    selected_signatures = {config_signature(config) for config in selected}

    for candidate in shuffled:
        if len(selected) >= total_configs:
            break

        signature = config_signature(candidate)

        if signature in selected_signatures:
            continue

        item = dict(candidate)
        item["config_source"] = "random_search"
        selected.append(item)
        selected_signatures.add(signature)

    if len(selected) != total_configs:
        raise ValueError(
            f"No se pudieron construir {total_configs} configuraciones."
        )

    # Validate coverage.
    observed = observed_levels(selected)

    missing = {
        parameter: sorted(levels[parameter] - observed[parameter], key=str)
        for parameter in levels
        if levels[parameter] - observed[parameter]
    }

    if missing:
        raise ValueError(
            f"El screening no cubre todos los niveles configurados: {missing}"
        )

    for index, config in enumerate(selected, start=1):
        config["config_id"] = f"O{index:03d}"

    # Defensive uniqueness.
    signatures = [config_signature(config) for config in selected]

    if len(signatures) != len(set(signatures)):
        raise ValueError("Se detectaron configuraciones online duplicadas.")

    return selected


def protected_config_ids(configs):
    return [
        config["config_id"]
        for config in configs
        if config["config_source"] in PROTECTED_SOURCES
    ]


def print_candidate_plan(configs):
    print("=" * 110)
    print("STAGE O-A ONLINE CONFIGURATIONS")
    print("=" * 110)

    for config in configs:
        print(
            f"{config['config_id']} | "
            f"source={config['config_source']:<24s} | "
            f"lr={config['online_learning_rate']:<7g} "
            f"lamda={config['online_lamda']:<8g} "
            f"batch={config['online_batch_size']:<4d} "
            f"epochs={config['n_epochs']} "
            f"loss={config['loss_mode']}"
        )

    print()
    print("Fixed online architecture:")
    print(f"  k            = {FROZEN_IBPR_CONFIG['k']}")
    print(f"  update_V     = {FIXED_UPDATE_V}")
    print(f"  neg_sampling = {FIXED_NEG_SAMPLING}")
    print(f"  normalize    = {FIXED_NORMALIZE}")
    print(f"  max_steps    = {FIXED_MAX_STEPS}")
    print()
    print("Budget:")
    print(
        f"  O-A: {len(configs)} configs × S50,S65 × seed {SCREENING_SEED}"
    )
    print(
        "  O-B: top 5 + protected anchors × S80 × seed 42"
    )
    print(
        "  O-C: top 3 + protected anchors × S50,S65,S80 "
        f"× seeds={CONFIRMATION_SEEDS}"
    )
    print()


# ============================================================
# Datasets / metrics
# ============================================================

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


def rows_to_pairs(rows, uid_map, iid_map):
    pairs = np.asarray(
        [[uid_map[row[0]], iid_map[row[1]]] for row in rows],
        dtype=np.int64,
    )

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("recent_pairs inválido.")

    return pairs


# ============================================================
# Scenario/seed base cache
# ============================================================

def train_frozen_base(scenario, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_train_set = build_dataset(
        scenario["base_rows"],
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
        name=f"IBPR_base_{scenario['name']}_seed{seed}",
    )

    start = time.perf_counter()
    model.fit(base_train_set)
    train_time = time.perf_counter() - start

    return model, base_train_set, train_time


def prepare_prequential_steps(scenario, seed, base_model, base_train_set):
    uid_map = base_train_set.uid_map
    iid_map = base_train_set.iid_map

    observed_rows = list(scenario["base_rows"])
    steps = []

    for update_index in range(N_STREAM_CHUNKS - 1):
        update_chunk_number = update_index + 1
        eval_chunk_number = update_chunk_number + 1

        update_rows = list(scenario["known_chunks"][update_index])
        eval_rows, allwarm_eval_rows, users_exposed = primary_eval_rows_for_step(
            scenario["known_chunks"], update_index
        )

        if ts_max(update_rows) >= ts_min(allwarm_eval_rows):
            raise RuntimeError(
                f"{scenario['name']} point {update_index+1}: update/eval no son estrictamente temporales."
            )

        eval_pairs = rows_pairs(allwarm_eval_rows)
        if eval_pairs & rows_pairs(observed_rows):
            sample = next(iter(eval_pairs & rows_pairs(observed_rows)))
            raise RuntimeError(
                f"{scenario['name']} point {update_index+1}: eval ya observado antes del update: {sample}"
            )

        # Current update chunk becomes observed before evaluating the next chunk.
        post_update_rows = observed_rows + update_rows
        if eval_pairs & rows_pairs(post_update_rows):
            sample = next(iter(eval_pairs & rows_pairs(post_update_rows)))
            raise RuntimeError(
                f"{scenario['name']} point {update_index+1}: eval aparece en historial post-update: {sample}"
            )

        history_train_set = build_dataset(
            post_update_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=False,
        )
        recent_pairs = rows_to_pairs(update_rows, uid_map, iid_map)

        eval_test_set = build_dataset(
            eval_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=True,
        )

        stale_metrics = evaluate_model(
            base_model,
            history_train_set,
            eval_test_set,
        )

        steps.append(
            {
                "eval_point": update_index + 1,
                "update_chunk": update_chunk_number,
                "eval_chunk": eval_chunk_number,
                "history_csr": history_train_set.csr_matrix.copy(),
                "recent_pairs": recent_pairs,
                "eval_train_set": history_train_set,
                "eval_test_set": eval_test_set,
                "eval_rows": eval_rows,
                "allwarm_eval_rows": allwarm_eval_rows,
                "n_users_exposed_to_update": len(users_exposed),
                "stale_metrics": stale_metrics,
            }
        )

        observed_rows = post_update_rows

    return steps


class BaseScenarioCache:
    def __init__(self, scenario_data):
        self.scenario_data = scenario_data
        self.cache = {}

    def get(self, scenario_name, seed):
        key = (scenario_name, int(seed))

        if key in self.cache:
            return self.cache[key]

        scenario = self.scenario_data[scenario_name]

        print(
            f"BASE   {scenario_name} | seed={seed} | "
            f"IBPR k={FROZEN_IBPR_CONFIG['k']} "
            f"iter={FROZEN_IBPR_CONFIG['max_iter']} "
            f"lr={FROZEN_IBPR_CONFIG['learning_rate']} "
            f"lamda={FROZEN_IBPR_CONFIG['lamda']} "
            f"batch={FROZEN_IBPR_CONFIG['batch_size']}"
        )

        base_model, base_train_set, train_time = train_frozen_base(
            scenario,
            seed,
        )

        steps = prepare_prequential_steps(
            scenario,
            seed,
            base_model,
            base_train_set,
        )

        payload = {
            "scenario": scenario,
            "base_model": base_model,
            "base_train_set": base_train_set,
            "base_train_time_s": train_time,
            "steps": steps,
        }

        self.cache[key] = payload

        print(
            f"       base train={train_time:.2f}s | "
            f"prepared_eval_points={len(steps)}"
        )

        return payload


# ============================================================
# Online physical trial
# ============================================================

def instantiate_online_from_base(base_model, base_train_set, config, seed):
    runtime_config = dict(config)
    runtime_config["_trial_seed"] = int(seed)

    model = OnlineIBPRMejorado(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=1,
        learning_rate=config["online_learning_rate"],
        lamda=config["online_lamda"],
        batch_size=config["online_batch_size"],
        init_params={
            "U": np.asarray(base_model.U).copy(),
            "V": np.asarray(base_model.V).copy(),
        },
        update_V=FIXED_UPDATE_V,
        neg_sampling=FIXED_NEG_SAMPLING,
        normalize=FIXED_NORMALIZE,
        loss_mode=config["loss_mode"],
        seed=int(runtime_config["_trial_seed"]),
        verbose=False,
        name=f"OnlineIBPRMejorado_{config['config_id']}",
    )

    # Metadata needed by score()/rank() before calling wrapper partial_fit.
    model.U = np.asarray(base_model.U).copy()
    model.V = np.asarray(base_model.V).copy()
    model.num_users = base_train_set.num_users
    model.num_items = base_train_set.num_items
    model.train_set = base_train_set

    return model


def apply_partial_update(model, step, config):
    """Run HPO updates through the validated public wrapper path."""
    before_v = np.asarray(model.V).copy()

    start = time.perf_counter()

    model.partial_fit_recent(
        recent_pairs=step["recent_pairs"],
        history_csr=step["history_csr"],
        max_steps=FIXED_MAX_STEPS,
        n_epochs=config["n_epochs"],
    )

    elapsed = time.perf_counter() - start

    # Keep recommender metadata aligned with the history now observed.
    model.train_set = step["eval_train_set"]

    exact_equal = bool(np.array_equal(model.V, before_v))
    max_abs_diff = float(
        np.max(np.abs(model.V - before_v))
    ) if model.V.size else 0.0

    if not exact_equal:
        raise AssertionError(
            "Invariante roto: update_V=False modificó V durante el HPO online. "
            f"max_abs_diff={max_abs_diff}. "
            "Verifica que estés usando la versión mejorada del experiments que "
            "preserva V incluso cuando normalize=True."
        )

    return elapsed, exact_equal, max_abs_diff


def physical_trial(
    origin_stage,
    config,
    scenario_name,
    seed,
    base_cache,
):
    payload = base_cache.get(scenario_name, seed)
    scenario = payload["scenario"]
    base_model = payload["base_model"]
    base_train_set = payload["base_train_set"]
    steps = payload["steps"]

    model = instantiate_online_from_base(
        base_model,
        base_train_set,
        config,
        seed,
    )

    initial_v = np.asarray(model.V).copy()

    update_times = []
    chunk_metric_rows = []

    for step_index, step in enumerate(steps):
        update_time, exact_equal_step, max_diff_step = apply_partial_update(
            model,
            step,
            config,
        )

        update_times.append(update_time)

        online_metrics = evaluate_model(
            model,
            step["eval_train_set"],
            step["eval_test_set"],
        )

        stale_metrics = step["stale_metrics"]

        row = {
            **run_meta(),
            "origin_stage": origin_stage,
            "config_id": config["config_id"],
            "seed": seed,
            "scenario": scenario_name,
            "eval_point": step["eval_point"],
            "eval_chunk": step["eval_chunk"],
            "trained_on_chunks": step["update_chunk"],
            "online_learning_rate": config["online_learning_rate"],
            "online_lamda": config["online_lamda"],
            "online_batch_size": config["online_batch_size"],
            "n_epochs": config["n_epochs"],
            "loss_mode": config["loss_mode"],
            "n_eval_rows": len(step["eval_rows"]),
            "n_eval_users": len({row[0] for row in step["eval_rows"]}),
            "n_eval_items": len({row[1] for row in step["eval_rows"]}),
            "n_allwarm_eval_rows": len(step["allwarm_eval_rows"]),
            "n_allwarm_eval_users": len({row[0] for row in step["allwarm_eval_rows"]}),
            "n_users_exposed_to_update": step["n_users_exposed_to_update"],
            "update_time_s_before_eval": update_time,
        }

        for metric in QUALITY_METRICS:
            row[metric] = online_metrics[metric]
            row[f"stale_{metric}"] = stale_metrics[metric]
            row[f"delta_{metric}"] = (
                online_metrics[metric] - stale_metrics[metric]
            )

        chunk_metric_rows.append(row)

    final_v_exact = bool(np.array_equal(model.V, initial_v))
    final_v_max_diff = float(
        np.max(np.abs(model.V - initial_v))
    ) if model.V.size else 0.0

    if not final_v_exact:
        raise AssertionError(
            "Invariante final roto: V cambió respecto del IBPR base."
        )

    base_rows = scenario["base_rows"]
    base_users = {row[0] for row in base_rows}
    base_items = {row[1] for row in base_rows}
    stream_known_rows = [
        row
        for chunk in scenario["known_chunks"]
        for row in chunk
    ]

    trial = {
        **run_meta(),
        "origin_stage": origin_stage,
        "config_id": config["config_id"],
        "config_source": config["config_source"],
        "seed": seed,
        "scenario": scenario_name,
        "base_ratio": scenario["base_ratio"],
        "online_learning_rate": config["online_learning_rate"],
        "online_lamda": config["online_lamda"],
        "online_batch_size": config["online_batch_size"],
        "n_epochs": config["n_epochs"],
        "loss_mode": config["loss_mode"],
        "update_V": FIXED_UPDATE_V,
        "neg_sampling": FIXED_NEG_SAMPLING,
        "normalize": FIXED_NORMALIZE,
        "max_steps": FIXED_MAX_STEPS,
        "n_base": len(base_rows),
        "n_base_users": len(base_users),
        "n_base_items": len(base_items),
        "n_stream_candidate": scenario["n_stream_candidate"],
        "n_stream_known": scenario["n_stream_known"],
        "stream_known_fraction": scenario["stream_known_fraction"],
        "n_stream_users": len({row[0] for row in stream_known_rows}),
        "n_stream_items": len({row[1] for row in stream_known_rows}),
        "n_eval_points": len(chunk_metric_rows),
        "total_update_time_s": float(np.sum(update_times)),
        "mean_update_time_s": float(np.mean(update_times)),
        "max_update_time_s": float(np.max(update_times)),
        "v_exact_equal": final_v_exact,
        "v_max_abs_diff": final_v_max_diff,
    }

    for metric in QUALITY_METRICS:
        trial[f"mean_{metric}"] = float(
            np.mean([row[metric] for row in chunk_metric_rows])
        )
        trial[f"mean_stale_{metric}"] = float(
            np.mean([row[f"stale_{metric}"] for row in chunk_metric_rows])
        )
        trial[f"mean_delta_{metric}"] = float(
            np.mean([row[f"delta_{metric}"] for row in chunk_metric_rows])
        )

    return trial, chunk_metric_rows


# ============================================================
# Resume
# ============================================================

def trial_key_from_config(config, scenario_name, seed):
    return (
        str(config["config_id"]),
        float(config["online_learning_rate"]),
        float(config["online_lamda"]),
        int(config["online_batch_size"]),
        int(config["n_epochs"]),
        str(config["loss_mode"]),
        str(scenario_name),
        int(seed),
    )


def trial_key_from_row(row):
    return (
        str(row["config_id"]),
        float(row["online_learning_rate"]),
        float(row["online_lamda"]),
        int(row["online_batch_size"]),
        int(row["n_epochs"]),
        str(row["loss_mode"]),
        str(row["scenario"]),
        int(row["seed"]),
    )


def build_trial_lookup(rows):
    lookup = {}

    for row in rows:
        key = trial_key_from_row(row)

        if key in lookup:
            raise ValueError(f"Trial online duplicado en CSV: {key}")

        lookup[key] = row

    return lookup


def persist_completed_chunk_rows(chunk_path, config_id, scenario_name, seed, rows):
    """Keep chunk CSV consistent if a previous run crashed mid-trial."""
    existing = load_csv(chunk_path)

    kept = [
        row
        for row in existing
        if not (
            str(row.get("config_id")) == str(config_id)
            and str(row.get("scenario")) == str(scenario_name)
            and int(row.get("seed")) == int(seed)
        )
    ]

    save_csv(
        chunk_path,
        CHUNK_FIELDS,
        kept + rows,
    )


def ensure_trial(
    origin_stage,
    config,
    scenario_name,
    seed,
    base_cache,
    trials_path,
    chunk_path,
    trial_rows,
    trial_lookup,
):
    key = trial_key_from_config(config, scenario_name, seed)

    if key in trial_lookup:
        print(
            "REUSE  "
            f"{config['config_id']} | {scenario_name} | seed={seed}"
        )
        return trial_lookup[key]

    print(
        "RUN    "
        f"{config['config_id']} | "
        f"lr={config['online_learning_rate']} "
        f"lamda={config['online_lamda']} "
        f"batch={config['online_batch_size']} "
        f"epochs={config['n_epochs']} "
        f"loss={config['loss_mode']} | "
        f"{scenario_name} | seed={seed}"
    )

    row, completed_chunk_rows = physical_trial(
        origin_stage,
        config,
        scenario_name,
        seed,
        base_cache,
    )

    # Persist chunk details only after the full physical trial has succeeded.
    # Existing rows for this trial are replaced, preventing duplicates after
    # interruption/resume.
    persist_completed_chunk_rows(
        chunk_path,
        config["config_id"],
        scenario_name,
        seed,
        completed_chunk_rows,
    )

    append_csv(trials_path, TRIAL_FIELDS, row)
    trial_rows.append(row)
    trial_lookup[key] = row

    print(
        f"       mean ΔNDCG@{TOP_K}="
        f"{float(row[f'mean_delta_NDCG@{TOP_K}']):+.6f} | "
        f"online NDCG@{TOP_K}="
        f"{float(row[f'mean_NDCG@{TOP_K}']):.6f} | "
        f"updates={float(row['total_update_time_s']):.4f}s | "
        f"V_equal={row['v_exact_equal']}"
    )

    return row


# ============================================================
# Aggregation / ranking
# ============================================================

def matching_trial_rows(trial_rows, config, scenarios, seeds):
    expected = {
        trial_key_from_config(config, scenario, seed)
        for scenario in scenarios
        for seed in seeds
    }

    selected = [
        row
        for row in trial_rows
        if trial_key_from_row(row) in expected
    ]

    if len(selected) != len(expected):
        raise ValueError(
            f"Faltan trials para {config['config_id']}: "
            f"esperados={len(expected)}, encontrados={len(selected)}."
        )

    return selected


def aggregate_config(stage, config, scenarios, seeds, trial_rows):
    rows = matching_trial_rows(
        trial_rows,
        config,
        scenarios,
        seeds,
    )

    summary = {
        **run_meta(),
        "stage": stage,
        "config_id": config["config_id"],
        "config_source": config["config_source"],
        "online_learning_rate": config["online_learning_rate"],
        "online_lamda": config["online_lamda"],
        "online_batch_size": config["online_batch_size"],
        "n_epochs": config["n_epochs"],
        "loss_mode": config["loss_mode"],
        "seeds": ",".join(str(seed) for seed in seeds),
        "scenarios": ",".join(scenarios),
        "n_runs": len(rows),
    }

    for metric in QUALITY_METRICS:
        values = [
            float(row[f"mean_{metric}"])
            for row in rows
        ]
        mean_value, std_value = mean_std(values)
        summary[f"mean_{metric}"] = mean_value
        summary[f"std_{metric}"] = std_value

        delta_values = [
            float(row[f"mean_delta_{metric}"])
            for row in rows
        ]
        mean_delta, std_delta = mean_std(delta_values)
        summary[f"mean_delta_{metric}"] = mean_delta
        summary[f"std_delta_{metric}"] = std_delta

    mean_time, std_time = mean_std(
        [float(row["total_update_time_s"]) for row in rows]
    )

    summary["mean_total_update_time_s"] = mean_time
    summary["std_total_update_time_s"] = std_time
    summary["all_v_exact_equal"] = all(
        str(row["v_exact_equal"]).lower() in {"true", "1"}
        for row in rows
    )
    summary["max_v_abs_diff"] = max(
        float(row["v_max_abs_diff"]) for row in rows
    )

    return summary


def ranking_key(row):
    return (
        -float(row[f"mean_delta_NDCG@{TOP_K}"]),
        -float(row[f"mean_NDCG@{TOP_K}"]),
        float(row[f"std_delta_NDCG@{TOP_K}"]),
        float(row["mean_total_update_time_s"]),
        int(row["n_epochs"]),
    )


def rank_summaries(rows):
    ranked = sorted(rows, key=ranking_key)

    output = []

    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = rank
        output.append(item)

    return output


def print_ranking(title, ranked):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    for row in ranked:
        print(
            f"#{int(row['rank']):02d} "
            f"{row['config_id']} | "
            f"lr={row['online_learning_rate']} "
            f"lamda={row['online_lamda']} "
            f"batch={row['online_batch_size']} "
            f"epochs={row['n_epochs']} "
            f"loss={row['loss_mode']} | "
            f"ΔNDCG@{TOP_K}="
            f"{float(row[f'mean_delta_NDCG@{TOP_K}']):+.6f} "
            f"± {float(row[f'std_delta_NDCG@{TOP_K}']):.6f} | "
            f"NDCG@{TOP_K}="
            f"{float(row[f'mean_NDCG@{TOP_K}']):.6f} | "
            f"update={float(row['mean_total_update_time_s']):.4f}s"
        )


def config_map(configs):
    return {config["config_id"]: config for config in configs}


# ============================================================
# Stages
# ============================================================

def run_stage_a(
    configs,
    base_cache,
    trials_path,
    chunk_path,
    trial_rows,
    trial_lookup,
):
    scenarios = ["S50", "S65"]
    seeds = [SCREENING_SEED]

    for config in configs:
        for scenario in scenarios:
            ensure_trial(
                "online_stage_A",
                config,
                scenario,
                SCREENING_SEED,
                base_cache,
                trials_path,
                chunk_path,
                trial_rows,
                trial_lookup,
            )

    summaries = [
        aggregate_config(
            "online_stage_A",
            config,
            scenarios,
            seeds,
            trial_rows,
        )
        for config in configs
    ]

    return rank_summaries(summaries)


def run_stage_b(
    stage_a_ranked,
    configs,
    base_cache,
    trials_path,
    chunk_path,
    trial_rows,
    trial_lookup,
):
    cmap = config_map(configs)

    selected_ids = unique_preserving_order(
        [row["config_id"] for row in stage_a_ranked[:TOP_STAGE_A]]
        + protected_config_ids(configs)
    )

    for config_id in selected_ids:
        ensure_trial(
            "online_stage_B",
            cmap[config_id],
            "S80",
            SCREENING_SEED,
            base_cache,
            trials_path,
            chunk_path,
            trial_rows,
            trial_lookup,
        )

    summaries = [
        aggregate_config(
            "online_stage_B",
            cmap[config_id],
            ["S50", "S65", "S80"],
            [SCREENING_SEED],
            trial_rows,
        )
        for config_id in selected_ids
    ]

    return rank_summaries(summaries)


def run_stage_c(
    stage_b_ranked,
    configs,
    base_cache,
    trials_path,
    chunk_path,
    trial_rows,
    trial_lookup,
):
    cmap = config_map(configs)

    selected_ids = unique_preserving_order(
        [row["config_id"] for row in stage_b_ranked[:TOP_STAGE_B]]
        + protected_config_ids(configs)
    )

    scenarios = ["S50", "S65", "S80"]

    for config_id in selected_ids:
        config = cmap[config_id]

        for seed in CONFIRMATION_SEEDS:
            for scenario in scenarios:
                ensure_trial(
                    "online_stage_C",
                    config,
                    scenario,
                    seed,
                    base_cache,
                    trials_path,
                    chunk_path,
                    trial_rows,
                    trial_lookup,
                )

    summaries = [
        aggregate_config(
            "online_stage_C",
            cmap[config_id],
            scenarios,
            CONFIRMATION_SEEDS,
            trial_rows,
        )
        for config_id in selected_ids
    ]

    return rank_summaries(summaries)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    validate_stabilized_online_contract()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = (
        args.timestamp
        if args.timestamp
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    trials_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_online_ibpr_mejorado_v2_trials_{timestamp}.csv",
    )
    chunks_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_online_ibpr_mejorado_v2_chunks_{timestamp}.csv",
    )
    summary_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_online_ibpr_mejorado_v2_summary_{timestamp}.csv",
    )
    best_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_online_ibpr_mejorado_v2_best_{timestamp}.csv",
    )
    log_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_online_ibpr_mejorado_v2_{timestamp}.txt",
    )

    log_mode = "a" if os.path.exists(log_path) else "w"

    with open(log_path, log_mode, encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)

        with redirect_stdout(tee):
            print()
            print("#" * 110)
            print(f"ONLINE IBPR HPO V2 TIMESTAMP: {timestamp}")
            print("#" * 110)
            print(f"Trials CSV : {trials_path}")
            print(f"Chunks CSV : {chunks_path}")
            print(f"Summary CSV: {summary_path}")
            print(f"Best CSV   : {best_path}")
            print()

            all_rows = load_positive_chrono_movielens()
            data_hash = dataset_sha256(all_rows)
            hpo_info = build_hpo_pool(all_rows)
            hpo_pool = hpo_info["hpo_pool"]

            configs = build_screening_configs(args.screening_configs)
            protocol_hash = current_protocol_hash(data_hash, configs)
            RUN_CONTEXT["protocol_hash"] = protocol_hash
            RUN_CONTEXT["data_sha256"] = data_hash

            scenario_data = {
                scenario["name"]: build_stream_scenario(hpo_pool, scenario)
                for scenario in STREAM_SCENARIOS
            }

            print_data_plan(
                all_rows,
                hpo_info,
                scenario_data,
                data_hash,
                protocol_hash,
            )

            print_candidate_plan(configs)

            if args.plan_only:
                print("PLAN ONLY: no se entrenó ningún modelo.")
                return

            trial_rows = load_csv(trials_path)
            chunk_rows_existing = load_csv(chunks_path)
            validate_resume_protocol(trial_rows, trials_path, protocol_hash, data_hash)
            validate_resume_protocol(chunk_rows_existing, chunks_path, protocol_hash, data_hash)
            trial_lookup = build_trial_lookup(trial_rows)

            if trial_rows:
                print(
                    f"Resume: {len(trial_rows)} trials online ya disponibles."
                )
                print()

            base_cache = BaseScenarioCache(scenario_data)

            stage_a = run_stage_a(
                configs,
                base_cache,
                trials_path,
                chunks_path,
                trial_rows,
                trial_lookup,
            )

            print_ranking(
                "STAGE O-A - ONLINE SCREENING",
                stage_a,
            )

            stage_b = run_stage_b(
                stage_a,
                configs,
                base_cache,
                trials_path,
                chunks_path,
                trial_rows,
                trial_lookup,
            )

            print_ranking(
                "STAGE O-B - THREE TEMPORAL SCENARIOS",
                stage_b,
            )

            stage_c = run_stage_c(
                stage_b,
                configs,
                base_cache,
                trials_path,
                chunks_path,
                trial_rows,
                trial_lookup,
            )

            print_ranking(
                "STAGE O-C - MULTI-SEED FINAL CONFIRMATION",
                stage_c,
            )

            all_summary_rows = stage_a + stage_b + stage_c

            save_csv(
                summary_path,
                SUMMARY_FIELDS,
                all_summary_rows,
            )

            best = dict(stage_c[0])

            BEST_FIELDS = list(SUMMARY_FIELDS) + [
                "selection_metric",
                "frozen_ibpr_k",
                "frozen_ibpr_max_iter",
                "frozen_ibpr_learning_rate",
                "frozen_ibpr_lamda",
                "frozen_ibpr_batch_size",
                "fixed_update_V",
                "fixed_neg_sampling",
                "fixed_normalize",
                "fixed_max_steps",
                "validation_protocol",
            ]

            best["selection_metric"] = f"mean_delta_NDCG@{TOP_K}"
            best["frozen_ibpr_k"] = FROZEN_IBPR_CONFIG["k"]
            best["frozen_ibpr_max_iter"] = FROZEN_IBPR_CONFIG["max_iter"]
            best["frozen_ibpr_learning_rate"] = (
                FROZEN_IBPR_CONFIG["learning_rate"]
            )
            best["frozen_ibpr_lamda"] = FROZEN_IBPR_CONFIG["lamda"]
            best["frozen_ibpr_batch_size"] = (
                FROZEN_IBPR_CONFIG["batch_size"]
            )
            best["fixed_update_V"] = FIXED_UPDATE_V
            best["fixed_neg_sampling"] = FIXED_NEG_SAMPLING
            best["fixed_normalize"] = FIXED_NORMALIZE
            best["fixed_max_steps"] = FIXED_MAX_STEPS
            best["validation_protocol"] = (
                "first_60pct_global_holdout_then_global_prequential_development_scenarios"
            )

            save_csv(
                best_path,
                BEST_FIELDS,
                [best],
            )

            print()
            print("=" * 110)
            print("SELECTED ONLINE ADAPTATION CONFIGURATION")
            print("=" * 110)
            print(
                "ONLINE_ADAPTATION_CONFIG = {\n"
                f'    "learning_rate": {best["online_learning_rate"]},\n'
                f'    "lamda": {best["online_lamda"]},\n'
                f'    "batch_size": {int(best["online_batch_size"])},\n'
                f'    "n_epochs": {int(best["n_epochs"])},\n'
                f'    "loss_mode": "{best["loss_mode"]}",\n'
                f'    "update_V": {FIXED_UPDATE_V},\n'
                f'    "neg_sampling": "{FIXED_NEG_SAMPLING}",\n'
                f'    "normalize": {FIXED_NORMALIZE},\n'
                f'    "max_steps": {FIXED_MAX_STEPS},\n'
                "}"
            )
            print()
            print(
                f"Final mean ΔNDCG@{TOP_K}: "
                f"{float(best[f'mean_delta_NDCG@{TOP_K}']):+.6f} "
                f"± {float(best[f'std_delta_NDCG@{TOP_K}']):.6f}"
            )
            print(
                f"Final mean NDCG@{TOP_K}: "
                f"{float(best[f'mean_NDCG@{TOP_K}']):.6f}"
            )
            print(
                f"Mean total partial-update time per scenario: "
                f"{float(best['mean_total_update_time_s']):.4f}s"
            )
            print(
                f"All V exact equal: {best['all_v_exact_equal']} | "
                f"max V diff={float(best['max_v_abs_diff']):.12g}"
            )
            print()
            if float(best[f"mean_delta_NDCG@{TOP_K}"]) <= 0.0:
                print(
                    "WARNING: the best development configuration did not "
                    "improve mean NDCG over IBPR_STALE. Do not claim H1 support "
                    "from HPO; freeze the configuration and let the untouched "
                    "final H1-H3 evaluation determine the conclusion."
                )
            else:
                print(
                    "HPO online V2 completed with positive development delta. "
                    "Freeze this adaptation configuration before running final H1-H3."
                )


if __name__ == "__main__":
    main()
