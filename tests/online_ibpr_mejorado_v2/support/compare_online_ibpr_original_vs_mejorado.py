import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
import time
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import torch
import cornac
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.models.online_ibpr_mejorado.online_ibpr_mejorado import (
    online_ibpr_mejorado as improved_online_core,
)

try:
    from cornac.models.online_ibpr.recom_online_ibpr import OnlineIBPR as OnlineIBPROriginal
    from cornac.models.online_ibpr.online_ibpr import online_ibpr as original_online_core
except ImportError:
    # Fallback only if the package exports the original class at top-level.
    from cornac.models import OnlineIBPR as OnlineIBPROriginal
    original_online_core = None


# ============================================================
# Frozen diagnostic protocol
# ============================================================

RATING_THRESHOLD = 3.0
VARIANT = "1M"
TOP_K = 20
HPO_END_FRAC = 0.60
N_STREAM_CHUNKS = 4

STREAM_SCENARIOS = [
    {"name": "S50", "base_ratio": 0.50},
    {"name": "S65", "base_ratio": 0.65},
    {"name": "S80", "base_ratio": 0.80},
]

# Deliberately distinct from:
#   online HPO seeds [42, 123, 2024]
#   final H1-H3 seeds [777, 999]
DIAGNOSTIC_SEEDS = [31415, 27182, 16180]

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

# Closest numeric settings shared by the original public API.
# Important: the original implementation itself is NOT modified.
ORIGINAL_DIAGNOSTIC_CONFIG = {
    "k": FROZEN_IBPR_CONFIG["k"],
    "learning_rate": FROZEN_ONLINE_CONFIG["learning_rate"],
    "lamda": FROZEN_ONLINE_CONFIG["lamda"],
    "batch_size": FROZEN_ONLINE_CONFIG["batch_size"],
    "max_iter": FROZEN_ONLINE_CONFIG["n_epochs"],
}

QUALITY_METRICS = [
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

PROTOCOL_VERSION = "original_vs_mejorado_diag_v2"


STEP_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "seed",
    "scenario",
    "base_ratio",
    "update_chunk",
    "eval_chunk",
    "n_update_rows",
    "n_eval_rows",
    "n_eval_users",
    "n_eval_items",
    "original_update_time_s_descriptive",
    "improved_update_time_s_descriptive",
    "original_j_unique_indices",
    "original_j_raw_items",
    "original_j_equals_positive_fraction",
    "original_j_known_positive_fraction",
    "original_v_exact_equal_base",
    "original_v_max_abs_diff_base",
    "improved_v_exact_equal_base",
    "improved_v_max_abs_diff_base",
    "original_mean_u_norm_all",
    "improved_mean_u_norm_all",
    "original_mean_u_norm_affected",
    "improved_mean_u_norm_affected",
]

for metric in QUALITY_METRICS:
    STEP_FIELDS.extend(
        [
            f"stale_{metric}",
            f"original_{metric}",
            f"improved_{metric}",
            f"original_minus_stale_{metric}",
            f"improved_minus_stale_{metric}",
            f"improved_minus_original_{metric}",
        ]
    )


TRIAL_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "seed",
    "scenario",
    "base_ratio",
    "n_base",
    "n_base_users",
    "n_base_items",
    "n_stream_candidate",
    "n_stream_known",
    "stream_known_fraction",
    "n_eval_points",
    "base_train_time_s",
    "total_original_update_time_s_descriptive",
    "total_improved_update_time_s_descriptive",
    "descriptive_original_over_improved_time_ratio",
    "mean_original_j_equals_positive_fraction",
    "mean_original_j_known_positive_fraction",
    "all_original_v_exact_equal_base",
    "max_original_v_abs_diff_base",
    "all_improved_v_exact_equal_base",
    "max_improved_v_abs_diff_base",
]

for metric in QUALITY_METRICS:
    TRIAL_FIELDS.extend(
        [
            f"mean_stale_{metric}",
            f"mean_original_{metric}",
            f"mean_improved_{metric}",
            f"mean_original_minus_stale_{metric}",
            f"mean_improved_minus_stale_{metric}",
            f"mean_improved_minus_original_{metric}",
        ]
    )


SUMMARY_FIELDS = [
    "protocol_version",
    "protocol_hash",
    "n_trials",
    "n_paired_points",
    "scenarios",
    "seeds",
    "mean_total_original_update_time_s_descriptive",
    "std_total_original_update_time_s_descriptive",
    "mean_total_improved_update_time_s_descriptive",
    "std_total_improved_update_time_s_descriptive",
    "mean_descriptive_original_over_improved_time_ratio",
    "mean_original_j_equals_positive_fraction",
    "std_original_j_equals_positive_fraction",
    "mean_original_j_known_positive_fraction",
    "std_original_j_known_positive_fraction",
    "all_original_v_exact_equal_base",
    "max_original_v_abs_diff_base",
    "all_improved_v_exact_equal_base",
    "max_improved_v_abs_diff_base",
]

for metric in QUALITY_METRICS:
    SUMMARY_FIELDS.extend(
        [
            f"mean_stale_{metric}",
            f"std_stale_{metric}",
            f"mean_original_{metric}",
            f"std_original_{metric}",
            f"mean_improved_{metric}",
            f"std_improved_{metric}",
            f"mean_original_minus_stale_{metric}",
            f"std_original_minus_stale_{metric}",
            f"mean_improved_minus_stale_{metric}",
            f"std_improved_minus_stale_{metric}",
            f"mean_improved_minus_original_{metric}",
            f"std_improved_minus_original_{metric}",
            f"positive_original_minus_stale_{metric}",
            f"positive_improved_minus_stale_{metric}",
            f"positive_improved_minus_original_{metric}",
        ]
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic comparison of the original OnlineIBPR implementation "
            "against frozen OnlineIBPRMejorado O014 on the first 60% "
            "MovieLens 1M development horizon."
        )
    )

    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Timestamp YYYYMMDD_HHMMSS. Reusing an existing timestamp resumes "
            "completed scenario/seed trials."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print and validate the diagnostic protocol without training models.",
    )

    return parser.parse_args()


# ============================================================
# Logging / CSV
# ============================================================

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


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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



# ============================================================
# Protocol fingerprint / resume safety
# ============================================================

def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_sha256(obj):
    try:
        return _sha256_text(inspect.getsource(obj))
    except (OSError, TypeError):
        return "unavailable"


def protocol_payload():
    """Canonical protocol identity used to protect resume/reuse."""
    script_path = os.path.abspath(__file__)

    try:
        with open(script_path, "rb") as f:
            script_sha256 = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        script_sha256 = "unavailable"

    return {
        "protocol_version": PROTOCOL_VERSION,
        "variant": VARIANT,
        "rating_threshold": RATING_THRESHOLD,
        "hpo_end_frac": HPO_END_FRAC,
        "n_stream_chunks": N_STREAM_CHUNKS,
        "stream_scenarios": STREAM_SCENARIOS,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "frozen_ibpr_config": FROZEN_IBPR_CONFIG,
        "frozen_online_config": FROZEN_ONLINE_CONFIG,
        "original_diagnostic_config": ORIGINAL_DIAGNOSTIC_CONFIG,
        "quality_metrics": QUALITY_METRICS,
        "script_sha256": script_sha256,
        "original_wrapper_sha256": _source_sha256(OnlineIBPROriginal),
        "original_core_sha256": (
            _source_sha256(original_online_core)
            if original_online_core is not None
            else "unavailable"
        ),
        "improved_wrapper_sha256": _source_sha256(OnlineIBPRMejorado),
        "improved_core_sha256": _source_sha256(improved_online_core),
    }


def current_protocol_hash():
    canonical = json.dumps(
        protocol_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_resume_protocol(rows, path, protocol_hash):
    """Refuse to mix rows from another protocol/configuration under same timestamp."""
    if not rows:
        return

    missing = [
        index
        for index, row in enumerate(rows)
        if not row.get("protocol_version") or not row.get("protocol_hash")
    ]

    if missing:
        raise RuntimeError(
            f"{path}: contiene resultados sin fingerprint de protocolo. "
            "No se permite reutilizarlos con esta versión del script. "
            "Usa un timestamp nuevo."
        )

    versions = {str(row["protocol_version"]) for row in rows}
    hashes = {str(row["protocol_hash"]) for row in rows}

    if versions != {PROTOCOL_VERSION} or hashes != {protocol_hash}:
        raise RuntimeError(
            f"{path}: fingerprint incompatible con el protocolo actual. "
            f"versions={sorted(versions)}, hashes={sorted(hashes)}. "
            "Usa un timestamp nuevo; no mezcles ejecuciones."
        )


# ============================================================
# Implementation guards
# ============================================================

def validate_implementations():
    improved_ctor = set(inspect.signature(OnlineIBPRMejorado.__init__).parameters)
    improved_partial = set(
        inspect.signature(OnlineIBPRMejorado.partial_fit_recent).parameters
    )

    if "seed" not in improved_ctor:
        raise RuntimeError(
            "OnlineIBPRMejorado obsoleto: el constructor no expone `seed`."
        )

    required_partial = {"recent_pairs", "history_csr", "max_steps", "n_epochs"}
    missing = required_partial - improved_partial

    if missing:
        raise RuntimeError(
            "OnlineIBPRMejorado obsoleto: partial_fit_recent no cumple el "
            f"contrato estabilizado. Faltan {sorted(missing)}."
        )

    if not hasattr(OnlineIBPROriginal, "fit"):
        raise RuntimeError("OnlineIBPR original no expone fit(...).")

    improved_source = inspect.getsource(improved_online_core)
    improved_expected_tokens = [
        "_sample_negatives_uniform",
        "history_csr",
        "recent_pairs",
        "update_V",
        "torch.nn.functional.normalize",
    ]
    missing_improved_tokens = [
        token
        for token in improved_expected_tokens
        if token not in improved_source
    ]

    if missing_improved_tokens:
        raise RuntimeError(
            "La implementación OnlineIBPRMejorado encontrada no coincide con "
            "el experiments estabilizado esperado. Tokens ausentes: "
            f"{missing_improved_tokens}."
        )

    # Guard against silently comparing a different implementation under the
    # same class name. This is intentionally semantic and conservative.
    if original_online_core is not None:
        source = inspect.getsource(original_online_core)

        expected_tokens = [
            "X.tocoo()",
            "triplets[:, 2] = X.data",
            "V[triplets[:, 2], :]",
            "torch.optim.Adam([U]",
        ]

        missing_tokens = [token for token in expected_tokens if token not in source]

        if missing_tokens:
            raise RuntimeError(
                "La implementación OnlineIBPR encontrada no coincide con la "
                "versión original auditada. Tokens ausentes: "
                f"{missing_tokens}. No ejecutes esta comparativa hasta revisar "
                "qué versión está instalada."
            )

    print("Implementation contract: OK")
    print(
        "  Original wrapper : "
        f"{inspect.getsourcefile(OnlineIBPROriginal) or 'unknown'}"
    )
    print(
        "  Improved wrapper : "
        f"{inspect.getsourcefile(OnlineIBPRMejorado) or 'unknown'}"
    )
    if original_online_core is not None:
        print(
            "  Original experiments    : "
            f"{inspect.getsourcefile(original_online_core) or 'unknown'}"
        )
    print()


# ============================================================
# Data preparation
# ============================================================

def load_positive_chrono_movielens():
    data = movielens.load_feedback(fmt="UIRT", variant=VARIANT)

    positive = [
        (str(u), str(i), 1.0, int(ts))
        for u, i, r, ts in data
        if float(r) >= RATING_THRESHOLD
    ]

    positive.sort(key=lambda row: row[3])

    return positive


def build_hpo_pool(all_positive_rows):
    end = int(len(all_positive_rows) * HPO_END_FRAC)

    if end <= 0:
        raise ValueError("El horizonte de desarrollo quedó vacío.")

    return list(all_positive_rows[:end])


def build_user_histories(rows):
    histories = {}

    for row in rows:
        histories.setdefault(row[0], []).append(row)

    for history in histories.values():
        history.sort(key=lambda row: row[3])

    return histories


def split_future_into_chunks(future_rows, n_chunks=N_STREAM_CHUNKS):
    n = len(future_rows)

    if n < n_chunks:
        raise ValueError(
            f"Se requieren al menos {n_chunks} interacciones futuras, recibidas={n}."
        )

    index_chunks = np.array_split(np.arange(n), n_chunks)

    return [
        [future_rows[int(index)] for index in indices]
        for indices in index_chunks
    ]


def build_stream_scenario(user_histories, scenario):
    base_ratio = float(scenario["base_ratio"])

    if not (0.0 < base_ratio < 1.0):
        raise ValueError(f"base_ratio inválido: {base_ratio}")

    base_rows = []
    candidate_chunks = [[] for _ in range(N_STREAM_CHUNKS)]
    eligible_users = set()

    for user_id, history in user_histories.items():
        n = len(history)

        if n == 0:
            continue

        base_end = int(np.floor(n * base_ratio))
        base_end = max(1, min(base_end, n))
        future = history[base_end:]

        if len(future) >= N_STREAM_CHUNKS:
            eligible_users.add(user_id)
            base_rows.extend(history[:base_end])

            user_chunks = split_future_into_chunks(future)

            for chunk_index, chunk_rows in enumerate(user_chunks):
                candidate_chunks[chunk_index].extend(chunk_rows)
        else:
            base_rows.extend(history[:base_end])

    base_rows.sort(key=lambda row: row[3])

    if not base_rows:
        raise ValueError(f"{scenario['name']}: base vacío.")

    known_users = {u for u, _, _, _ in base_rows}
    known_items = {i for _, i, _, _ in base_rows}

    known_chunks = []

    for candidate in candidate_chunks:
        candidate.sort(key=lambda row: row[3])

        known = [
            row
            for row in candidate
            if row[0] in known_users and row[1] in known_items
        ]

        known_chunks.append(known)

    if any(len(chunk) == 0 for chunk in known_chunks):
        raise ValueError(
            f"{scenario['name']}: al menos un chunk quedó vacío "
            "después del filtrado warm-start."
        )

    # Same per-user temporal audit used in the online HPO protocol.
    last_ts_by_user = {}

    for u, _, _, ts in base_rows:
        previous = last_ts_by_user.get(u)
        if previous is None or ts > previous:
            last_ts_by_user[u] = ts

    for chunk_index, chunk in enumerate(known_chunks, start=1):
        min_ts_in_chunk = {}
        max_ts_in_chunk = {}

        for u, _, _, ts in chunk:
            min_ts_in_chunk[u] = min(min_ts_in_chunk.get(u, ts), ts)
            max_ts_in_chunk[u] = max(max_ts_in_chunk.get(u, ts), ts)

        for user_id, min_ts in min_ts_in_chunk.items():
            if user_id in last_ts_by_user and last_ts_by_user[user_id] > min_ts:
                raise ValueError(
                    f"{scenario['name']} chunk {chunk_index}: fuga temporal "
                    f"detectada para usuario {user_id}."
                )

        for user_id, max_ts in max_ts_in_chunk.items():
            last_ts_by_user[user_id] = max_ts

    candidate_total = sum(len(chunk) for chunk in candidate_chunks)
    known_total = sum(len(chunk) for chunk in known_chunks)

    return {
        "name": scenario["name"],
        "base_ratio": base_ratio,
        "base_rows": base_rows,
        "candidate_chunks": candidate_chunks,
        "known_chunks": known_chunks,
        "eligible_users": eligible_users,
        "n_stream_candidate": candidate_total,
        "n_stream_known": known_total,
        "stream_known_fraction": (
            known_total / candidate_total if candidate_total else 0.0
        ),
    }


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

    return Dataset.build(rows, **kwargs)


def rows_to_pairs(rows, uid_map, iid_map):
    pairs = np.asarray(
        [[uid_map[u], iid_map[i]] for u, i, _, _ in rows],
        dtype=np.int64,
    )

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("recent_pairs inválido.")

    return pairs


# ============================================================
# Evaluation
# ============================================================

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


def mean_user_norm(U):
    U = np.asarray(U)

    if U.size == 0:
        return 0.0

    return float(np.mean(np.linalg.norm(U, axis=1)))


def mean_affected_user_norm(U, recent_pairs):
    U = np.asarray(U)

    if U.size == 0 or len(recent_pairs) == 0:
        return 0.0

    users = np.unique(np.asarray(recent_pairs)[:, 0].astype(np.int64))

    return float(np.mean(np.linalg.norm(U[users], axis=1)))


def invert_map(raw_to_internal):
    return {int(internal): raw for raw, internal in raw_to_internal.items()}


def diagnose_original_j(original_update_set, recent_pairs, history_csr, iid_map):
    """
    Reproduce what the original experiments interprets as j:
        j := X.data
    and quantify how often that j is invalid under BPR semantics.
    """
    X = original_update_set.matrix.tocoo()
    values = np.asarray(X.data, dtype=np.float64)

    if values.size == 0:
        raise AssertionError("El update set original quedó vacío.")

    if not np.all(np.isfinite(values)):
        raise AssertionError("X.data del baseline original contiene valores no finitos.")

    rounded = np.rint(values)

    if not np.allclose(values, rounded):
        raise AssertionError(
            "El baseline original usaría valores no enteros como índices j. "
            f"Valores únicos={np.unique(values).tolist()}"
        )

    j_indices = rounded.astype(np.int64)

    n_items = history_csr.shape[1]

    if np.any(j_indices < 0) or np.any(j_indices >= n_items):
        raise AssertionError(
            "El baseline original produciría índices j fuera de rango: "
            f"min={int(j_indices.min())}, max={int(j_indices.max())}, "
            f"n_items={n_items}."
        )

    pairs = np.asarray(recent_pairs, dtype=np.int64)

    if len(j_indices) != len(pairs):
        raise AssertionError(
            "No coincide la cantidad de valores j originales con recent_pairs: "
            f"j={len(j_indices)}, pairs={len(pairs)}."
        )

    # Dataset COO order and recent_pairs order are expected to refer to the same
    # interactions, but we avoid relying on row order for the invalid-negative
    # statistic: map each update interaction by (u, i).
    # For implicit MovieLens each pair appears at most once in a chunk.
    j_by_pair = {
        (int(u), int(i)): int(j)
        for u, i, j in zip(X.row, X.col, j_indices)
    }

    if len(j_by_pair) != len(j_indices):
        raise AssertionError(
            "El update chunk contiene pares (u,i) duplicados después de Dataset.build; "
            "el diagnóstico de j requiere revisar agregación."
        )

    missing_pairs = [
        (int(u), int(i))
        for u, i in pairs
        if (int(u), int(i)) not in j_by_pair
    ]

    if missing_pairs:
        raise AssertionError(
            "No fue posible alinear recent_pairs con el COO del baseline original. "
            f"Ejemplos={missing_pairs[:5]}"
        )

    aligned_j = np.asarray(
        [j_by_pair[(int(u), int(i))] for u, i in pairs],
        dtype=np.int64,
    )

    equals_positive = aligned_j == pairs[:, 1]

    known_positive = np.asarray(
        [
            history_csr[int(u), int(j)] != 0
            for (u, _), j in zip(pairs, aligned_j)
        ],
        dtype=bool,
    )

    inverse_iid = invert_map(iid_map)
    unique_indices = sorted(set(map(int, aligned_j.tolist())))
    raw_items = [str(inverse_iid.get(j, f"<missing:{j}>")) for j in unique_indices]

    return {
        "unique_indices": unique_indices,
        "raw_items": raw_items,
        "equals_positive_fraction": float(np.mean(equals_positive)),
        "known_positive_fraction": float(np.mean(known_positive)),
    }


def assert_same_model_universe(model, base_train_set, base_u, base_v, uid_map, iid_map, label):
    if int(model.num_users) != int(base_train_set.num_users):
        raise AssertionError(
            f"{label}: num_users cambió: {model.num_users} vs {base_train_set.num_users}."
        )

    if int(model.num_items) != int(base_train_set.num_items):
        raise AssertionError(
            f"{label}: num_items cambió: {model.num_items} vs {base_train_set.num_items}."
        )

    if np.asarray(model.U).shape != np.asarray(base_u).shape:
        raise AssertionError(
            f"{label}: shape U cambió: {np.asarray(model.U).shape} vs {np.asarray(base_u).shape}."
        )

    if np.asarray(model.V).shape != np.asarray(base_v).shape:
        raise AssertionError(
            f"{label}: shape V cambió: {np.asarray(model.V).shape} vs {np.asarray(base_v).shape}."
        )

    train_set = getattr(model, "train_set", None)

    if train_set is None:
        raise AssertionError(f"{label}: train_set ausente después del update.")

    if dict(train_set.uid_map) != dict(uid_map):
        raise AssertionError(f"{label}: uid_map cambió después del update.")

    if dict(train_set.iid_map) != dict(iid_map):
        raise AssertionError(f"{label}: iid_map cambió después del update.")


# ============================================================
# Plan
# ============================================================

def print_plan(all_rows, hpo_pool, scenario_data, protocol_hash):
    print("=" * 110)
    print("DIAGNOSTIC COMPARISON - ORIGINAL ONLINEIBPR VS ONLINEIBPRMEJORADO")
    print("=" * 110)
    print(f"Dataset                         : MovieLens {VARIANT}")
    print(f"Positive threshold              : rating >= {RATING_THRESHOLD}")
    print("Feedback                        : implicit positive = 1.0")
    print(f"Development horizon             : first {HPO_END_FRAC:.0%} global chronological")
    print("Later 40% used here             : NO")
    print(f"Total implicit-positive rows    : {len(all_rows):,}")
    print(f"Rows in diagnostic horizon      : {len(hpo_pool):,}")
    print(f"Diagnostic seeds                : {DIAGNOSTIC_SEEDS}")
    print(f"Protocol version                : {PROTOCOL_VERSION}")
    print(f"Protocol hash                   : {protocol_hash}")
    print(f"Online-HPO seeds reused         : NO")
    print(f"Final H1-H3 seeds reused        : NO")
    print()

    print("Frozen IBPR R900:")
    print(f"  {FROZEN_IBPR_CONFIG}")
    print()

    print("Original OnlineIBPR diagnostic numeric config:")
    print(f"  {ORIGINAL_DIAGNOSTIC_CONFIG}")
    print("  original code is executed literally; no fixes are applied")
    print()

    print("Frozen OnlineIBPRMejorado O014:")
    print(f"  {FROZEN_ONLINE_CONFIG}")
    print()

    for scenario_name in ["S50", "S65", "S80"]:
        scenario = scenario_data[scenario_name]
        base_rows = scenario["base_rows"]
        base_users = {u for u, _, _, _ in base_rows}
        base_items = {i for _, i, _, _ in base_rows}

        print(
            f"{scenario_name}: per-user base={scenario['base_ratio']:.0%} | "
            f"base={len(base_rows):,} "
            f"({len(base_users):,} users, {len(base_items):,} items) | "
            f"stream_candidate={scenario['n_stream_candidate']:,} | "
            f"stream_known={scenario['n_stream_known']:,} "
            f"({scenario['stream_known_fraction']:.2%}) | "
            f"stream_users={len(scenario['eligible_users']):,}"
        )

        for index, (candidate, known) in enumerate(
            zip(scenario["candidate_chunks"], scenario["known_chunks"]),
            start=1,
        ):
            retention = len(known) / len(candidate) if candidate else 0.0

            print(
                f"    chunk_{index}: candidate={len(candidate):,} | "
                f"known={len(known):,} ({retention:.2%}) | "
                f"users={len({u for u, _, _, _ in known}):,} | "
                f"items={len({i for _, i, _, _ in known}):,}"
            )

    print()
    print("Prequential sequence:")
    print("  update chunk1 -> evaluate chunk2")
    print("  update chunk2 -> evaluate chunk3")
    print("  update chunk3 -> evaluate chunk4")
    print()
    print("Branches:")
    print("  IBPR_STALE")
    print("  OnlineIBPR ORIGINAL via fit(update_chunk_train_set)")
    print("  OnlineIBPRMejorado O014 via partial_fit_recent(...)")
    print()
    print(
        "Expected paired points          : "
        f"{len(STREAM_SCENARIOS)} scenarios x "
        f"{len(DIAGNOSTIC_SEEDS)} seeds x "
        f"{N_STREAM_CHUNKS - 1} eval points = "
        f"{len(STREAM_SCENARIOS) * len(DIAGNOSTIC_SEEDS) * (N_STREAM_CHUNKS - 1)}"
    )
    print()
    print("Timing interpretation           : DESCRIPTIVE ONLY; computation budgets are not matched")
    print(
        "Interpretation: diagnostic development comparison only. "
        "No retuning of R900 or O014 is allowed."
    )
    print()


# ============================================================
# Models
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
        name=f"IBPR_diag_{scenario['name']}_seed{seed}",
    )

    start = time.perf_counter()
    model.fit(base_train_set)
    elapsed = time.perf_counter() - start

    return model, base_train_set, elapsed


def instantiate_original_from_base(base_model):
    return OnlineIBPROriginal(
        k=ORIGINAL_DIAGNOSTIC_CONFIG["k"],
        max_iter=ORIGINAL_DIAGNOSTIC_CONFIG["max_iter"],
        learning_rate=ORIGINAL_DIAGNOSTIC_CONFIG["learning_rate"],
        lamda=ORIGINAL_DIAGNOSTIC_CONFIG["lamda"],
        batch_size=ORIGINAL_DIAGNOSTIC_CONFIG["batch_size"],
        init_params={
            "U": np.asarray(base_model.U).copy(),
            "V": np.asarray(base_model.V).copy(),
        },
        trainable=True,
        verbose=False,
        name="OnlineIBPR_original_diagnostic",
    )


def instantiate_improved_from_base(base_model, base_train_set, seed):
    model = OnlineIBPRMejorado(
        k=FROZEN_IBPR_CONFIG["k"],
        max_iter=1,
        learning_rate=FROZEN_ONLINE_CONFIG["learning_rate"],
        lamda=FROZEN_ONLINE_CONFIG["lamda"],
        batch_size=FROZEN_ONLINE_CONFIG["batch_size"],
        init_params={
            "U": np.asarray(base_model.U).copy(),
            "V": np.asarray(base_model.V).copy(),
        },
        update_V=FROZEN_ONLINE_CONFIG["update_V"],
        neg_sampling=FROZEN_ONLINE_CONFIG["neg_sampling"],
        normalize=FROZEN_ONLINE_CONFIG["normalize"],
        loss_mode=FROZEN_ONLINE_CONFIG["loss_mode"],
        seed=int(seed),
        verbose=False,
        name="OnlineIBPRMejorado_O014_diagnostic",
    )

    # Metadata required by score/rank before the first wrapper partial update.
    model.U = np.asarray(base_model.U).copy()
    model.V = np.asarray(base_model.V).copy()
    model.num_users = base_train_set.num_users
    model.num_items = base_train_set.num_items
    model.train_set = base_train_set

    return model


# ============================================================
# Trial execution
# ============================================================

def trial_key(scenario_name, seed):
    return (str(scenario_name), int(seed))


def parse_trial_key(row):
    return (str(row["scenario"]), int(row["seed"]))


def steps_for_key(step_rows, key):
    scenario_name, seed = key

    return [
        row
        for row in step_rows
        if str(row.get("scenario")) == scenario_name
        and int(row.get("seed")) == seed
    ]


def replace_rows_for_key(rows, key, replacements):
    scenario_name, seed = key

    kept = [
        row
        for row in rows
        if not (
            str(row.get("scenario")) == scenario_name
            and int(row.get("seed")) == seed
        )
    ]

    return kept + replacements


def physical_trial(scenario, seed, protocol_hash):
    print("=" * 110)
    print(f"RUN {scenario['name']} | seed={seed}")
    print("=" * 110)

    base_model, base_train_set, base_train_time = train_frozen_base(
        scenario,
        seed,
    )

    print(
        f"BASE trained: rows={len(scenario['base_rows']):,} | "
        f"users={base_train_set.num_users:,} | "
        f"items={base_train_set.num_items:,} | "
        f"time={base_train_time:.2f}s"
    )

    original_model = instantiate_original_from_base(base_model)
    improved_model = instantiate_improved_from_base(
        base_model,
        base_train_set,
        seed,
    )

    base_u = np.asarray(base_model.U).copy()
    base_v = np.asarray(base_model.V).copy()

    if not np.array_equal(np.asarray(original_model.U), base_u):
        raise AssertionError("Original no parte exactamente de U_base.")
    if not np.array_equal(np.asarray(original_model.V), base_v):
        raise AssertionError("Original no parte exactamente de V_base.")
    if not np.array_equal(np.asarray(improved_model.U), base_u):
        raise AssertionError("Improved no parte exactamente de U_base.")
    if not np.array_equal(np.asarray(improved_model.V), base_v):
        raise AssertionError("Improved no parte exactamente de V_base.")

    # Defensive isolation between branches.
    if np.shares_memory(original_model.U, improved_model.U):
        raise AssertionError("Original e Improved comparten memoria en U.")
    if np.shares_memory(original_model.V, improved_model.V):
        raise AssertionError("Original e Improved comparten memoria en V.")

    uid_map = base_train_set.uid_map
    iid_map = base_train_set.iid_map

    observed_rows = list(scenario["base_rows"])
    step_rows = []
    original_times = []
    improved_times = []

    for update_index in range(N_STREAM_CHUNKS - 1):
        update_chunk_number = update_index + 1
        eval_chunk_number = update_chunk_number + 1

        update_rows = list(scenario["known_chunks"][update_index])
        eval_rows = list(scenario["known_chunks"][update_index + 1])

        if not update_rows:
            raise AssertionError(
                f"{scenario['name']} seed={seed}: update chunk vacío."
            )
        if not eval_rows:
            raise AssertionError(
                f"{scenario['name']} seed={seed}: eval chunk vacío."
            )

        # The evaluation chunk must remain unseen. Check both exact rows and
        # user-item pairs; the latter is the relevant recommendation constraint.
        eval_row_ids = {(u, i, ts) for u, i, _, ts in eval_rows}
        observed_row_ids = {(u, i, ts) for u, i, _, ts in observed_rows}
        eval_pairs = {(u, i) for u, i, _, _ in eval_rows}
        observed_pairs = {(u, i) for u, i, _, _ in observed_rows}

        overlap_rows_before = eval_row_ids & observed_row_ids
        overlap_pairs_before = eval_pairs & observed_pairs

        if overlap_rows_before or overlap_pairs_before:
            raise AssertionError(
                "Fuga temporal antes del update: "
                f"rows={len(overlap_rows_before)}, "
                f"pairs={len(overlap_pairs_before)}."
            )

        post_update_rows = observed_rows + update_rows
        post_update_pairs = {(u, i) for u, i, _, _ in post_update_rows}
        overlap_pairs_after = eval_pairs & post_update_pairs

        if overlap_pairs_after:
            raise AssertionError(
                "Fuga temporal después de incorporar el update chunk: "
                f"pairs={len(overlap_pairs_after)}."
            )

        # Same accumulated history for filtering already-observed candidates.
        history_train_set = build_dataset(
            post_update_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=False,
        )

        eval_test_set = build_dataset(
            eval_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=True,
        )

        # Original receives ONLY the just-arrived chunk, but with frozen maps
        # so U/V dimensionality stays identical to the base.
        original_update_set = build_dataset(
            update_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=False,
        )

        recent_pairs = rows_to_pairs(update_rows, uid_map, iid_map)

        # Literal diagnosis of what the original implementation will use as j.
        original_matrix = original_update_set.matrix
        unique_values = np.unique(np.asarray(original_matrix.data))

        if not np.allclose(unique_values, np.asarray([1.0])):
            raise AssertionError(
                "El update set original no contiene exclusivamente feedback "
                f"implícito 1.0. Valores={unique_values.tolist()}"
            )

        original_j_diag = diagnose_original_j(
            original_update_set=original_update_set,
            recent_pairs=recent_pairs,
            history_csr=history_train_set.csr_matrix,
            iid_map=iid_map,
        )

        stale_metrics = evaluate_model(
            base_model,
            history_train_set,
            eval_test_set,
        )

        # ------------------------
        # Original update
        # ------------------------
        before_original_v = np.asarray(original_model.V).copy()

        start = time.perf_counter()
        original_model.fit(original_update_set)
        original_elapsed = time.perf_counter() - start

        original_times.append(original_elapsed)

        assert_same_model_universe(
            model=original_model,
            base_train_set=base_train_set,
            base_u=base_u,
            base_v=base_v,
            uid_map=uid_map,
            iid_map=iid_map,
            label="OnlineIBPR original",
        )

        original_v_exact_step = bool(
            np.array_equal(original_model.V, before_original_v)
        )

        if not original_v_exact_step:
            raise AssertionError(
                "OnlineIBPR original modificó V aunque su optimizer histórico "
                "sólo actualiza U. Revisar versión del baseline."
            )

        # ------------------------
        # Improved update
        # ------------------------
        before_improved_v = np.asarray(improved_model.V).copy()

        start = time.perf_counter()
        improved_model.partial_fit_recent(
            recent_pairs=recent_pairs,
            history_csr=history_train_set.csr_matrix.copy(),
            max_steps=FROZEN_ONLINE_CONFIG["max_steps"],
            n_epochs=FROZEN_ONLINE_CONFIG["n_epochs"],
        )
        improved_elapsed = time.perf_counter() - start

        improved_times.append(improved_elapsed)

        # partial_fit_recent updates num_users/num_items, while train_set remains
        # the frozen base dataset. Maps must therefore remain the base maps.
        assert_same_model_universe(
            model=improved_model,
            base_train_set=base_train_set,
            base_u=base_u,
            base_v=base_v,
            uid_map=uid_map,
            iid_map=iid_map,
            label="OnlineIBPRMejorado",
        )

        improved_v_exact_step = bool(
            np.array_equal(improved_model.V, before_improved_v)
        )

        if not improved_v_exact_step:
            raise AssertionError(
                "OnlineIBPRMejorado modificó V con update_V=False."
            )

        if not np.array_equal(improved_model.V, base_v):
            raise AssertionError(
                "OnlineIBPRMejorado dejó de ser bit-a-bit igual a V_base."
            )

        # Original should also keep V fixed in this audited implementation.
        if not np.array_equal(original_model.V, base_v):
            raise AssertionError(
                "OnlineIBPR original dejó de ser bit-a-bit igual a V_base."
            )

        if not np.array_equal(np.asarray(base_model.U), base_u):
            raise AssertionError(
                "La rama IBPR_STALE fue modificada indirectamente en U."
            )

        if not np.array_equal(np.asarray(base_model.V), base_v):
            raise AssertionError(
                "La rama IBPR_STALE fue modificada indirectamente en V."
            )

        original_metrics = evaluate_model(
            original_model,
            history_train_set,
            eval_test_set,
        )

        improved_metrics = evaluate_model(
            improved_model,
            history_train_set,
            eval_test_set,
        )

        original_v_diff = (
            float(np.max(np.abs(np.asarray(original_model.V) - base_v)))
            if base_v.size
            else 0.0
        )

        improved_v_diff = (
            float(np.max(np.abs(np.asarray(improved_model.V) - base_v)))
            if base_v.size
            else 0.0
        )

        row = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash,
            "seed": seed,
            "scenario": scenario["name"],
            "base_ratio": scenario["base_ratio"],
            "update_chunk": update_chunk_number,
            "eval_chunk": eval_chunk_number,
            "n_update_rows": len(update_rows),
            "n_eval_rows": len(eval_rows),
            "n_eval_users": len({u for u, _, _, _ in eval_rows}),
            "n_eval_items": len({i for _, i, _, _ in eval_rows}),
            "original_update_time_s_descriptive": original_elapsed,
            "improved_update_time_s_descriptive": improved_elapsed,
            "original_j_unique_indices": ",".join(map(str, original_j_diag["unique_indices"])),
            "original_j_raw_items": ",".join(original_j_diag["raw_items"]),
            "original_j_equals_positive_fraction": original_j_diag["equals_positive_fraction"],
            "original_j_known_positive_fraction": original_j_diag["known_positive_fraction"],
            "original_v_exact_equal_base": bool(
                np.array_equal(original_model.V, base_v)
            ),
            "original_v_max_abs_diff_base": original_v_diff,
            "improved_v_exact_equal_base": bool(
                np.array_equal(improved_model.V, base_v)
            ),
            "improved_v_max_abs_diff_base": improved_v_diff,
            "original_mean_u_norm_all": mean_user_norm(original_model.U),
            "improved_mean_u_norm_all": mean_user_norm(improved_model.U),
            "original_mean_u_norm_affected": mean_affected_user_norm(
                original_model.U,
                recent_pairs,
            ),
            "improved_mean_u_norm_affected": mean_affected_user_norm(
                improved_model.U,
                recent_pairs,
            ),
        }

        for metric in QUALITY_METRICS:
            row[f"stale_{metric}"] = stale_metrics[metric]
            row[f"original_{metric}"] = original_metrics[metric]
            row[f"improved_{metric}"] = improved_metrics[metric]
            row[f"original_minus_stale_{metric}"] = (
                original_metrics[metric] - stale_metrics[metric]
            )
            row[f"improved_minus_stale_{metric}"] = (
                improved_metrics[metric] - stale_metrics[metric]
            )
            row[f"improved_minus_original_{metric}"] = (
                improved_metrics[metric] - original_metrics[metric]
            )

        step_rows.append(row)
        observed_rows = post_update_rows

        print(
            f"  update {update_chunk_number} -> eval {eval_chunk_number} | "
            f"NDCG stale={stale_metrics[f'NDCG@{TOP_K}']:.6f} | "
            f"original={original_metrics[f'NDCG@{TOP_K}']:.6f} | "
            f"improved={improved_metrics[f'NDCG@{TOP_K}']:.6f} | "
            f"Δ imp-orig="
            f"{row[f'improved_minus_original_NDCG@{TOP_K}']:+.6f} | "
            f"j_known_positive={original_j_diag['known_positive_fraction']:.2%} | "
            f"time(desc) orig={original_elapsed:.4f}s | "
            f"imp={improved_elapsed:.4f}s"
        )

    if len(step_rows) != N_STREAM_CHUNKS - 1:
        raise AssertionError("Trial incompleto: cantidad inesperada de eval points.")

    trial = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "seed": seed,
        "scenario": scenario["name"],
        "base_ratio": scenario["base_ratio"],
        "n_base": len(scenario["base_rows"]),
        "n_base_users": len({u for u, _, _, _ in scenario["base_rows"]}),
        "n_base_items": len({i for _, i, _, _ in scenario["base_rows"]}),
        "n_stream_candidate": scenario["n_stream_candidate"],
        "n_stream_known": scenario["n_stream_known"],
        "stream_known_fraction": scenario["stream_known_fraction"],
        "n_eval_points": len(step_rows),
        "base_train_time_s": base_train_time,
        "total_original_update_time_s_descriptive": float(np.sum(original_times)),
        "total_improved_update_time_s_descriptive": float(np.sum(improved_times)),
        "descriptive_original_over_improved_time_ratio": (
            float(np.sum(original_times) / np.sum(improved_times))
            if np.sum(improved_times) > 0
            else np.nan
        ),
        "mean_original_j_equals_positive_fraction": float(
            np.mean(
                [
                    row["original_j_equals_positive_fraction"]
                    for row in step_rows
                ]
            )
        ),
        "mean_original_j_known_positive_fraction": float(
            np.mean(
                [
                    row["original_j_known_positive_fraction"]
                    for row in step_rows
                ]
            )
        ),
        "all_original_v_exact_equal_base": all(
            bool(row["original_v_exact_equal_base"])
            for row in step_rows
        ),
        "max_original_v_abs_diff_base": max(
            float(row["original_v_max_abs_diff_base"])
            for row in step_rows
        ),
        "all_improved_v_exact_equal_base": all(
            bool(row["improved_v_exact_equal_base"])
            for row in step_rows
        ),
        "max_improved_v_abs_diff_base": max(
            float(row["improved_v_max_abs_diff_base"])
            for row in step_rows
        ),
    }

    for metric in QUALITY_METRICS:
        trial[f"mean_stale_{metric}"] = float(
            np.mean([row[f"stale_{metric}"] for row in step_rows])
        )
        trial[f"mean_original_{metric}"] = float(
            np.mean([row[f"original_{metric}"] for row in step_rows])
        )
        trial[f"mean_improved_{metric}"] = float(
            np.mean([row[f"improved_{metric}"] for row in step_rows])
        )
        trial[f"mean_original_minus_stale_{metric}"] = float(
            np.mean(
                [row[f"original_minus_stale_{metric}"] for row in step_rows]
            )
        )
        trial[f"mean_improved_minus_stale_{metric}"] = float(
            np.mean(
                [row[f"improved_minus_stale_{metric}"] for row in step_rows]
            )
        )
        trial[f"mean_improved_minus_original_{metric}"] = float(
            np.mean(
                [row[f"improved_minus_original_{metric}"] for row in step_rows]
            )
        )

    print(
        f"DONE {scenario['name']} seed={seed}: "
        f"mean ΔNDCG improved-original="
        f"{trial[f'mean_improved_minus_original_NDCG@{TOP_K}']:+.6f}"
    )
    print()

    return trial, step_rows


# ============================================================
# Aggregation
# ============================================================

def build_summary(trial_rows, step_rows, protocol_hash):
    expected_trials = len(STREAM_SCENARIOS) * len(DIAGNOSTIC_SEEDS)
    expected_steps = expected_trials * (N_STREAM_CHUNKS - 1)

    if len(trial_rows) != expected_trials:
        raise ValueError(
            f"Trials incompletos: esperados={expected_trials}, "
            f"recibidos={len(trial_rows)}."
        )

    if len(step_rows) != expected_steps:
        raise ValueError(
            f"Steps incompletos: esperados={expected_steps}, "
            f"recibidos={len(step_rows)}."
        )

    for row in trial_rows + step_rows:
        if str(row.get("protocol_version")) != PROTOCOL_VERSION:
            raise ValueError("Fila con protocol_version incompatible en agregación.")
        if str(row.get("protocol_hash")) != protocol_hash:
            raise ValueError("Fila con protocol_hash incompatible en agregación.")

    trial_keys = [parse_trial_key(row) for row in trial_rows]

    if len(set(trial_keys)) != len(trial_keys):
        raise ValueError("Trials duplicados detectados.")

    step_keys = [
        (
            str(row["scenario"]),
            int(row["seed"]),
            int(row["eval_chunk"]),
        )
        for row in step_rows
    ]

    if len(set(step_keys)) != len(step_keys):
        raise ValueError("Steps duplicados detectados.")

    for scenario in [s["name"] for s in STREAM_SCENARIOS]:
        for seed in DIAGNOSTIC_SEEDS:
            key_rows = [
                row
                for row in step_rows
                if str(row["scenario"]) == scenario
                and int(row["seed"]) == int(seed)
            ]
            eval_chunks = {int(row["eval_chunk"]) for row in key_rows}

            if len(key_rows) != N_STREAM_CHUNKS - 1 or eval_chunks != {2, 3, 4}:
                raise ValueError(
                    f"Trial {scenario}/seed={seed} incompleto o corrupto: "
                    f"n_steps={len(key_rows)}, eval_chunks={sorted(eval_chunks)}."
                )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "n_trials": len(trial_rows),
        "n_paired_points": len(step_rows),
        "scenarios": ",".join(s["name"] for s in STREAM_SCENARIOS),
        "seeds": ",".join(map(str, DIAGNOSTIC_SEEDS)),
    }

    for field in [
        "total_original_update_time_s_descriptive",
        "total_improved_update_time_s_descriptive",
        "descriptive_original_over_improved_time_ratio",
    ]:
        mean_value, std_value = mean_std(
            float(row[field]) for row in trial_rows
        )

        summary[f"mean_{field}"] = mean_value

        if field == "total_original_update_time_s_descriptive":
            summary["std_total_original_update_time_s_descriptive"] = std_value
        elif field == "total_improved_update_time_s_descriptive":
            summary["std_total_improved_update_time_s_descriptive"] = std_value

    for field in [
        "original_j_equals_positive_fraction",
        "original_j_known_positive_fraction",
    ]:
        mean_value, std_value = mean_std(
            float(row[field]) for row in step_rows
        )
        summary[f"mean_{field}"] = mean_value
        summary[f"std_{field}"] = std_value

    summary["all_original_v_exact_equal_base"] = all(
        str(row["all_original_v_exact_equal_base"]).lower() == "true"
        if isinstance(row["all_original_v_exact_equal_base"], str)
        else bool(row["all_original_v_exact_equal_base"])
        for row in trial_rows
    )

    summary["max_original_v_abs_diff_base"] = max(
        float(row["max_original_v_abs_diff_base"])
        for row in trial_rows
    )

    summary["all_improved_v_exact_equal_base"] = all(
        str(row["all_improved_v_exact_equal_base"]).lower() == "true"
        if isinstance(row["all_improved_v_exact_equal_base"], str)
        else bool(row["all_improved_v_exact_equal_base"])
        for row in trial_rows
    )

    summary["max_improved_v_abs_diff_base"] = max(
        float(row["max_improved_v_abs_diff_base"])
        for row in trial_rows
    )

    for metric in QUALITY_METRICS:
        for prefix in [
            "stale",
            "original",
            "improved",
            "original_minus_stale",
            "improved_minus_stale",
            "improved_minus_original",
        ]:
            field = f"{prefix}_{metric}"
            values = [float(row[field]) for row in step_rows]
            mean_value, std_value = mean_std(values)

            summary[f"mean_{field}"] = mean_value
            summary[f"std_{field}"] = std_value

        for prefix in [
            "original_minus_stale",
            "improved_minus_stale",
            "improved_minus_original",
        ]:
            field = f"{prefix}_{metric}"
            summary[f"positive_{field}"] = sum(
                float(row[field]) > 0.0 for row in step_rows
            )

    return summary


def print_summary(summary):
    ndcg = f"NDCG@{TOP_K}"

    print("=" * 110)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 110)
    print(f"Trials                         : {summary['n_trials']}")
    print(f"Paired evaluation points       : {summary['n_paired_points']}")
    print()
    print(
        f"Stale mean {ndcg:<12}          : "
        f"{float(summary[f'mean_stale_{ndcg}']):.6f}"
    )
    print(
        f"Original mean {ndcg:<9}        : "
        f"{float(summary[f'mean_original_{ndcg}']):.6f}"
    )
    print(
        f"Improved mean {ndcg:<9}        : "
        f"{float(summary[f'mean_improved_{ndcg}']):.6f}"
    )
    print()
    print(
        f"Original - Stale               : "
        f"{float(summary[f'mean_original_minus_stale_{ndcg}']):+.6f} "
        f"± {float(summary[f'std_original_minus_stale_{ndcg}']):.6f} | "
        f"positive="
        f"{int(summary[f'positive_original_minus_stale_{ndcg}'])}/"
        f"{int(summary['n_paired_points'])}"
    )
    print(
        f"Improved - Stale               : "
        f"{float(summary[f'mean_improved_minus_stale_{ndcg}']):+.6f} "
        f"± {float(summary[f'std_improved_minus_stale_{ndcg}']):.6f} | "
        f"positive="
        f"{int(summary[f'positive_improved_minus_stale_{ndcg}'])}/"
        f"{int(summary['n_paired_points'])}"
    )
    print(
        f"Improved - Original            : "
        f"{float(summary[f'mean_improved_minus_original_{ndcg}']):+.6f} "
        f"± {float(summary[f'std_improved_minus_original_{ndcg}']):.6f} | "
        f"positive="
        f"{int(summary[f'positive_improved_minus_original_{ndcg}'])}/"
        f"{int(summary['n_paired_points'])}"
    )
    print()
    print(
        "Mean total original update time (descriptive): "
        f"{float(summary['mean_total_original_update_time_s_descriptive']):.4f}s"
    )
    print(
        "Mean total improved update time (descriptive): "
        f"{float(summary['mean_total_improved_update_time_s_descriptive']):.4f}s"
    )
    print(
        "Original/improved time ratio (descriptive): "
        f"{float(summary['mean_descriptive_original_over_improved_time_ratio']):.4f}x"
    )
    print()
    print(
        "Original j==positive fraction   : "
        f"{float(summary['mean_original_j_equals_positive_fraction']):.2%} "
        f"± {float(summary['std_original_j_equals_positive_fraction']):.2%}"
    )
    print(
        "Original j known-positive frac  : "
        f"{float(summary['mean_original_j_known_positive_fraction']):.2%} "
        f"± {float(summary['std_original_j_known_positive_fraction']):.2%}"
    )
    print("Timing guard                    : DESCRIPTIVE ONLY; optimizer-step budgets are not matched")
    print()
    print(
        "Original V exact                : "
        f"{summary['all_original_v_exact_equal_base']} | "
        f"max diff={float(summary['max_original_v_abs_diff_base']):.12g}"
    )
    print(
        "Improved V exact                : "
        f"{summary['all_improved_v_exact_equal_base']} | "
        f"max diff={float(summary['max_improved_v_abs_diff_base']):.12g}"
    )
    print()
    print(
        "Interpretation guard: development diagnostic only; "
        "do not retune R900/O014 from these results."
    )
    print()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    validate_implementations()
    protocol_hash = current_protocol_hash()

    all_rows = load_positive_chrono_movielens()
    hpo_pool = build_hpo_pool(all_rows)
    user_histories = build_user_histories(hpo_pool)

    scenario_data = {
        spec["name"]: build_stream_scenario(user_histories, spec)
        for spec in STREAM_SCENARIOS
    }

    print_plan(all_rows, hpo_pool, scenario_data, protocol_hash)

    if args.plan_only:
        print("PLAN-ONLY complete. No model was trained.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = (
        args.timestamp
        if args.timestamp
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    log_path = os.path.join(
        RESULTS_DIR,
        f"compare_online_ibpr_original_vs_mejorado_{timestamp}.txt",
    )
    steps_path = os.path.join(
        RESULTS_DIR,
        f"compare_online_ibpr_original_vs_mejorado_steps_{timestamp}.csv",
    )
    trials_path = os.path.join(
        RESULTS_DIR,
        f"compare_online_ibpr_original_vs_mejorado_trials_{timestamp}.csv",
    )
    summary_path = os.path.join(
        RESULTS_DIR,
        f"compare_online_ibpr_original_vs_mejorado_summary_{timestamp}.csv",
    )

    existing_trials = load_csv(trials_path)
    existing_steps = load_csv(steps_path)

    validate_resume_protocol(existing_trials, trials_path, protocol_hash)
    validate_resume_protocol(existing_steps, steps_path, protocol_hash)

    with open(log_path, "a", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)

        with redirect_stdout(tee):
            print()
            print(f"Timestamp: {timestamp}")
            print(f"Results directory: {RESULTS_DIR}")
            print()

            # Repeat plan inside the persisted log.
            print_plan(all_rows, hpo_pool, scenario_data, protocol_hash)

            trial_rows = list(existing_trials)
            step_rows = list(existing_steps)

            for scenario_name in ["S50", "S65", "S80"]:
                scenario = scenario_data[scenario_name]

                for seed in DIAGNOSTIC_SEEDS:
                    key = trial_key(scenario_name, seed)

                    matching_trials = [
                        row
                        for row in trial_rows
                        if parse_trial_key(row) == key
                    ]
                    matching_steps = steps_for_key(step_rows, key)

                    matching_eval_chunks = {
                        int(row["eval_chunk"])
                        for row in matching_steps
                    }

                    complete = (
                        len(matching_trials) == 1
                        and int(matching_trials[0]["n_eval_points"])
                        == N_STREAM_CHUNKS - 1
                        and len(matching_steps) == N_STREAM_CHUNKS - 1
                        and matching_eval_chunks == {2, 3, 4}
                    )

                    if complete:
                        print(
                            f"REUSE {scenario_name} | seed={seed} | "
                            f"steps={len(matching_steps)}"
                        )
                        continue

                    if len(matching_trials) > 1:
                        raise ValueError(f"Trial duplicado para {key}.")

                    if matching_trials or matching_steps:
                        print(
                            f"RESET incomplete {scenario_name} | seed={seed} | "
                            f"trial_rows={len(matching_trials)} | "
                            f"step_rows={len(matching_steps)}"
                        )

                    trial_rows = replace_rows_for_key(trial_rows, key, [])
                    step_rows = replace_rows_for_key(step_rows, key, [])

                    trial, completed_steps = physical_trial(
                        scenario,
                        seed,
                        protocol_hash,
                    )

                    # Persist only after the full scenario/seed trial succeeds.
                    trial_rows = replace_rows_for_key(
                        trial_rows,
                        key,
                        [trial],
                    )
                    step_rows = replace_rows_for_key(
                        step_rows,
                        key,
                        completed_steps,
                    )

                    save_csv(trials_path, TRIAL_FIELDS, trial_rows)
                    save_csv(steps_path, STEP_FIELDS, step_rows)

            summary = build_summary(trial_rows, step_rows, protocol_hash)
            save_csv(summary_path, SUMMARY_FIELDS, [summary])

            print_summary(summary)

            print("Files:")
            print(f"  log     : {log_path}")
            print(f"  steps   : {steps_path}")
            print(f"  trials  : {trials_path}")
            print(f"  summary : {summary_path}")


if __name__ == "__main__":
    main()
