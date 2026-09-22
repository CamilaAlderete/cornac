import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import cornac
import scipy
from cornac.data import Dataset
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.models.ibpr.ibpr import ibpr as ibpr_core
from cornac.models.online_ibpr_mejorado.online_ibpr_mejorado import (
    online_ibpr_mejorado as online_core,
)

# The audited final H1-H3 script must live in the same directory when this
# experiment is executed inside tests/online_ibpr_mejorado.
import final_h1_h3_online_ibpr_mejorado_v3_1 as final_ref


# ============================================================================
# Frozen protocol
# ============================================================================

PROTOCOL_VERSION = "ablation_a1_update_dynamics_v1_2_20260921"
TOP_K = 20
SEEDS = [777, 999]
BRANCHES = ["SEQUENTIAL_ORIGINAL", "RETROSPECTIVE_FIXED_HISTORY", "RESET_CUMULATIVE", "RESET_CURRENT"]

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

EXPECTED_DATA_SHA256 = "29da5346c5bcf37dc927771d8ffd7ec3323dc7857ed4b0f6a45278b666954d3e"
EXPECTED_TOTAL_POSITIVES = 836_478
EXPECTED_TARGET_BASE = 501_886
EXPECTED_EFFECTIVE_BASE = 501_887
EXPECTED_FUTURE_RAW = 334_591
EXPECTED_FUTURE_WARM = 52_467
EXPECTED_BASE_USERS = 4_037
EXPECTED_BASE_ITEMS = 3_505
EXPECTED_CHUNKS = [13_118, 13_118, 13_114, 13_117]
EXPECTED_PRIMARY_ROWS = [5_300, 8_546, 9_143]
EXPECTED_PRIMARY_USERS = [212, 315, 306]

EXPECTED_ENV = {
    "python": "3.12.0",
    "cornac": "2.3.5",
    "numpy": "2.4.2",
    "scipy": "1.17.1",
    "torch": "2.10.0+cpu",
}

# inspect.getsource fingerprints frozen after H1-H4 audits.
EXPECTED_SOURCE_SHA = {
    "ibpr_core": "10e07672f092ec4995ebadd81975c87a6e3a82cb89e8e25b8ae31bf386a5151c",
    "ibpr_wrapper": "e173018d3cd4de03a2b4e4abea3446f26081be3e96a1d494d3d75c107c9cee8e",
    "online_core": "867e31e364159f77ddde84e0fbfc46b5799be21467c5fda3de103b8dccd5891f",
    "online_wrapper": "4d298f4490324252f9dc5cd61e78bafad0e91e582e7c29f3312c67101bf215da",
}

EXPECTED_FINAL_H1H3_PROTOCOL_VERSION = "final_h1_h3_v3_1_hardened_20260914"
EXPECTED_FINAL_H1H3_PROTOCOL_HASH = "05e4f07096b02ae1730278076b35c13a031179ff170a1d6eb902d633173fdf77"
DEFAULT_REFERENCE_STEPS_NAME = "final_h1_h3_online_ibpr_mejorado_steps_20260914_175307.csv"
REFERENCE_REPRO_TOL = 1e-10
EXPECTED_REFERENCE_STEPS_SEMANTIC_SHA256 = "182a1ca5195b886e7e3c57d618c8650c66d07913503c8dba54a87c3dda51906c"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

STEP_FIELDS = [
    "protocol_version", "protocol_hash", "data_sha256", "seed", "eval_point",
    "update_chunk", "eval_chunk", "branch", "latest_call_seed",
    "n_recent_pairs_this_state", "n_recent_pairs_last_call", "n_partial_calls_to_state",
    "optimizer_steps_this_state", "optimizer_steps_last_call", "n_update_users_this_state",
    "interactions_per_update_user_mean", "interactions_per_update_user_median",
    "interactions_per_update_user_p95", "last_call_time_s", "state_construction_time_s",
    "call_seeds", "history_rows_by_call", "n_eval_rows", "n_eval_users",
    "n_current_eval_rows", "n_current_eval_users",
    "n_allwarm_eval_rows", "n_allwarm_eval_users",
    "u_fro_diff_all", "u_fro_diff_update_users", "u_cosine_mean_update_users",
    "u_cosine_median_update_users", "u_cosine_p05_update_users",
    "n_user_rows_bitwise_changed", "u_max_abs_diff_non_update_users", "v_exact_equal_base", "v_max_abs_diff_base",
    "uid_map_exact", "iid_map_exact", "stale_reference_reproduced",
    "sequential_reference_reproduced", "point1_all_branch_u_bitwise_equal",
]
for metric in QUALITY_METRICS:
    STEP_FIELDS.extend([
        f"{metric}", f"stale_{metric}", f"full_reference_{metric}",
        f"branch_minus_stale_{metric}", f"branch_minus_full_reference_{metric}",
        f"reference_sequential_{metric}", f"branch_minus_reference_sequential_{metric}",
        f"allwarm_{metric}", f"allwarm_stale_{metric}",
        f"allwarm_full_reference_{metric}", f"allwarm_branch_minus_stale_{metric}",
        f"current_{metric}", f"current_stale_{metric}",
        f"current_branch_minus_stale_{metric}",
    ])

TRIAL_FIELDS = [
    "protocol_version", "protocol_hash", "data_sha256", "seed", "branch",
    "n_eval_points", "mean_last_call_time_s", "mean_state_construction_time_s",
    "point3_state_construction_time_s", "all_v_exact_equal_base", "max_v_abs_diff_base", "max_u_fro_diff_all",
]
for metric in QUALITY_METRICS:
    TRIAL_FIELDS.extend([
        f"mean_{metric}", f"mean_stale_{metric}", f"mean_full_reference_{metric}",
        f"mean_branch_minus_stale_{metric}", f"mean_branch_minus_full_reference_{metric}",
    ])

SUMMARY_FIELDS = [
    "protocol_version", "protocol_hash", "data_sha256", "seeds", "branches",
    "n_step_rows", "all_v_exact_equal_base", "max_v_abs_diff_base",
    "all_sequential_reference_reproduced", "all_stale_reference_reproduced",
]
for branch in BRANCHES:
    safe = branch.lower()
    SUMMARY_FIELDS.extend([
        f"{safe}_mean_NDCG@20", f"{safe}_mean_minus_stale_NDCG@20",
        f"{safe}_mean_minus_full_reference_NDCG@20",
        f"{safe}_mean_state_construction_time_s", f"{safe}_point3_state_construction_time_s", f"{safe}_max_u_fro_diff_all",
    ])
for point in [1, 2, 3]:
    for branch in BRANCHES:
        safe = branch.lower()
        SUMMARY_FIELDS.extend([
            f"{safe}_mean_NDCG@20_point_{point}",
            f"{safe}_mean_minus_stale_NDCG@20_point_{point}",
        ])
SUMMARY_FIELDS.extend([
    "mean_sequential_minus_fixed_history_NDCG@20",
    "mean_fixed_history_minus_reset_cumulative_NDCG@20",
    "mean_sequential_minus_reset_cumulative_NDCG@20",
    "point3_sequential_minus_fixed_history_NDCG@20",
    "point3_fixed_history_minus_reset_cumulative_NDCG@20",
    "point3_sequential_minus_reset_cumulative_NDCG@20",
    "mean_current_sequential_minus_reset_current_NDCG@20",
    "point3_current_sequential_minus_reset_current_NDCG@20",
    "point1_all_branch_u_bitwise_equal",
])


# ============================================================================
# Utilities and guards
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Focused ablation A1: sequential update dynamics of frozen O014."
    )
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--reference-steps",
        default=os.path.join(RESULTS_DIR, DEFAULT_REFERENCE_STEPS_NAME),
        help="Final H1-H3 steps CSV used only as an audited reference; Full is not retrained.",
    )
    return parser.parse_args()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def source_sha(obj):
    return sha256_bytes(inspect.getsource(obj).encode("utf-8"))


def arrays_bitwise_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    )


def semantic_rows_sha256(rows):
    canonical = json.dumps(
        sorted(rows, key=lambda r: (int(r["seed"]), int(r["eval_point"]))),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256_bytes(canonical.encode("utf-8"))


def current_environment():
    return {
        "python": platform.python_version(),
        "cornac": getattr(cornac, "__version__", "unavailable"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }


def validate_exact_environment_and_sources():
    env = current_environment()
    mismatch = {k: (env[k], v) for k, v in EXPECTED_ENV.items() if env[k] != v}
    if mismatch:
        raise RuntimeError(f"Frozen environment mismatch: {mismatch}")

    actual_sources = {
        "ibpr_core": source_sha(ibpr_core),
        "ibpr_wrapper": source_sha(IBPR),
        "online_core": source_sha(online_core),
        "online_wrapper": source_sha(OnlineIBPRMejorado),
    }
    source_mismatch = {
        k: (actual_sources[k], expected)
        for k, expected in EXPECTED_SOURCE_SHA.items()
        if actual_sources[k] != expected
    }
    if source_mismatch:
        raise RuntimeError(f"Frozen model source mismatch: {source_mismatch}")

    ref_path = inspect.getsourcefile(final_ref)
    if not ref_path or not os.path.exists(ref_path):
        raise RuntimeError(f"Final H1-H3 reference script not found: {ref_path}")

    # This exact frozen protocol fingerprint already covers the final script raw SHA,
    # environment, Dataset.build/ranking_eval sources, and R900/O014 sources.
    local_final_protocol = final_ref.current_protocol_hash(EXPECTED_DATA_SHA256)
    if local_final_protocol != EXPECTED_FINAL_H1H3_PROTOCOL_HASH:
        raise RuntimeError(
            "Local final H1-H3 protocol fingerprint no longer matches the frozen final run: "
            f"actual={local_final_protocol}, expected={EXPECTED_FINAL_H1H3_PROTOCOL_HASH}"
        )

    # Reuse the validated functional smoke tests from the exact final script.
    final_ref.validate_implementation_contracts()

    print("Frozen environment/source contracts: OK")
    print(f"  Environment       : {env}")
    print(f"  Final H1-H3 script: {ref_path}")
    print(f"  Final script SHA  : {sha256_file(ref_path)}")
    print("  R900/O014 source fingerprints: OK")
    print()


def validate_data(data):
    checks = {
        "total_positives": len(data["all_rows"]) if "all_rows" in data else EXPECTED_TOTAL_POSITIVES,
        "target_base": data["target_base_rows"],
        "effective_base": len(data["base_rows"]),
        "future_raw": len(data["future_rows"]),
        "future_warm": len(data["warm_future_rows"]),
        "base_users": len({r[0] for r in data["base_rows"]}),
        "base_items": len({r[1] for r in data["base_rows"]}),
        "chunks": [len(c) for c in data["known_chunks"]],
        "primary_rows": [x["n_primary_rows"] for x in data["primary_eval_plan"]],
        "primary_users": [x["n_primary_users"] for x in data["primary_eval_plan"]],
    }
    expected = {
        "total_positives": EXPECTED_TOTAL_POSITIVES,
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
    mismatch = {k: (checks[k], expected[k]) for k in expected if checks[k] != expected[k]}
    if mismatch:
        raise RuntimeError(f"Frozen H1-H3 data guards mismatch: {mismatch}")

    data_hash = final_ref.dataset_sha256(data["all_rows"])
    if data_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"Data SHA mismatch: {data_hash} != {EXPECTED_DATA_SHA256}")
    return data_hash


def load_reference_steps(path):
    if not os.path.exists(path):
        raise RuntimeError(
            f"Reference H1-H3 steps CSV not found: {path}. "
            "Pass --reference-steps with the audited final file."
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 6:
        raise RuntimeError(f"Expected 6 final H1-H3 step rows, found {len(rows)}")
    semantic_sha = semantic_rows_sha256(rows)
    if semantic_sha != EXPECTED_REFERENCE_STEPS_SEMANTIC_SHA256:
        raise RuntimeError(
            "Reference final H1-H3 steps semantic SHA mismatch: "
            f"actual={semantic_sha}, expected={EXPECTED_REFERENCE_STEPS_SEMANTIC_SHA256}"
        )
    if {r["protocol_version"] for r in rows} != {EXPECTED_FINAL_H1H3_PROTOCOL_VERSION}:
        raise RuntimeError("Reference final H1-H3 protocol_version mismatch")
    if {r["protocol_hash"] for r in rows} != {EXPECTED_FINAL_H1H3_PROTOCOL_HASH}:
        raise RuntimeError("Reference final H1-H3 protocol_hash mismatch")
    if {r["data_sha256"] for r in rows} != {EXPECTED_DATA_SHA256}:
        raise RuntimeError("Reference final H1-H3 data SHA mismatch")

    index = {}
    for r in rows:
        key = (int(r["seed"]), int(r["eval_point"]))
        if key in index:
            raise RuntimeError(f"Duplicate reference key: {key}")
        index[key] = r
    expected_keys = {(s, p) for s in SEEDS for p in [1, 2, 3]}
    if set(index) != expected_keys:
        raise RuntimeError(f"Reference key mismatch: {sorted(index)}")
    return index


def optimizer_steps(n_pairs):
    batch = FROZEN_ONLINE_CONFIG["batch_size"]
    epochs = FROZEN_ONLINE_CONFIG["n_epochs"]
    return math.ceil(n_pairs / batch) * epochs if n_pairs else 0


def per_user_stats(recent_pairs):
    if len(recent_pairs) == 0:
        return 0, 0.0, 0.0, 0.0
    _, counts = np.unique(recent_pairs[:, 0], return_counts=True)
    return (
        int(len(counts)), float(np.mean(counts)), float(np.median(counts)),
        float(np.percentile(counts, 95)),
    )


def rowwise_cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.ones(len(a), dtype=np.float64)
    nz = denom > 0
    out[nz] = np.sum(a[nz] * b[nz], axis=1) / denom[nz]
    return np.clip(out, -1.0, 1.0)


def u_diagnostics(U, base_u, update_user_indices):
    U = np.asarray(U)
    base_u = np.asarray(base_u)
    diff = U - base_u
    U_bytes = np.ascontiguousarray(U).view(np.uint8).reshape(U.shape[0], -1)
    base_bytes = np.ascontiguousarray(base_u).view(np.uint8).reshape(base_u.shape[0], -1)
    changed = np.any(U_bytes != base_bytes, axis=1)
    all_fro = float(np.linalg.norm(diff))
    idx = np.asarray(sorted(set(int(x) for x in update_user_indices)), dtype=np.int64)
    mask = np.ones(U.shape[0], dtype=bool)
    if len(idx):
        mask[idx] = False
    non_update_max = float(np.max(np.abs(diff[mask]))) if np.any(mask) else 0.0
    if len(idx) == 0:
        return all_fro, 0.0, 1.0, 1.0, 1.0, int(np.sum(changed)), non_update_max
    sub_fro = float(np.linalg.norm(diff[idx]))
    cos = rowwise_cosine(U[idx], base_u[idx])
    return (
        all_fro, sub_fro, float(np.mean(cos)), float(np.median(cos)),
        float(np.percentile(cos, 5)), int(np.sum(changed)), non_update_max,
    )


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def save_csv_atomic(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)




def matched_population_sizes(data):
    out = []
    for step_idx in range(3):
        eval_rows, _, _ = final_ref.primary_eval_rows_for_step(data["known_chunks"], step_idx)
        current_rows = list(data["known_chunks"][step_idx])
        current_users = {r[0] for r in current_rows}
        matched = [r for r in eval_rows if r[0] in current_users]
        out.append({
            "eval_point": step_idx + 1,
            "rows": len(matched),
            "users": len({r[0] for r in matched}),
        })
    return out

def protocol_payload(data, data_hash, reference_steps_path):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Which operational aspects of repeated frozen-O014 partial updates are associated "
            "with the point-3 relative drop versus Stale?"
        ),
        "data_sha256": data_hash,
        "seeds": SEEDS,
        "branches": {
            "SEQUENTIAL_ORIGINAL": (
                "original prequential policy: persistent model; Ci uses history Base+C1..Ci"
            ),
            "RETROSPECTIVE_FIXED_HISTORY": (
                "diagnostic retrospective control rebuilt from base at each point; C1..Ci are "
                "separate calls and every call uses the point-final observed history Base+C1..Ci"
            ),
            "RESET_CUMULATIVE": (
                "diagnostic replay rebuilt from base at each point; one call on C1..Ci using "
                "point-final observed history"
            ),
            "RESET_CURRENT": (
                "diagnostic reset rebuilt from base at each point; one call only on Ci using "
                "point-final observed history"
            ),
        },
        "seed_policy": {
            "SEQUENTIAL_ORIGINAL": "wrapper starts at seed; calls consume seed,seed+1,...",
            "RETROSPECTIVE_FIXED_HISTORY": (
                "fresh wrapper starts at seed for each point; replayed calls consume seed..seed+point-1"
            ),
            "RESET_CUMULATIVE": "single point call uses seed+point-1",
            "RESET_CURRENT": "single point call uses seed+point-1",
        },
        "causal_guard": (
            "No contrast identifies a unique causal mechanism. S-vs-H changes negative-sampling "
            "history for earlier calls; H-vs-C changes the operational packaging of the same "
            "positives/history into multiple calls versus one call, including optimizer resets, "
            "intermediate normalization, shuffling/batching and seed progression."
        ),
        "retrospective_guard": (
            "RETROSPECTIVE_FIXED_HISTORY is not a deployable online policy. At point i it uses only "
            "information already observed before eval chunk i+1, so it does not leak evaluation data."
        ),
        "matched_current_population": matched_population_sizes(data),
        "point1_guard": (
            "At point 1 all four branches are the same single C1 update with H1 and the same seed; "
            "their U arrays must be bitwise identical."
        ),
        "ibpr_config": FROZEN_IBPR_CONFIG,
        "online_config": FROZEN_ONLINE_CONFIG,
        "primary_eval": "exact H1-H3 PRIMARY populations",
        "allwarm_eval": "same supplementary all-warm populations",
        "reference_h1h3_protocol_hash": EXPECTED_FINAL_H1H3_PROTOCOL_HASH,
        "reference_steps_sha256": sha256_file(reference_steps_path),
        "reference_steps_semantic_sha256": EXPECTED_REFERENCE_STEPS_SEMANTIC_SHA256,
        "reference_repro_tolerance": REFERENCE_REPRO_TOL,
        "final_ref_script_sha256": sha256_file(inspect.getsourcefile(final_ref)),
        "environment": current_environment(),
        "source_sha": {k: source_sha(v) for k, v in {
            "ibpr_core": ibpr_core,
            "ibpr_wrapper": IBPR,
            "online_core": online_core,
            "online_wrapper": OnlineIBPRMejorado,
        }.items()},
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "interpretation_guard": (
            "diagnostic ablation only; no retuning, no new winner, no direct temporal-degradation "
            "claim across changing PRIMARY populations; report paired within-point contrasts"
        ),
    }

def protocol_hash(data, data_hash, reference_steps_path):
    payload = json.dumps(
        protocol_payload(data, data_hash, reference_steps_path), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    )
    return sha256_bytes(payload.encode("utf-8"))


def branch_model_from_base(base_train_set, base_u, base_v, point_seed):
    # Reuse the exact audited H1-H3 public initialization path.
    model = final_ref.initialize_online(
        base_train_set,
        base_u,
        base_v,
        point_seed,
    )
    model.name = "A1_reset_branch"
    if int(getattr(model, "_partial_update_count", -1)) != 0:
        raise RuntimeError("Reset branch did not start with _partial_update_count=0")
    if not arrays_bitwise_equal(model.U, base_u):
        raise RuntimeError("Reset branch initialization modified U_base")
    if not arrays_bitwise_equal(model.V, base_v):
        raise RuntimeError("Reset branch initialization modified V_base")
    return model

def assert_branch_invariants(model, base_v, uid_map, iid_map, label):
    exact = arrays_bitwise_equal(np.asarray(model.V), base_v)
    diff = max_abs_diff(model.V, base_v)
    maps_ok = model.uid_map == uid_map and model.iid_map == iid_map
    if not exact or diff != 0.0:
        raise RuntimeError(f"{label}: V changed; exact={exact}, max_diff={diff}")
    if not maps_ok:
        raise RuntimeError(f"{label}: uid_map/iid_map changed")
    return exact, diff


def reference_metric(ref_row, prefix, metric):
    return float(ref_row[f"{prefix}_{metric}"])


def compare_to_reference(actual, ref_row, prefix, label):
    diffs = {}
    for metric in QUALITY_METRICS:
        ref = reference_metric(ref_row, prefix, metric)
        d = abs(float(actual[metric]) - ref)
        diffs[metric] = d
        if d > REFERENCE_REPRO_TOL:
            raise RuntimeError(
                f"{label}: reference reproduction failed for {metric}: "
                f"actual={actual[metric]:.15g}, ref={ref:.15g}, abs_diff={d:.3g}"
            )
    return True


# ============================================================================
# Execution
# ============================================================================

def run_seed(seed, data, refs, p_hash):
    print("=" * 118)
    print(f"ABLATION A1 | seed={seed}")
    print("=" * 118)

    base_model, base_train_set, base_train_time = final_ref.train_base_model(data["base_rows"], seed)
    uid_map = dict(base_train_set.uid_map)
    iid_map = dict(base_train_set.iid_map)
    n_users = base_train_set.num_users
    n_items = base_train_set.num_items
    base_u = np.asarray(base_model.U).copy()
    base_v = np.asarray(base_model.V).copy()

    stale_model = final_ref.initialize_stale(base_train_set, base_u, base_v, seed)
    sequential_model = final_ref.initialize_online(base_train_set, base_u, base_v, seed)

    print(
        f"BASE: rows={len(data['base_rows']):,} | users={n_users:,} | items={n_items:,} | "
        f"train={base_train_time:.2f}s"
    )

    observed_rows = list(data["base_rows"])
    cumulative_recent_rows = []
    sequential_call_times = []
    sequential_history_rows = []
    rows_out = []

    for step_idx in range(3):
        eval_point = step_idx + 1
        update_chunk = step_idx + 1
        eval_chunk = step_idx + 2
        current_rows = list(data["known_chunks"][step_idx])
        cumulative_recent_rows.extend(current_rows)

        eval_rows, allwarm_eval_rows, _ = final_ref.primary_eval_rows_for_step(
            data["known_chunks"], step_idx
        )
        post_update_rows = observed_rows + current_rows

        eval_train_set = final_ref.build_dataset(
            post_update_rows, uid_map=uid_map, iid_map=iid_map,
            seed=seed, exclude_unknowns=False,
        )
        test_set = final_ref.build_dataset(
            eval_rows, uid_map=uid_map, iid_map=iid_map,
            seed=seed, exclude_unknowns=True,
        )
        allwarm_test_set = final_ref.build_dataset(
            allwarm_eval_rows, uid_map=uid_map, iid_map=iid_map,
            seed=seed, exclude_unknowns=True,
        )
        current_update_users_raw = {r[0] for r in current_rows}
        current_eval_rows = [r for r in eval_rows if r[0] in current_update_users_raw]
        if not current_eval_rows:
            raise RuntimeError(f"point={eval_point}: current-user matched eval population is empty")
        current_test_set = final_ref.build_dataset(
            current_eval_rows, uid_map=uid_map, iid_map=iid_map,
            seed=seed, exclude_unknowns=True,
        )

        final_ref.assert_dataset_contract(eval_train_set, uid_map, iid_map, n_users, n_items, "A1 eval_train")
        final_ref.assert_dataset_contract(test_set, uid_map, iid_map, n_users, n_items, "A1 test")
        final_ref.assert_dataset_contract(allwarm_test_set, uid_map, iid_map, n_users, n_items, "A1 allwarm")
        final_ref.assert_dataset_contract(current_test_set, uid_map, iid_map, n_users, n_items, "A1 current-matched")
        final_ref.assert_dataset_row_count(eval_train_set, len(post_update_rows), "A1 eval_train")
        final_ref.assert_dataset_row_count(test_set, len(eval_rows), "A1 test")
        final_ref.assert_dataset_row_count(allwarm_test_set, len(allwarm_eval_rows), "A1 allwarm")
        final_ref.assert_dataset_row_count(current_test_set, len(current_eval_rows), "A1 current-matched")

        stale_metrics = final_ref.evaluate_model(stale_model, eval_train_set, test_set)
        stale_allwarm = final_ref.evaluate_model(stale_model, eval_train_set, allwarm_test_set)
        stale_current = final_ref.evaluate_model(stale_model, eval_train_set, current_test_set)
        ref_row = refs[(seed, eval_point)]
        stale_primary_repro = compare_to_reference(
            stale_metrics, ref_row, "stale", f"seed={seed} point={eval_point} stale"
        )
        stale_allwarm_repro = compare_to_reference(
            stale_allwarm, ref_row, "allwarm_stale", f"seed={seed} point={eval_point} stale allwarm"
        )
        stale_repro = stale_primary_repro and stale_allwarm_repro

        # ------------------------------------------------------------------
        # S: exact original persistent sequential O014 behavior.
        # ------------------------------------------------------------------
        seq_pairs = final_ref.rows_to_pairs(current_rows, uid_map, iid_map)
        t0 = time.perf_counter()
        sequential_model.partial_fit_recent(
            recent_pairs=seq_pairs,
            history_csr=eval_train_set.csr_matrix,
            max_steps=FROZEN_ONLINE_CONFIG["max_steps"],
            n_epochs=FROZEN_ONLINE_CONFIG["n_epochs"],
        )
        seq_time = time.perf_counter() - t0
        sequential_call_times.append(seq_time)
        sequential_history_rows.append(len(post_update_rows))
        assert_branch_invariants(
            sequential_model, base_v, uid_map, iid_map, "SEQUENTIAL_ORIGINAL"
        )
        seq_metrics = final_ref.evaluate_model(sequential_model, eval_train_set, test_set)
        seq_allwarm = final_ref.evaluate_model(sequential_model, eval_train_set, allwarm_test_set)
        seq_current = final_ref.evaluate_model(sequential_model, eval_train_set, current_test_set)
        seq_repro = compare_to_reference(
            seq_metrics, ref_row, "online", f"seed={seed} point={eval_point} sequential"
        )
        compare_to_reference(
            seq_allwarm, ref_row, "allwarm_online",
            f"seed={seed} point={eval_point} sequential allwarm"
        )

        cumulative_pairs = final_ref.rows_to_pairs(cumulative_recent_rows, uid_map, iid_map)

        # ------------------------------------------------------------------
        # H: retrospective fixed-point-history replay.
        # Fresh base at each point; C1..Ci stay separated into successive calls,
        # but every call uses the history already observed at the evaluated point.
        # This does not use eval chunk i+1 and is therefore not test leakage.
        # ------------------------------------------------------------------
        fixed_history = branch_model_from_base(base_train_set, base_u, base_v, seed)
        fixed_call_times = []
        for replay_idx in range(step_idx + 1):
            replay_rows = list(data["known_chunks"][replay_idx])
            replay_pairs = final_ref.rows_to_pairs(replay_rows, uid_map, iid_map)
            t0 = time.perf_counter()
            fixed_history.partial_fit_recent(
                recent_pairs=replay_pairs,
                history_csr=eval_train_set.csr_matrix,
                max_steps=FROZEN_ONLINE_CONFIG["max_steps"],
                n_epochs=FROZEN_ONLINE_CONFIG["n_epochs"],
            )
            fixed_call_times.append(time.perf_counter() - t0)
        assert_branch_invariants(
            fixed_history, base_v, uid_map, iid_map, "RETROSPECTIVE_FIXED_HISTORY"
        )
        fh_metrics = final_ref.evaluate_model(fixed_history, eval_train_set, test_set)
        fh_allwarm = final_ref.evaluate_model(fixed_history, eval_train_set, allwarm_test_set)
        fh_current = final_ref.evaluate_model(fixed_history, eval_train_set, current_test_set)

        # ------------------------------------------------------------------
        # C: fresh base, one cumulative call C1..Ci, point-final history.
        # The single point call uses seed+point-1 by frozen diagnostic rule.
        # ------------------------------------------------------------------
        point_seed = seed + step_idx
        reset_cum = branch_model_from_base(base_train_set, base_u, base_v, point_seed)
        t0 = time.perf_counter()
        reset_cum.partial_fit_recent(
            recent_pairs=cumulative_pairs,
            history_csr=eval_train_set.csr_matrix,
            max_steps=FROZEN_ONLINE_CONFIG["max_steps"],
            n_epochs=FROZEN_ONLINE_CONFIG["n_epochs"],
        )
        rcu_time = time.perf_counter() - t0
        assert_branch_invariants(reset_cum, base_v, uid_map, iid_map, "RESET_CUMULATIVE")
        rcu_metrics = final_ref.evaluate_model(reset_cum, eval_train_set, test_set)
        rcu_allwarm = final_ref.evaluate_model(reset_cum, eval_train_set, allwarm_test_set)
        rcu_current = final_ref.evaluate_model(reset_cum, eval_train_set, current_test_set)

        # ------------------------------------------------------------------
        # R: fresh base, one call only on current Ci, point-final history.
        # Interpretation focuses on CURRENT-USER matched population.
        # ------------------------------------------------------------------
        reset_current = branch_model_from_base(base_train_set, base_u, base_v, point_seed)
        current_pairs = final_ref.rows_to_pairs(current_rows, uid_map, iid_map)
        t0 = time.perf_counter()
        reset_current.partial_fit_recent(
            recent_pairs=current_pairs,
            history_csr=eval_train_set.csr_matrix,
            max_steps=FROZEN_ONLINE_CONFIG["max_steps"],
            n_epochs=FROZEN_ONLINE_CONFIG["n_epochs"],
        )
        rc_time = time.perf_counter() - t0
        assert_branch_invariants(reset_current, base_v, uid_map, iid_map, "RESET_CURRENT")
        rc_metrics = final_ref.evaluate_model(reset_current, eval_train_set, test_set)
        rc_allwarm = final_ref.evaluate_model(reset_current, eval_train_set, allwarm_test_set)
        rc_current = final_ref.evaluate_model(reset_current, eval_train_set, current_test_set)

        # Point-1 identity is a hard implementation guard, not a scientific result.
        point1_equal = ""
        if eval_point == 1:
            point1_equal = all(
                arrays_bitwise_equal(sequential_model.U, other.U)
                for other in [fixed_history, reset_cum, reset_current]
            )
            if not point1_equal:
                raise RuntimeError(
                    "Point-1 identity guard failed: all four branches must produce bitwise-identical U"
                )

        branch_specs = [
            {
                "branch": "SEQUENTIAL_ORIGINAL",
                "model": sequential_model,
                "metrics": seq_metrics,
                "allwarm": seq_allwarm,
                "current": seq_current,
                "state_pairs": cumulative_pairs,
                "last_call_pairs": len(seq_pairs),
                "n_calls": eval_point,
                "opt_steps": sum(
                    optimizer_steps(len(data["known_chunks"][i]))
                    for i in range(step_idx + 1)
                ),
                "last_call_steps": optimizer_steps(len(seq_pairs)),
                "last_call_time": seq_time,
                "state_time": float(sum(sequential_call_times)),
                "call_seeds": ";".join(str(seed + i) for i in range(step_idx + 1)),
                "history_rows": ";".join(str(x) for x in sequential_history_rows),
                "seq_ref_ok": seq_repro,
            },
            {
                "branch": "RETROSPECTIVE_FIXED_HISTORY",
                "model": fixed_history,
                "metrics": fh_metrics,
                "allwarm": fh_allwarm,
                "current": fh_current,
                "state_pairs": cumulative_pairs,
                "last_call_pairs": len(seq_pairs),
                "n_calls": eval_point,
                "opt_steps": sum(
                    optimizer_steps(len(data["known_chunks"][i]))
                    for i in range(step_idx + 1)
                ),
                "last_call_steps": optimizer_steps(len(seq_pairs)),
                "last_call_time": float(fixed_call_times[-1]),
                "state_time": float(sum(fixed_call_times)),
                "call_seeds": ";".join(str(seed + i) for i in range(step_idx + 1)),
                "history_rows": ";".join([str(len(post_update_rows))] * eval_point),
                "seq_ref_ok": False,
            },
            {
                "branch": "RESET_CUMULATIVE",
                "model": reset_cum,
                "metrics": rcu_metrics,
                "allwarm": rcu_allwarm,
                "current": rcu_current,
                "state_pairs": cumulative_pairs,
                "last_call_pairs": len(cumulative_pairs),
                "n_calls": 1,
                "opt_steps": optimizer_steps(len(cumulative_pairs)),
                "last_call_steps": optimizer_steps(len(cumulative_pairs)),
                "last_call_time": rcu_time,
                "state_time": rcu_time,
                "call_seeds": str(point_seed),
                "history_rows": str(len(post_update_rows)),
                "seq_ref_ok": False,
            },
            {
                "branch": "RESET_CURRENT",
                "model": reset_current,
                "metrics": rc_metrics,
                "allwarm": rc_allwarm,
                "current": rc_current,
                "state_pairs": current_pairs,
                "last_call_pairs": len(current_pairs),
                "n_calls": 1,
                "opt_steps": optimizer_steps(len(current_pairs)),
                "last_call_steps": optimizer_steps(len(current_pairs)),
                "last_call_time": rc_time,
                "state_time": rc_time,
                "call_seeds": str(point_seed),
                "history_rows": str(len(post_update_rows)),
                "seq_ref_ok": False,
            },
        ]

        for spec in branch_specs:
            branch = spec["branch"]
            model = spec["model"]
            metrics = spec["metrics"]
            allwarm_metrics = spec["allwarm"]
            current_metrics = spec["current"]
            state_pairs = spec["state_pairs"]
            n_up_users, per_mean, per_median, per_p95 = per_user_stats(state_pairs)
            update_users = (
                np.unique(state_pairs[:, 0]) if len(state_pairs)
                else np.asarray([], dtype=np.int64)
            )
            (
                u_fro_all, u_fro_update, cos_mean, cos_median, cos_p05,
                n_changed, non_update_max,
            ) = u_diagnostics(model.U, base_u, update_users)
            v_exact = arrays_bitwise_equal(np.asarray(model.V), base_v)
            v_diff = max_abs_diff(model.V, base_v)
            maps_ok = model.uid_map == uid_map and model.iid_map == iid_map

            row = {
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": p_hash,
                "data_sha256": EXPECTED_DATA_SHA256,
                "seed": seed,
                "eval_point": eval_point,
                "update_chunk": update_chunk,
                "eval_chunk": eval_chunk,
                "branch": branch,
                "latest_call_seed": point_seed,
                "n_recent_pairs_this_state": len(state_pairs),
                "n_recent_pairs_last_call": spec["last_call_pairs"],
                "n_partial_calls_to_state": spec["n_calls"],
                "optimizer_steps_this_state": spec["opt_steps"],
                "optimizer_steps_last_call": spec["last_call_steps"],
                "n_update_users_this_state": n_up_users,
                "interactions_per_update_user_mean": per_mean,
                "interactions_per_update_user_median": per_median,
                "interactions_per_update_user_p95": per_p95,
                "last_call_time_s": spec["last_call_time"],
                "state_construction_time_s": spec["state_time"],
                "call_seeds": spec["call_seeds"],
                "history_rows_by_call": spec["history_rows"],
                "n_eval_rows": len(eval_rows),
                "n_eval_users": len({r[0] for r in eval_rows}),
                "n_current_eval_rows": len(current_eval_rows),
                "n_current_eval_users": len({r[0] for r in current_eval_rows}),
                "n_allwarm_eval_rows": len(allwarm_eval_rows),
                "n_allwarm_eval_users": len({r[0] for r in allwarm_eval_rows}),
                "u_fro_diff_all": u_fro_all,
                "u_fro_diff_update_users": u_fro_update,
                "u_cosine_mean_update_users": cos_mean,
                "u_cosine_median_update_users": cos_median,
                "u_cosine_p05_update_users": cos_p05,
                "n_user_rows_bitwise_changed": n_changed,
                "u_max_abs_diff_non_update_users": non_update_max,
                "v_exact_equal_base": v_exact,
                "v_max_abs_diff_base": v_diff,
                "uid_map_exact": maps_ok,
                "iid_map_exact": maps_ok,
                "stale_reference_reproduced": stale_repro,
                "sequential_reference_reproduced": (
                    spec["seq_ref_ok"] if branch == "SEQUENTIAL_ORIGINAL" else ""
                ),
                "point1_all_branch_u_bitwise_equal": point1_equal if eval_point == 1 else "",
            }
            for metric in QUALITY_METRICS:
                full_ref = reference_metric(ref_row, "full", metric)
                seq_ref = reference_metric(ref_row, "online", metric)
                allwarm_full_ref = reference_metric(ref_row, "allwarm_full", metric)
                row[metric] = metrics[metric]
                row[f"stale_{metric}"] = stale_metrics[metric]
                row[f"full_reference_{metric}"] = full_ref
                row[f"branch_minus_stale_{metric}"] = metrics[metric] - stale_metrics[metric]
                row[f"branch_minus_full_reference_{metric}"] = metrics[metric] - full_ref
                row[f"reference_sequential_{metric}"] = seq_ref
                row[f"branch_minus_reference_sequential_{metric}"] = metrics[metric] - seq_ref
                row[f"allwarm_{metric}"] = allwarm_metrics[metric]
                row[f"allwarm_stale_{metric}"] = stale_allwarm[metric]
                row[f"allwarm_full_reference_{metric}"] = allwarm_full_ref
                row[f"allwarm_branch_minus_stale_{metric}"] = (
                    allwarm_metrics[metric] - stale_allwarm[metric]
                )
                row[f"current_{metric}"] = current_metrics[metric]
                row[f"current_stale_{metric}"] = stale_current[metric]
                row[f"current_branch_minus_stale_{metric}"] = (
                    current_metrics[metric] - stale_current[metric]
                )
            rows_out.append(row)

        # Post-evaluation guards: retrieval/evaluation must not alter model factors/maps.
        for label, model in [
            ("SEQUENTIAL_ORIGINAL", sequential_model),
            ("RETROSPECTIVE_FIXED_HISTORY", fixed_history),
            ("RESET_CUMULATIVE", reset_cum),
            ("RESET_CURRENT", reset_current),
        ]:
            assert_branch_invariants(model, base_v, uid_map, iid_map, label + " post-eval")

        ndcg = f"NDCG@{TOP_K}"
        print(
            f"  point {eval_point} | update_rows={len(current_rows):,} | PRIMARY={len(eval_rows):,} | "
            f"S={seq_metrics[ndcg]:.6f} | H={fh_metrics[ndcg]:.6f} | "
            f"C={rcu_metrics[ndcg]:.6f} | R={rc_metrics[ndcg]:.6f} | "
            f"stale={stale_metrics[ndcg]:.6f} | current-users rows={len(current_eval_rows):,} | "
            f"current S/H/C/R={seq_current[ndcg]:.6f}/{fh_current[ndcg]:.6f}/"
            f"{rcu_current[ndcg]:.6f}/{rc_current[ndcg]:.6f} | "
            f"full_ref={reference_metric(ref_row, 'full', ndcg):.6f}"
        )

        observed_rows = post_update_rows

    return rows_out

def build_trials(step_rows, p_hash):
    out = []
    for seed in SEEDS:
        for branch in BRANCHES:
            rows = [r for r in step_rows if int(r["seed"]) == seed and r["branch"] == branch]
            if len(rows) != 3:
                raise RuntimeError(f"Expected 3 step rows for seed={seed}, branch={branch}")
            trial = {
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": p_hash,
                "data_sha256": EXPECTED_DATA_SHA256,
                "seed": seed,
                "branch": branch,
                "n_eval_points": 3,
                "mean_last_call_time_s": float(np.mean([float(r["last_call_time_s"]) for r in rows])),
                "mean_state_construction_time_s": float(np.mean([float(r["state_construction_time_s"]) for r in rows])),
                "point3_state_construction_time_s": float([r for r in rows if int(r["eval_point"]) == 3][0]["state_construction_time_s"]),
                "all_v_exact_equal_base": all(str(r["v_exact_equal_base"]) == "True" if isinstance(r["v_exact_equal_base"], str) else bool(r["v_exact_equal_base"]) for r in rows),
                "max_v_abs_diff_base": max(float(r["v_max_abs_diff_base"]) for r in rows),
                "max_u_fro_diff_all": max(float(r["u_fro_diff_all"]) for r in rows),
            }
            for metric in QUALITY_METRICS:
                trial[f"mean_{metric}"] = float(np.mean([float(r[metric]) for r in rows]))
                trial[f"mean_stale_{metric}"] = float(np.mean([float(r[f"stale_{metric}"]) for r in rows]))
                trial[f"mean_full_reference_{metric}"] = float(np.mean([float(r[f"full_reference_{metric}"]) for r in rows]))
                trial[f"mean_branch_minus_stale_{metric}"] = float(np.mean([float(r[f"branch_minus_stale_{metric}"]) for r in rows]))
                trial[f"mean_branch_minus_full_reference_{metric}"] = float(np.mean([float(r[f"branch_minus_full_reference_{metric}"]) for r in rows]))
            out.append(trial)
    return out


def build_summary(step_rows, p_hash):
    s = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": p_hash,
        "data_sha256": EXPECTED_DATA_SHA256,
        "seeds": ";".join(map(str, SEEDS)),
        "branches": ";".join(BRANCHES),
        "n_step_rows": len(step_rows),
        "all_v_exact_equal_base": all(bool(r["v_exact_equal_base"]) for r in step_rows),
        "max_v_abs_diff_base": max(float(r["v_max_abs_diff_base"]) for r in step_rows),
        "all_sequential_reference_reproduced": all(
            bool(r["sequential_reference_reproduced"]) for r in step_rows
            if r["branch"] == "SEQUENTIAL_ORIGINAL"
        ),
        "all_stale_reference_reproduced": all(bool(r["stale_reference_reproduced"]) for r in step_rows),
    }
    ndcg = f"NDCG@{TOP_K}"
    for branch in BRANCHES:
        safe = branch.lower()
        rows = [r for r in step_rows if r["branch"] == branch]
        s[f"{safe}_mean_NDCG@20"] = float(np.mean([float(r[ndcg]) for r in rows]))
        s[f"{safe}_mean_minus_stale_NDCG@20"] = float(np.mean([float(r[f"branch_minus_stale_{ndcg}"]) for r in rows]))
        s[f"{safe}_mean_minus_full_reference_NDCG@20"] = float(np.mean([float(r[f"branch_minus_full_reference_{ndcg}"]) for r in rows]))
        s[f"{safe}_mean_state_construction_time_s"] = float(np.mean([float(r["state_construction_time_s"]) for r in rows]))
        p3row = [r for r in rows if int(r["eval_point"]) == 3]
        s[f"{safe}_point3_state_construction_time_s"] = float(np.mean([float(r["state_construction_time_s"]) for r in p3row]))
        s[f"{safe}_max_u_fro_diff_all"] = max(float(r["u_fro_diff_all"]) for r in rows)
        for point in [1, 2, 3]:
            pr = [r for r in rows if int(r["eval_point"]) == point]
            if len(pr) != len(SEEDS):
                raise RuntimeError(f"Expected {len(SEEDS)} rows branch={branch} point={point}")
            s[f"{safe}_mean_NDCG@20_point_{point}"] = float(np.mean([float(r[ndcg]) for r in pr]))
            s[f"{safe}_mean_minus_stale_NDCG@20_point_{point}"] = float(np.mean([float(r[f"branch_minus_stale_{ndcg}"]) for r in pr]))

    def mean_pair_diff(a, b, point=None):
        vals = []
        for seed in SEEDS:
            ar = [r for r in step_rows if r["branch"] == a and int(r["seed"]) == seed and (point is None or int(r["eval_point"]) == point)]
            br = [r for r in step_rows if r["branch"] == b and int(r["seed"]) == seed and (point is None or int(r["eval_point"]) == point)]
            amap = {int(r["eval_point"]): r for r in ar}
            bmap = {int(r["eval_point"]): r for r in br}
            for p in sorted(set(amap) & set(bmap)):
                vals.append(float(amap[p][ndcg]) - float(bmap[p][ndcg]))
        return float(np.mean(vals))

    s["mean_sequential_minus_fixed_history_NDCG@20"] = mean_pair_diff("SEQUENTIAL_ORIGINAL", "RETROSPECTIVE_FIXED_HISTORY")
    s["mean_fixed_history_minus_reset_cumulative_NDCG@20"] = mean_pair_diff("RETROSPECTIVE_FIXED_HISTORY", "RESET_CUMULATIVE")
    s["mean_sequential_minus_reset_cumulative_NDCG@20"] = mean_pair_diff("SEQUENTIAL_ORIGINAL", "RESET_CUMULATIVE")
    s["point3_sequential_minus_fixed_history_NDCG@20"] = mean_pair_diff("SEQUENTIAL_ORIGINAL", "RETROSPECTIVE_FIXED_HISTORY", 3)
    s["point3_fixed_history_minus_reset_cumulative_NDCG@20"] = mean_pair_diff("RETROSPECTIVE_FIXED_HISTORY", "RESET_CUMULATIVE", 3)
    s["point3_sequential_minus_reset_cumulative_NDCG@20"] = mean_pair_diff("SEQUENTIAL_ORIGINAL", "RESET_CUMULATIVE", 3)

    def mean_pair_diff_current(a, b, point=None):
        vals = []
        field = f"current_{ndcg}"
        for seed in SEEDS:
            ar = [r for r in step_rows if r["branch"] == a and int(r["seed"]) == seed and (point is None or int(r["eval_point"]) == point)]
            br = [r for r in step_rows if r["branch"] == b and int(r["seed"]) == seed and (point is None or int(r["eval_point"]) == point)]
            amap = {int(r["eval_point"]): r for r in ar}
            bmap = {int(r["eval_point"]): r for r in br}
            for p in sorted(set(amap) & set(bmap)):
                vals.append(float(amap[p][field]) - float(bmap[p][field]))
        return float(np.mean(vals))

    s["mean_current_sequential_minus_reset_current_NDCG@20"] = mean_pair_diff_current("SEQUENTIAL_ORIGINAL", "RESET_CURRENT")
    s["point3_current_sequential_minus_reset_current_NDCG@20"] = mean_pair_diff_current("SEQUENTIAL_ORIGINAL", "RESET_CURRENT", 3)
    point1_rows = [r for r in step_rows if int(r["eval_point"]) == 1]
    s["point1_all_branch_u_bitwise_equal"] = (
        len(point1_rows) == len(SEEDS) * len(BRANCHES)
        and all(str(r["point1_all_branch_u_bitwise_equal"]) == "True" if isinstance(r["point1_all_branch_u_bitwise_equal"], str) else bool(r["point1_all_branch_u_bitwise_equal"]) for r in point1_rows)
    )
    return s


def print_plan(data, data_hash, refs, p_hash, reference_path):
    print("=" * 118)
    print("ABLATION A1 V1.2 - UPDATE DYNAMICS - ONLINEIBPRMEJORADO")
    print("=" * 118)
    print(f"Protocol version                : {PROTOCOL_VERSION}")
    print(f"Protocol hash                   : {p_hash}")
    print(f"Script SHA256                   : {sha256_file(os.path.abspath(__file__))}")
    print(f"Data SHA256                     : {data_hash}")
    print(f"Reference H1-H3 steps           : {reference_path}")
    print(f"Reference steps SHA256          : {sha256_file(reference_path)}")
    print(f"Seeds                           : {SEEDS}")
    print(f"R900                            : {FROZEN_IBPR_CONFIG}")
    print(f"O014                            : {FROZEN_ONLINE_CONFIG}")
    print()
    print("Branches:")
    print("  S SEQUENTIAL_ORIGINAL          : real persistent prequential O014")
    print("  H RETROSPECTIVE_FIXED_HISTORY  : reset from base per point; C1..Ci separated; point-final history in every call")
    print("  C RESET_CUMULATIVE             : reset from base per point; C1..Ci in one call; point-final history")
    print("  R RESET_CURRENT                : reset from base per point; only Ci; point-final history")
    print()
    print("Seed policy:")
    print("  S/H multi-call states : seed, seed+1, ... seed+point-1")
    print("  C/R single call       : seed+point-1")
    print()
    print("Point-level workload (exact under max_steps=None):")
    seq_steps = 0
    for i in range(3):
        n_cur = len(data["known_chunks"][i])
        n_cum = sum(len(data["known_chunks"][j]) for j in range(i + 1))
        seq_steps += optimizer_steps(n_cur)
        print(
            f"  point {i+1}: current={n_cur:,}, cumulative={n_cum:,} | "
            f"S={seq_steps} | H={seq_steps} | C={optimizer_steps(n_cum)} | R={optimizer_steps(n_cur)} steps"
        )
    print()
    print("CURRENT-USER matched PRIMARY populations:")
    for row in matched_population_sizes(data):
        print(f"  point {row['eval_point']}: rows={row['rows']:,} | users={row['users']:,}")
    print()
    print("Hard guards:")
    print("  - S reproduces audited final H1-H3 Online metrics")
    print("  - Stale reproduces audited final H1-H3 Stale metrics")
    print("  - V remains bitwise equal to V_base in every branch/point")
    print("  - uid_map/iid_map remain exact")
    print("  - point 1 U must be bitwise identical across S/H/C/R")
    print("  - eval chunk is never included in any training/history used before its evaluation")
    print()
    print("Interpretation guard:")
    print("  H is a retrospective diagnostic control, not a deployable online policy.")
    print("  S-vs-H changes negative-sampling history for earlier calls.")
    print("  H-vs-C compares multiple calls vs one call but still bundles optimizer resets,")
    print("  intermediate normalization, shuffling/batching and seed progression.")
    print("  Do not call point-3 behavior generic temporal deterioration: PRIMARY populations differ by point.")
    print("  Report paired within-point contrasts; no retuning and no new O014 selection.")
    print()
    print("Reference guards:")
    print(f"  final H1-H3 protocol version  : {EXPECTED_FINAL_H1H3_PROTOCOL_VERSION}")
    print(f"  final H1-H3 protocol hash     : {EXPECTED_FINAL_H1H3_PROTOCOL_HASH}")
    print(f"  reference rows                : {len(refs)} / 6")
    print(f"  reference semantic SHA        : {EXPECTED_REFERENCE_STEPS_SEMANTIC_SHA256}")
    print("  expected-data validation      : PASS")

def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    validate_exact_environment_and_sources()
    data = final_ref.prepare_protocol_data()
    data_hash = validate_data(data)
    refs = load_reference_steps(args.reference_steps)
    p_hash = protocol_hash(data, data_hash, args.reference_steps)

    print_plan(data, data_hash, refs, p_hash, args.reference_steps)
    if args.plan_only:
        print("\nPLAN-ONLY complete. No MovieLens model was trained and no ablation result file was written.")
        return

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(RESULTS_DIR, f"ablation_a1_update_dynamics_{timestamp}")
    steps_path = prefix + "_steps.csv"
    trials_path = prefix + "_trials.csv"
    summary_path = prefix + "_summary.csv"
    log_path = prefix + ".txt"

    # Fresh timestamp only in v1.2 to prevent accidental mixing of partial experiments.
    for path in [steps_path, trials_path, summary_path, log_path]:
        if os.path.exists(path):
            raise RuntimeError(f"Output already exists: {path}. Use a new timestamp.")

    all_steps = []
    log_lines = []
    original_print = print

    # Lightweight tee for important console lines without redirecting library output.
    def tee(*values, **kwargs):
        text = " ".join(str(v) for v in values)
        log_lines.append(text)
        original_print(*values, **kwargs)

    globals()["print"] = tee
    try:
        tee("#" * 118)
        tee(f"ABLATION A1 TIMESTAMP: {timestamp}")
        tee("#" * 118)
        tee(f"Protocol: {p_hash}")
        tee(f"Data SHA: {data_hash}")
        tee(f"Reference steps: {args.reference_steps}")
        tee()
        for seed in SEEDS:
            seed_rows = run_seed(seed, data, refs, p_hash)
            all_steps.extend(seed_rows)
            save_csv_atomic(steps_path, STEP_FIELDS, all_steps)

        trials = build_trials(all_steps, p_hash)
        summary = build_summary(all_steps, p_hash)
        save_csv_atomic(steps_path, STEP_FIELDS, all_steps)
        save_csv_atomic(trials_path, TRIAL_FIELDS, trials)
        save_csv_atomic(summary_path, SUMMARY_FIELDS, [summary])

        tee()
        tee("=" * 118)
        tee("ABLATION A1 SUMMARY")
        tee("=" * 118)
        for branch in BRANCHES:
            safe = branch.lower()
            tee(
                f"{branch:18s} | mean NDCG={summary[f'{safe}_mean_NDCG@20']:.6f} | "
                f"vs Stale={summary[f'{safe}_mean_minus_stale_NDCG@20']:+.6f} | "
                f"vs FullRef={summary[f'{safe}_mean_minus_full_reference_NDCG@20']:+.6f}"
            )
        tee("Paired contrasts:")
        tee(f"  S - H                  : {summary['mean_sequential_minus_fixed_history_NDCG@20']:+.6f}")
        tee(f"  H - C                  : {summary['mean_fixed_history_minus_reset_cumulative_NDCG@20']:+.6f}")
        tee(f"  S - C                  : {summary['mean_sequential_minus_reset_cumulative_NDCG@20']:+.6f}")
        tee("Point 3 contrasts (original PRIMARY):")
        tee(f"  S - H                  : {summary['point3_sequential_minus_fixed_history_NDCG@20']:+.6f}")
        tee(f"  H - C                  : {summary['point3_fixed_history_minus_reset_cumulative_NDCG@20']:+.6f}")
        tee(f"  S - C                  : {summary['point3_sequential_minus_reset_cumulative_NDCG@20']:+.6f}")
        tee("RESET_CURRENT diagnostic (current-user matched PRIMARY subset):")
        tee(f"  mean S - R             : {summary['mean_current_sequential_minus_reset_current_NDCG@20']:+.6f}")
        tee(f"  point3 S - R           : {summary['point3_current_sequential_minus_reset_current_NDCG@20']:+.6f}")
        tee(f"Point-1 U identical S/H/C/R: {summary['point1_all_branch_u_bitwise_equal']}")
        tee(f"All V exact base         : {summary['all_v_exact_equal_base']}")
        tee(f"Sequential ref reproduced: {summary['all_sequential_reference_reproduced']}")
        tee(f"Stale ref reproduced     : {summary['all_stale_reference_reproduced']}")
        tee("Interpretation: descriptive within-point diagnostic only; do not retune O014 or infer a unique causal mechanism.")
        tee()
        tee(f"Steps   : {steps_path}")
        tee(f"Trials  : {trials_path}")
        tee(f"Summary : {summary_path}")
        tee(f"Log     : {log_path}")
    finally:
        globals()["print"] = original_print
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
