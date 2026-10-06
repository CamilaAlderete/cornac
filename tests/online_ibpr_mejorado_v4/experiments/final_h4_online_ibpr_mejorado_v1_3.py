import os

# Freeze numerical thread pools BEFORE importing numerical libraries.
for _name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ[_name] = "1"

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import inspect
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import cornac
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.models.recommender import ANNMixin, MEASURE_DOT

try:
    import faiss
except ImportError as exc:
    raise RuntimeError("FAISS no está disponible; H4 final requiere faiss.") from exc


# =============================================================================
# FINAL H4 EXECUTOR V1.3
# =============================================================================
# This executor is downstream of the frozen H4 V1.4 manifest.
# It MUST NOT retune or alter the frozen protocol.
#
# Modes:
#   --validate-only : validate freeze/data/code/environment/helpers; no final training.
#   --run-final     : run the final 5-seed x 3-point H4 campaign.
#
# A timestamp may be supplied to --run-final. Reusing that exact timestamp resumes
# only completed seed files that pass contract validation.
# =============================================================================

EXECUTOR_VERSION = "final_h4_executor_v1_3_20261005"

FREEZE_FILENAME = "h4_final_v1_4_freeze_manifest_20261002_061642.json"
FREEZE_SHA256 = "56fd0521e91dc52ab2141b97d0edece28efdc368df66befdd8db18a6e83d13fe"
FROZEN_PROTOCOL_HASH = "2578f488951af341d2f96b240300f0818e05323ba546a7aeb5e0f89b9bf28886"
FROZEN_PROTOCOL_VERSION = "h4_final_v1_4_freeze_20261002"

RATING_THRESHOLD = 3.0
VARIANT = "1M"
TARGET_BASE_FRAC = 0.60
N_STREAM_CHUNKS = 4

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENTS_DIR = SCRIPT_PATH.parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"


# =============================================================================
# Generic helpers
# =============================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj):
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.view(np.uint8).tobytes()).hexdigest()


def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def assert_finite_matrix(x, label):
    x = np.asarray(x)
    if x.ndim != 2:
        raise RuntimeError(f"{label}: expected 2D matrix, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{label}: NaN/Inf detectado.")


def sample_std(values):
    values = [float(x) for x in values]
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def mean(values):
    values = [float(x) for x in values]
    return float(statistics.fmean(values)) if values else float("nan")


def percentile(values, q):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def flatten_dict(d, prefix=""):
    out = {}
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            out[name] = value
    return out



# =============================================================================
# Frozen manifest validation
# =============================================================================

def resolve_paths():
    if EXPERIMENTS_DIR.name != "experiments":
        raise RuntimeError(
            "El ejecutor debe permanecer en tests/online_ibpr_mejorado_v4/experiments."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "script": SCRIPT_PATH,
        "experiments": EXPERIMENTS_DIR,
        "results": RESULTS_DIR,
        "freeze": RESULTS_DIR / FREEZE_FILENAME,
    }


def load_and_validate_freeze(path):
    if not path.exists():
        raise RuntimeError(f"No se encontró el freeze H4 final:\n  {path}")

    observed_sha = sha256_file(path)
    if observed_sha != FREEZE_SHA256:
        raise RuntimeError(
            "SHA256 del freeze H4 inesperado:\n"
            f"  observed={observed_sha}\n"
            f"  expected={FREEZE_SHA256}"
        )

    with open(path, "r", encoding="utf-8") as f:
        freeze = json.load(f)

    if freeze.get("status") != "FROZEN_PLAN_ONLY":
        raise RuntimeError("Freeze H4 no tiene status=FROZEN_PLAN_ONLY.")
    if freeze.get("protocol_version") != FROZEN_PROTOCOL_VERSION:
        raise RuntimeError("protocol_version del freeze H4 no coincide.")
    if freeze.get("protocol_hash") != FROZEN_PROTOCOL_HASH:
        raise RuntimeError("protocol_hash almacenado del freeze H4 no coincide.")

    protocol = freeze.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("Freeze H4 no contiene payload protocol válido.")

    recomputed = canonical_sha256(protocol)
    if recomputed != FROZEN_PROTOCOL_HASH:
        raise RuntimeError(
            "El payload protocol del freeze H4 no reproduce su hash congelado:\n"
            f"  recomputed={recomputed}\n"
            f"  expected  ={FROZEN_PROTOCOL_HASH}"
        )

    guard = freeze.get("execution_guard", {})
    required_false = [
        "h4_final_base_training_executed",
        "h4_final_full_training_executed",
        "h4_final_indices_built",
        "h4_final_metrics_evaluated",
        "h4_final_online_update_executed",
    ]
    if not guard.get("plan_only", False):
        raise RuntimeError("Freeze no fue generado en modo plan_only.")
    for key in required_false:
        if guard.get(key) is not False:
            raise RuntimeError(f"Freeze inválido: execution_guard.{key} != false.")
    if guard.get("retuning_after_freeze") != "PROHIBITED":
        raise RuntimeError("Freeze no prohíbe explícitamente retuning.")

    return freeze


# =============================================================================
# Runtime / source continuity
# =============================================================================

def runtime_environment():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(1)

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cornac": getattr(cornac, "__version__", None),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "faiss": getattr(faiss, "__version__", None),
        "thread_env": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            ]
        },
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "faiss_threads": (
            int(faiss.omp_get_max_threads())
            if hasattr(faiss, "omp_get_max_threads")
            else None
        ),
    }

    try:
        from threadpoolctl import threadpool_info
        env["native_threadpools"] = threadpool_info()
    except Exception as exc:
        raise RuntimeError(
            "threadpoolctl es requerido para validar threads nativos."
        ) from exc

    return env


def environment_signature(env):
    return {
        "python": env["python"],
        "platform": env["platform"],
        "machine": env["machine"],
        "processor": env["processor"],
        "logical_cpu_count": env["logical_cpu_count"],
        "cornac": env["cornac"],
        "numpy": env["numpy"],
        "scipy": env["scipy"],
        "torch": env["torch"],
        "faiss": env["faiss"],
        "thread_env": env["thread_env"],
        "torch_num_threads": env["torch_num_threads"],
        "torch_num_interop_threads": env["torch_num_interop_threads"],
        "faiss_threads": env["faiss_threads"],
    }


def validate_environment(env, freeze):
    expected = freeze["protocol"]["runtime_environment_signature"]
    observed = environment_signature(env)

    if observed != expected:
        raise RuntimeError(
            "El entorno actual no coincide exactamente con la firma congelada H4.\n"
            f"observed={json.dumps(observed, ensure_ascii=False, sort_keys=True)}\n"
            f"expected={json.dumps(expected, ensure_ascii=False, sort_keys=True)}"
        )

    for pool in env.get("native_threadpools", []):
        n_threads = pool.get("num_threads")
        if n_threads is not None and int(n_threads) != 1:
            raise RuntimeError(
                "Threadpool nativo no single-threaded: "
                f"{pool.get('user_api')}/{pool.get('internal_api')} "
                f"{pool.get('prefix')} num_threads={n_threads}"
            )


def discover_source_files():
    files = {}

    for label, cls in [
        ("ibpr_wrapper", IBPR),
        ("online_wrapper", OnlineIBPRMejorado),
    ]:
        source = inspect.getsourcefile(cls)
        if source is None:
            raise RuntimeError(f"No se pudo resolver source file para {label}.")
        files[label] = Path(source).resolve()

    ibpr_wrapper_module = importlib.import_module(IBPR.__module__)
    online_wrapper_module = importlib.import_module(OnlineIBPRMejorado.__module__)

    for label, wrapper_module, sibling in [
        ("ibpr_core", ibpr_wrapper_module, "ibpr.py"),
        ("online_core", online_wrapper_module, "online_ibpr_mejorado.py"),
    ]:
        wrapper_source = inspect.getsourcefile(wrapper_module)
        if wrapper_source is None:
            raise RuntimeError(f"No se pudo resolver wrapper module para {label}.")
        path = Path(wrapper_source).resolve().with_name(sibling)
        if not path.exists():
            raise RuntimeError(f"No existe source core esperado: {path}")
        files[label] = path

    return files


def validate_sources(freeze):
    expected = freeze["protocol"]["source_continuity"]["observed_hashes"]
    files = discover_source_files()
    observed = {label: sha256_file(path) for label, path in files.items()}

    if observed != expected:
        raise RuntimeError(
            "Source drift respecto del freeze H4:\n"
            f"observed={observed}\nexpected={expected}"
        )
    return files, observed


def assert_final_provenance_unchanged(context, executor_sha_at_start):
    current_executor_sha = sha256_file(SCRIPT_PATH)
    if current_executor_sha != executor_sha_at_start:
        raise RuntimeError(
            "El ejecutor cambió durante la campaña final:\n"
            f"  start={executor_sha_at_start}\n"
            f"  final={current_executor_sha}"
        )

    freeze_path = context["paths"]["freeze"]
    current_freeze_sha = sha256_file(freeze_path)
    if current_freeze_sha != FREEZE_SHA256:
        raise RuntimeError(
            "El freeze H4 cambió durante la campaña final:\n"
            f"  observed={current_freeze_sha}\n"
            f"  expected={FREEZE_SHA256}"
        )

    current_source_files = discover_source_files()
    current_source_hashes = {
        label: sha256_file(path)
        for label, path in current_source_files.items()
    }
    if current_source_hashes != context["source_hashes"]:
        raise RuntimeError(
            "Uno o más source files cambiaron durante la campaña final:\n"
            f"  start={context['source_hashes']}\n"
            f"  final={current_source_hashes}"
        )

    return {
        "executor_sha256": current_executor_sha,
        "freeze_manifest_sha256": current_freeze_sha,
        "source_hashes": current_source_hashes,
    }


def validate_wrapper_contracts():
    if not issubclass(IBPR, ANNMixin):
        raise RuntimeError("IBPR no hereda ANNMixin.")
    if not issubclass(OnlineIBPRMejorado, ANNMixin):
        raise RuntimeError("OnlineIBPRMejorado no hereda ANNMixin.")

    U = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    V = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

    for cls, name in [(IBPR, "IBPR"), (OnlineIBPRMejorado, "OnlineIBPRMejorado")]:
        kwargs = {
            "k": 2,
            "trainable": False,
            "init_params": {"U": U.copy(), "V": V.copy()},
            "name": f"h4_executor_contract_{name}",
        }
        if cls is OnlineIBPRMejorado:
            kwargs.update({"update_V": False, "normalize": True})
        model = cls(**kwargs)
        model.num_users = U.shape[0]
        model.num_items = V.shape[0]

        if model.get_vector_measure() != MEASURE_DOT:
            raise RuntimeError(f"{name} no declara MEASURE_DOT.")
        if not np.array_equal(model.get_user_vectors(), U):
            raise RuntimeError(f"{name}.get_user_vectors() no coincide.")
        if not np.array_equal(model.get_item_vectors(), V):
            raise RuntimeError(f"{name}.get_item_vectors() no coincide.")
        if not np.allclose(
            np.asarray(model.score(0)),
            V.dot(U[0]),
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError(f"{name}.score() != V @ U[user].")


# =============================================================================
# Exact H1-H3 data reconstruction
# =============================================================================

def rows_pairs(rows):
    return {(row[0], row[1]) for row in rows}


def assert_unique_user_item_pairs(rows, label):
    if len(rows_pairs(rows)) != len(rows):
        raise RuntimeError(f"{label}: pares (u,i) duplicados.")


def assert_dataset_row_count(dataset, expected_rows, label):
    expected_rows = int(expected_rows)
    actual_nnz = int(dataset.csr_matrix.nnz)
    if actual_nnz != expected_rows:
        raise RuntimeError(
            f"{label}: csr_nnz={actual_nnz}, expected={expected_rows}."
        )
    try:
        actual_uir = len(dataset.uir_tuple[0])
    except Exception:
        actual_uir = actual_nnz
    if int(actual_uir) != expected_rows:
        raise RuntimeError(
            f"{label}: uir_rows={actual_uir}, expected={expected_rows}."
        )


def ts_min(rows):
    return min(int(row[3]) for row in rows)


def ts_max(rows):
    return max(int(row[3]) for row in rows)


def dataset_sha256(rows):
    digest = hashlib.sha256()
    for u, i, value, timestamp, original_position in rows:
        payload = (
            f"{u}\t{i}\t{float(value):.1f}\t{int(timestamp)}\t"
            f"{int(original_position)}\n"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def c1_compatible_dataset_sha256(rows):
    digest = hashlib.sha256()
    for u, i, value, timestamp, _ in rows:
        payload = f"{u}\t{i}\t{float(value):.1f}\t{int(timestamp)}\n"
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def load_positive_chrono_movielens():
    data = movielens.load_feedback(fmt="UIRT", variant=VARIANT)
    positive = []
    for original_position, (u, i, rating, timestamp) in enumerate(data):
        if float(rating) >= RATING_THRESHOLD:
            positive.append(
                (str(u), str(i), 1.0, int(timestamp), int(original_position))
            )
    positive.sort(key=lambda row: (row[3], row[4]))
    if not positive:
        raise RuntimeError("MovieLens 1M no produjo positivos.")
    return positive


def split_base_future_strict(rows):
    n = len(rows)
    target = int(n * TARGET_BASE_FRAC)
    if target <= 0 or target >= n:
        raise RuntimeError("Corte 60/40 inválido.")

    effective = target
    last_base_ts = rows[target - 1][3]
    while effective < n and rows[effective][3] == last_base_ts:
        effective += 1
    if effective >= n:
        raise RuntimeError("Ajuste temporal consumió todo future.")

    base_rows = list(rows[:effective])
    future_rows = list(rows[effective:])
    if ts_max(base_rows) >= ts_min(future_rows):
        raise RuntimeError("Corte base/future no estrictamente cronológico.")

    return {
        "target_base_rows": target,
        "effective_base_rows": effective,
        "base_rows": base_rows,
        "future_rows": future_rows,
    }


def split_chrono_chunks_strict(rows, n_chunks=N_STREAM_CHUNKS):
    n = len(rows)
    boundaries = [0]

    for k in range(1, n_chunks):
        target = int(np.floor(n * k / n_chunks))
        target = max(target, boundaries[-1] + 1)
        effective = target
        boundary_ts = rows[target - 1][3]
        while effective < n and rows[effective][3] == boundary_ts:
            effective += 1
        if effective >= n or effective <= boundaries[-1]:
            raise RuntimeError("No se pudo crear límite temporal de warm chunk.")
        boundaries.append(effective)

    boundaries.append(n)
    chunks = [
        list(rows[boundaries[i]:boundaries[i + 1]])
        for i in range(n_chunks)
    ]
    for i in range(n_chunks - 1):
        if ts_max(chunks[i]) >= ts_min(chunks[i + 1]):
            raise RuntimeError("Warm chunks no estrictamente temporales.")
    return chunks, boundaries


def warm_start_filter(base_rows, future_rows):
    base_users = {row[0] for row in base_rows}
    base_items = {row[1] for row in base_rows}
    warm_rows = [
        row for row in future_rows
        if row[0] in base_users and row[1] in base_items
    ]
    return {
        "base_users": base_users,
        "base_items": base_items,
        "warm_future_rows": warm_rows,
        "n_future_raw": len(future_rows),
        "n_future_warm": len(warm_rows),
    }


def primary_eval_rows_for_step(known_chunks, step_idx):
    users_exposed = {
        row[0]
        for chunk in known_chunks[: step_idx + 1]
        for row in chunk
    }
    allwarm = list(known_chunks[step_idx + 1])
    primary = [row for row in allwarm if row[0] in users_exposed]
    if not primary:
        raise RuntimeError(f"point {step_idx+1}: PRIMARY vacío.")
    return primary, allwarm, users_exposed


def build_primary_eval_plan(known_chunks):
    out = []
    for step_idx in range(3):
        primary, allwarm, exposed = primary_eval_rows_for_step(
            known_chunks, step_idx
        )
        allwarm_users = {row[0] for row in allwarm}
        out.append({
            "point": step_idx + 1,
            "update_chunk": step_idx + 1,
            "eval_chunk": step_idx + 2,
            "users_exposed_to_update": len(exposed),
            "primary_rows": len(primary),
            "primary_users": len({row[0] for row in primary}),
            "primary_items": len({row[1] for row in primary}),
            "all_warm_rows": len(allwarm),
            "all_warm_users": len(allwarm_users),
            "unadapted_eval_users": len(allwarm_users - exposed),
        })
    return out


def prepare_protocol_data():
    all_rows = load_positive_chrono_movielens()
    split = split_base_future_strict(all_rows)
    warm = warm_start_filter(split["base_rows"], split["future_rows"])
    chunks, boundaries = split_chrono_chunks_strict(warm["warm_future_rows"])

    assert_unique_user_item_pairs(all_rows, "all_positive_rows")
    assert_unique_user_item_pairs(split["base_rows"], "base_rows")
    observed = set(rows_pairs(split["base_rows"]))
    for idx, chunk in enumerate(chunks, start=1):
        assert_unique_user_item_pairs(chunk, f"warm_chunk_{idx}")
        overlap = observed & rows_pairs(chunk)
        if overlap:
            raise RuntimeError(
                f"warm_chunk_{idx}: par ya observado: {next(iter(overlap))}"
            )
        observed.update(rows_pairs(chunk))

    return {
        "all_rows": all_rows,
        "data_sha256": dataset_sha256(all_rows),
        "c1_data_sha256": c1_compatible_dataset_sha256(all_rows),
        "target_base_rows": split["target_base_rows"],
        "effective_base_rows": split["effective_base_rows"],
        "base_rows": split["base_rows"],
        "future_rows": split["future_rows"],
        "known_chunks": chunks,
        "warm_chunk_boundaries": boundaries,
        "primary_eval_plan": build_primary_eval_plan(chunks),
        **warm,
    }


def validate_protocol_data(data, freeze):
    p = freeze["protocol"]
    plan = p["dataset_plan"]

    checks = {
        "data_sha256": (data["data_sha256"], p["parent"]["data_sha256"]),
        "c1_data_sha256": (data["c1_data_sha256"], p["parent"]["c1_data_sha256"]),
        "total_implicit_positive_rows": (
            len(data["all_rows"]), plan["total_implicit_positive_rows"]
        ),
        "target_base_rows": (data["target_base_rows"], plan["target_base_rows"]),
        "effective_base_rows": (
            data["effective_base_rows"], plan["effective_base_rows"]
        ),
        "effective_future_rows": (
            len(data["future_rows"]), plan["effective_future_rows"]
        ),
        "future_warm_rows": (
            data["n_future_warm"], plan["future_warm_rows"]
        ),
        "base_users": (len(data["base_users"]), plan["base_users"]),
        "base_items": (len(data["base_items"]), plan["base_items"]),
        "warm_chunk_boundaries": (
            data["warm_chunk_boundaries"], plan["warm_chunk_boundaries"]
        ),
        "primary_eval_plan": (
            data["primary_eval_plan"], plan["eval_points"]
        ),
    }

    for idx, chunk in enumerate(data["known_chunks"]):
        expected = plan["chunks"][idx]
        actual = {
            "chunk": idx + 1,
            "warm_rows": len(chunk),
            "users": len({r[0] for r in chunk}),
            "items": len({r[1] for r in chunk}),
            "ts_min": ts_min(chunk),
            "ts_max": ts_max(chunk),
        }
        checks[f"chunk_{idx+1}"] = (actual, expected)

    mismatches = [
        (label, actual, expected)
        for label, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        lines = [
            f"{label}: observed={actual!r}, expected={expected!r}"
            for label, actual, expected in mismatches
        ]
        raise RuntimeError(
            "La reconstrucción MovieLens no coincide con el freeze H4:\n  - "
            + "\n  - ".join(lines)
        )


# =============================================================================
# Cornac training chronology inherited from FINAL H1-H3
# =============================================================================

def cornac_rows(rows):
    return [(u, i, float(v), int(ts)) for u, i, v, ts, _ in rows]


def build_dataset(rows, uid_map=None, iid_map=None, seed=42, exclude_unknowns=False):
    kwargs = {
        "fmt": "UIRT",
        "seed": int(seed),
        "exclude_unknowns": bool(exclude_unknowns),
    }
    if uid_map is not None:
        kwargs["global_uid_map"] = dict(uid_map)
    if iid_map is not None:
        kwargs["global_iid_map"] = dict(iid_map)
    return Dataset.build(cornac_rows(rows), **kwargs)


def assert_dataset_contract(dataset, uid_map, iid_map, n_users, n_items, label):
    if dataset.num_users != n_users or dataset.num_items != n_items:
        raise RuntimeError(f"{label}: dimensiones incompatibles.")
    if dataset.uid_map != uid_map or dataset.iid_map != iid_map:
        raise RuntimeError(f"{label}: mappings incompatibles.")


def rows_to_pairs(rows, uid_map, iid_map):
    pairs = np.asarray(
        [[uid_map[row[0]], iid_map[row[1]]] for row in rows],
        dtype=np.int64,
    )
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise RuntimeError("recent_pairs shape inválida.")
    return pairs


def train_base_model(base_rows, seed, r900):
    train_set = build_dataset(base_rows, seed=seed, exclude_unknowns=False)
    assert_dataset_row_count(train_set, len(base_rows), "base_train_set")

    # Frozen seed policy: reset immediately before each fresh IBPR fit.
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = IBPR(
        k=r900["k"],
        max_iter=r900["max_iter"],
        learning_rate=r900["learning_rate"],
        lamda=r900["lamda"],
        batch_size=r900["batch_size"],
        verbose=False,
        name=f"H4_IBPR_R900_BASE_seed{seed}",
    )
    start = time.perf_counter()
    model.fit(train_set)
    elapsed = time.perf_counter() - start

    if model.U.shape != (train_set.num_users, r900["k"]):
        raise RuntimeError("U_base shape inválida.")
    if model.V.shape != (train_set.num_items, r900["k"]):
        raise RuntimeError("V_base shape inválida.")
    assert_finite_matrix(model.U, "U_base")
    assert_finite_matrix(model.V, "V_base")
    return model, train_set, elapsed


def initialize_online(base_train_set, base_u, base_v, seed, r900, o019):
    model = OnlineIBPRMejorado(
        k=r900["k"],
        max_iter=o019["n_epochs"],
        learning_rate=o019["learning_rate"],
        lamda=o019["lamda"],
        batch_size=o019["batch_size"],
        trainable=False,
        init_params={"U": base_u.copy(), "V": base_v.copy()},
        update_V=o019["update_V"],
        neg_sampling=o019["neg_sampling"],
        normalize=o019["normalize"],
        loss_mode=o019["loss_mode"],
        seed=seed,
        verbose=False,
        name=f"H4_OnlineIBPRMejorado_O019_seed{seed}",
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


def train_full_retrain(rows, uid_map, iid_map, n_users, n_items, seed, r900):
    train_set = build_dataset(
        rows,
        uid_map=uid_map,
        iid_map=iid_map,
        seed=seed,
        exclude_unknowns=False,
    )
    assert_dataset_contract(
        train_set, uid_map, iid_map, n_users, n_items, "full_train_set"
    )
    assert_dataset_row_count(train_set, len(rows), "full_train_set")

    np.random.seed(seed)
    torch.manual_seed(seed)
    model = IBPR(
        k=r900["k"],
        max_iter=r900["max_iter"],
        learning_rate=r900["learning_rate"],
        lamda=r900["lamda"],
        batch_size=r900["batch_size"],
        verbose=False,
        name=f"H4_IBPR_FULL_seed{seed}",
    )
    start = time.perf_counter()
    model.fit(train_set)
    elapsed = time.perf_counter() - start

    if model.U.shape != (n_users, r900["k"]):
        raise RuntimeError("Full U shape incompatible.")
    if model.V.shape != (n_items, r900["k"]):
        raise RuntimeError("Full V shape incompatible.")
    assert_finite_matrix(model.U, "U_full")
    assert_finite_matrix(model.V, "V_full")
    if model.uid_map != uid_map or model.iid_map != iid_map:
        raise RuntimeError("Full reconstruyó mappings incompatibles.")
    return model, elapsed


# =============================================================================
# FAISS: one instrumented builder for every final H4 IVF construction
# =============================================================================

def index_sha256(index):
    return hashlib.sha256(bytes(faiss.serialize_index(index))).hexdigest()


def index_fingerprint(index):
    return {
        "sha256": index_sha256(index),
        "ntotal": int(index.ntotal),
        "d": int(index.d),
        "metric_type": int(index.metric_type),
        "nlist": int(index.nlist),
        "nprobe": int(index.nprobe),
        "object_id": int(id(index)),
    }


def fingerprint_equal(a, b):
    keys = ["sha256", "ntotal", "d", "metric_type", "nlist", "nprobe"]
    return all(a[k] == b[k] for k in keys)


class IndexBuildTracker:
    def __init__(self, ann_cfg):
        self.ann_cfg = dict(ann_cfg)
        self.events = []

    def build_ivf(self, kind, label, V):
        # SINGLE BUILDER RULE: this is the only function in this executor that
        # constructs IndexIVFFlat.
        started = time.perf_counter_ns()

        vectors = np.array(V, dtype=np.float32, order="C", copy=True)
        quantizer = faiss.IndexFlatIP(vectors.shape[1])
        index = faiss.IndexIVFFlat(
            quantizer,
            vectors.shape[1],
            int(self.ann_cfg["nlist"]),
            faiss.METRIC_INNER_PRODUCT,
        )

        if hasattr(index, "cp") and hasattr(index.cp, "seed"):
            index.cp.seed = int(self.ann_cfg["ann_clustering_seed"])

        index.nprobe = int(self.ann_cfg["nprobe"])

        if int(index.d) != vectors.shape[1]:
            raise RuntimeError(f"{label}: dimensión FAISS inesperada.")
        if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
            raise RuntimeError(f"{label}: métrica FAISS no es inner product.")
        if int(index.nlist) != int(self.ann_cfg["nlist"]):
            raise RuntimeError(f"{label}: nlist inesperado.")
        if int(index.nprobe) != int(self.ann_cfg["nprobe"]):
            raise RuntimeError(f"{label}: nprobe inesperado.")

        index.train(vectors)
        if not index.is_trained:
            raise RuntimeError(f"{label}: IndexIVFFlat no entrenado.")
        index.add(vectors)

        if int(index.ntotal) != vectors.shape[0]:
            raise RuntimeError(
                f"{label}: ntotal={index.ntotal}, expected={vectors.shape[0]}"
            )

        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        event = {
            "kind": str(kind),
            "label": str(label),
            "elapsed_ms": float(elapsed_ms),
            "fingerprint": index_fingerprint(index),
        }
        self.events.append(event)
        return index, event

    def count(self, kind):
        return sum(e["kind"] == kind for e in self.events)

    def total(self):
        return len(self.events)


# =============================================================================
# Exact / ANN retrieval and metrics
# =============================================================================

def build_seen_by_user(rows, uid_map, iid_map):
    seen = {int(u_idx): set() for u_idx in uid_map.values()}
    for row in rows:
        seen[int(uid_map[row[0]])].add(int(iid_map[row[1]]))
    return seen


def exact_filtered_topk(U, V, user_ids, seen_by_user, k):
    V = np.asarray(V)
    item_ids = np.arange(V.shape[0], dtype=np.int64)
    output = []

    for user_idx in user_ids:
        scores = V.dot(np.asarray(U[user_idx]))
        order = np.lexsort((item_ids, -scores))
        seen = seen_by_user.get(int(user_idx), set())
        kept = [int(i) for i in order if int(i) not in seen]
        output.append(kept[: min(k, len(kept))])

    return output


def ann_filtered_topk(index, U, user_ids, seen_by_user, k, initial_k):
    queries = np.ascontiguousarray(
        np.asarray(U, dtype=np.float32)[np.asarray(user_ids, dtype=np.int64)],
        dtype=np.float32,
    )
    ntotal = int(index.ntotal)
    search_k = min(ntotal, int(initial_k))
    if search_k <= 0:
        return [[] for _ in user_ids], search_k

    while True:
        _, candidates = index.search(queries, search_k)
        output = []
        all_complete = True

        for row_idx, user_idx in enumerate(user_ids):
            observed = seen_by_user.get(int(user_idx), set())
            dedupe = set()
            kept = []

            for item in candidates[row_idx]:
                item = int(item)
                if item < 0 or item in observed or item in dedupe:
                    continue
                dedupe.add(item)
                kept.append(item)
                if len(kept) == k:
                    break

            output.append(kept)
            if len(kept) < k:
                all_complete = False

        if all_complete or search_k >= ntotal:
            return output, search_k

        search_k = min(ntotal, max(search_k + 1, search_k * 2))


def compare_lists(reference, retrieved):
    ref = list(reference)
    got = list(retrieved)
    if not ref:
        return {
            "set_recall_at_20": 1.0 if not got else 0.0,
            "positional_agreement": 1.0 if not got else 0.0,
            "exact_ordered_list_match": bool(ref == got),
            "candidate_shortfall": 0,
        }

    recall = len(set(ref) & set(got)) / len(ref)
    positional = sum(
        1 for j, item in enumerate(ref)
        if j < len(got) and got[j] == item
    ) / len(ref)
    return {
        "set_recall_at_20": float(recall),
        "positional_agreement": float(positional),
        "exact_ordered_list_match": bool(ref == got),
        "candidate_shortfall": int(max(0, len(ref) - len(got))),
    }


def geometry_metrics(base_exact, full_exact, k):
    overlap = len(set(base_exact) & set(full_exact)) / k

    reference_len = len(base_exact)
    if reference_len == 0:
        positional = 1.0 if len(full_exact) == 0 else 0.0
    else:
        positional = sum(
            1 for j in range(reference_len)
            if j < len(full_exact) and base_exact[j] == full_exact[j]
        ) / reference_len

    return {
        "set_overlap_at_20": float(overlap),
        "positional_agreement": float(positional),
        "exact_ordered_list_match": bool(base_exact == full_exact),
    }


def aggregate_comparisons(rows):
    if not rows:
        raise RuntimeError("No hay filas de comparación para agregar.")

    return {
        "mean_set_recall_at_20": mean(r["set_recall_at_20"] for r in rows),
        "mean_positional_agreement": mean(
            r["positional_agreement"] for r in rows
        ),
        "exact_ordered_list_match_fraction": mean(
            1.0 if r["exact_ordered_list_match"] else 0.0 for r in rows
        ),
        "candidate_shortfall_total": int(
            sum(int(r["candidate_shortfall"]) for r in rows)
        ),
        "candidate_shortfall_affected_users": int(
            sum(int(r["candidate_shortfall"]) > 0 for r in rows)
        ),
        "candidate_shortfall_fraction": mean(
            1.0 if int(r["candidate_shortfall"]) > 0 else 0.0 for r in rows
        ),
    }


def aggregate_geometry(rows):
    return {
        "mean_set_overlap_at_20": mean(r["set_overlap_at_20"] for r in rows),
        "mean_positional_agreement": mean(
            r["positional_agreement"] for r in rows
        ),
        "exact_ordered_list_match_fraction": mean(
            1.0 if r["exact_ordered_list_match"] else 0.0 for r in rows
        ),
    }


def latency_summary(fn, warmups, repeats):
    for _ in range(int(warmups)):
        fn()

    samples = []
    for _ in range(int(repeats)):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)

    return {
        "median_batch_ms": float(statistics.median(samples)),
        "p95_batch_ms": percentile(samples, 95),
        "n_repeats": len(samples),
    }


def benchmark_branch(index, U, user_ids, seen_by_user, ann_cfg):
    user_ids_array = np.asarray(user_ids, dtype=np.int64)
    queries = np.ascontiguousarray(
        np.asarray(U, dtype=np.float32)[user_ids_array],
        dtype=np.float32,
    )
    raw_k = min(int(ann_cfg["raw_faiss_k"]), int(index.ntotal))

    raw = latency_summary(
        lambda: index.search(queries, raw_k),
        ann_cfg["latency_warmups"],
        ann_cfg["latency_repeats"],
    )
    filtered = latency_summary(
        lambda: ann_filtered_topk(
            index,
            U,
            user_ids,
            seen_by_user,
            int(ann_cfg["top_k"]),
            int(ann_cfg["initial_search_k"]),
        ),
        ann_cfg["latency_warmups"],
        ann_cfg["latency_repeats"],
    )

    n = len(user_ids)
    raw["median_ms_per_user"] = raw["median_batch_ms"] / n
    raw["p95_ms_per_user"] = raw["p95_batch_ms"] / n
    filtered["median_ms_per_user"] = filtered["median_batch_ms"] / n
    filtered["p95_ms_per_user"] = filtered["p95_batch_ms"] / n

    return {"raw": raw, "filtered": filtered}


def row_l2_deltas(after, before):
    return np.linalg.norm(
        np.asarray(after, dtype=np.float64)
        - np.asarray(before, dtype=np.float64),
        axis=1,
    )


def self_test_helpers(frozen_ann_cfg):
    # Basic exact/metric tests.
    U = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    V = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    seen = {0: {0}, 1: {2}}
    exact = exact_filtered_topk(U, V, [0, 1], seen, 2)
    if exact != [[1, 3], [3, 1]]:
        raise RuntimeError(f"Self-test exact top-k inesperado: {exact}")

    cmp = compare_lists([1, 3], [1, 2])
    if cmp["set_recall_at_20"] != 0.5:
        raise RuntimeError("Self-test recall falló.")
    if cmp["positional_agreement"] != 0.5:
        raise RuntimeError("Self-test positional agreement falló.")

    geom = geometry_metrics([1, 3], [1, 2], 2)
    if geom["set_overlap_at_20"] != 0.5:
        raise RuntimeError("Self-test geometry overlap falló.")

    # Executor ANN pipeline test. nlist=1/nprobe=1 deliberately converts IVF into
    # a full-list scan so ANN and exhaustive references must coincide exactly.
    # This is synthetic validation only; it does not alter frozen final H4 params.
    rng = np.random.default_rng(20261005)
    U_syn = rng.normal(size=(4, 8)).astype(np.float32)
    V_syn = rng.normal(size=(64, 8)).astype(np.float32)

    U_syn /= np.linalg.norm(U_syn, axis=1, keepdims=True)
    V_syn /= np.linalg.norm(V_syn, axis=1, keepdims=True)

    self_cfg = dict(frozen_ann_cfg)
    self_cfg["nlist"] = 1
    self_cfg["nprobe"] = 1
    self_cfg["top_k"] = 5
    self_cfg["initial_search_k"] = 2
    self_cfg["raw_faiss_k"] = 8

    tracker = IndexBuildTracker(self_cfg)
    index, event = tracker.build_ivf("selftest", "executor_ann_selftest", V_syn)

    if tracker.total() != 1 or tracker.count("selftest") != 1:
        raise RuntimeError("Self-test ANN builder accounting falló.")

    fp_before = index_fingerprint(index)
    if not fingerprint_equal(event["fingerprint"], fp_before):
        raise RuntimeError("Self-test fingerprint inicial inconsistente.")

    # Force adaptive growth by marking each user's two highest unfiltered items seen.
    item_ids = np.arange(V_syn.shape[0], dtype=np.int64)
    seen_syn = {}
    for u in range(U_syn.shape[0]):
        scores = V_syn.dot(U_syn[u])
        order = np.lexsort((item_ids, -scores))
        seen_syn[u] = {int(order[0]), int(order[1])}

    user_ids = list(range(U_syn.shape[0]))
    exact_syn = exact_filtered_topk(
        U_syn, V_syn, user_ids, seen_syn, self_cfg["top_k"]
    )
    ann_syn, final_search_k = ann_filtered_topk(
        index,
        U_syn,
        user_ids,
        seen_syn,
        self_cfg["top_k"],
        self_cfg["initial_search_k"],
    )

    if final_search_k <= self_cfg["initial_search_k"]:
        raise RuntimeError("Self-test no ejercitó adaptive search growth.")
    if ann_syn != exact_syn:
        raise RuntimeError(
            "Self-test ANN/exhaustive no coincide bajo nlist=1/nprobe=1."
        )

    for ref, got in zip(exact_syn, ann_syn):
        metrics = compare_lists(ref, got)
        if metrics["set_recall_at_20"] != 1.0:
            raise RuntimeError("Self-test ANN recall != 1.")
        if metrics["positional_agreement"] != 1.0:
            raise RuntimeError("Self-test ANN positional agreement != 1.")
        if not metrics["exact_ordered_list_match"]:
            raise RuntimeError("Self-test ANN ordered list mismatch.")
        if metrics["candidate_shortfall"] != 0:
            raise RuntimeError("Self-test ANN shortfall inesperado.")

    fp_after = index_fingerprint(index)
    if not fingerprint_equal(fp_before, fp_after):
        raise RuntimeError("Self-test ANN query mutó el índice.")

    return {
        "basic_metrics_pass": True,
        "ann_builder_pass": True,
        "adaptive_filtering_pass": True,
        "ann_exact_coherence_full_scan_pass": True,
        "index_fingerprint_stable_pass": True,
    }


# =============================================================================
# Seed-level final execution
# =============================================================================

def run_seed(seed, data, freeze):
    p = freeze["protocol"]
    r900 = p["r900"]
    o019 = p["o019"]
    ann_cfg = p["h4_ann"]
    top_k = int(ann_cfg["top_k"])

    print()
    print("=" * 118)
    print(f"H4 FINAL | seed={seed}")
    print("=" * 118)

    base_model, base_train_set, base_train_time = train_base_model(
        data["base_rows"], seed, r900
    )

    uid_map = dict(base_train_set.uid_map)
    iid_map = dict(base_train_set.iid_map)
    n_users = int(base_train_set.num_users)
    n_items = int(base_train_set.num_items)

    if n_users != p["dataset_plan"]["base_users"]:
        raise RuntimeError("base user count inconsistente.")
    if n_items != p["dataset_plan"]["base_items"]:
        raise RuntimeError("base item count inconsistente.")

    base_u = np.asarray(base_model.U).copy()
    base_v = np.asarray(base_model.V).copy()
    base_v_hash = array_sha256(base_v)

    online = initialize_online(
        base_train_set, base_u, base_v, seed, r900, o019
    )

    tracker = IndexBuildTracker(ann_cfg)
    base_index, base_build_event = tracker.build_ivf(
        "base", f"base_seed{seed}", base_v
    )
    base_fp_initial = index_fingerprint(base_index)
    base_object_id = id(base_index)

    if tracker.count("base") != 1 or tracker.total() != 1:
        raise RuntimeError("Conteo de build base inválido.")

    inverse_uid_map = {int(v): str(k) for k, v in uid_map.items()}
    observed_rows = list(data["base_rows"])
    steps = []

    for step_idx in range(3):
        point = step_idx + 1
        update_rows = list(data["known_chunks"][step_idx])
        primary_rows, allwarm_rows, _ = primary_eval_rows_for_step(
            data["known_chunks"], step_idx
        )

        if ts_max(update_rows) >= ts_min(allwarm_rows):
            raise RuntimeError(f"point {point}: update/eval temporal inválido.")

        eval_pairs = rows_pairs(allwarm_rows)
        post_update_rows = observed_rows + update_rows
        if eval_pairs & rows_pairs(post_update_rows):
            raise RuntimeError(f"point {point}: leakage de pares hacia eval.")

        history_set = build_dataset(
            post_update_rows,
            uid_map=uid_map,
            iid_map=iid_map,
            seed=seed,
            exclude_unknowns=False,
        )
        assert_dataset_contract(
            history_set,
            uid_map,
            iid_map,
            n_users,
            n_items,
            f"history_set_p{point}",
        )
        assert_dataset_row_count(
            history_set, len(post_update_rows), f"history_set_p{point}"
        )

        # Non-trivial Online update evidence is measured on CURRENT update users.
        current_update_user_ids = sorted({
            int(uid_map[row[0]]) for row in update_rows
        })
        online_u_before = np.asarray(online.U).copy()
        online_v_before = np.asarray(online.V).copy()
        online_builds_before = tracker.total()

        recent_pairs = rows_to_pairs(update_rows, uid_map, iid_map)
        online_update_start = time.perf_counter()
        online.partial_fit_recent(
            recent_pairs=recent_pairs,
            history_csr=history_set.csr_matrix,
            max_steps=o019["max_steps"],
            n_epochs=o019["n_epochs"],
        )
        online_update_time_s = time.perf_counter() - online_update_start

        online_build_delta = tracker.total() - online_builds_before
        if online_build_delta != 0:
            raise RuntimeError(f"point {point}: Online construyó/reconstruyó índice.")

        assert_finite_matrix(online.U, f"U_online_p{point}")
        assert_finite_matrix(online.V, f"V_online_p{point}")

        online_v_after_update = np.asarray(online.V).copy()
        v_equal_previous = bool(np.array_equal(online_v_after_update, online_v_before))
        v_equal = bool(np.array_equal(online_v_after_update, base_v))
        v_diff = max_abs_diff(online_v_after_update, base_v)
        v_hash = array_sha256(online_v_after_update)

        # IMPORTANT: V inequality is an H4 outcome, not an executor exception.
        # We keep running so a negative H4 result is preserved rather than censored.
        if online.uid_map != uid_map or online.iid_map != iid_map:
            raise RuntimeError(f"point {point}: Online mappings cambiaron.")

        deltas = row_l2_deltas(
            np.asarray(online.U)[current_update_user_ids],
            online_u_before[current_update_user_ids],
        )
        n_changed = int(np.sum(deltas > 0.0))
        current_update_user_deltas_l2 = [
            {
                "user_idx": int(user_idx),
                "raw_user_id": inverse_uid_map[int(user_idx)],
                "delta_U_l2": float(delta),
            }
            for user_idx, delta in zip(current_update_user_ids, deltas)
        ]
        u_update_evidence = {
            "n_current_update_users": len(current_update_user_ids),
            "n_current_update_users_with_delta_gt_0": n_changed,
            "fraction_current_update_users_with_delta_gt_0": (
                n_changed / len(current_update_user_ids)
            ),
            "median_current_update_user_delta_l2": float(np.median(deltas)),
            "max_current_update_user_delta_l2": float(np.max(deltas)),
            "current_update_user_deltas_l2": current_update_user_deltas_l2,
        }
        if n_changed < 1:
            raise RuntimeError(
                f"point {point}: update Online fue trivial en usuarios actuales."
            )

        # Fresh Full retrain with the SAME final trial seed.
        full_model, full_train_time_s = train_full_retrain(
            post_update_rows,
            uid_map,
            iid_map,
            n_users,
            n_items,
            seed,
            r900,
        )

        if full_model.uid_map != uid_map or full_model.iid_map != iid_map:
            raise RuntimeError(f"point {point}: Full mappings cambiaron.")

        full_index, full_build_event = tracker.build_ivf(
            "full_rebuilt",
            f"full_rebuilt_seed{seed}_p{point}",
            np.asarray(full_model.V),
        )

        # Base-index operational integrity BEFORE retrieval.
        base_fp_before = index_fingerprint(base_index)
        if not fingerprint_equal(base_fp_initial, base_fp_before):
            raise RuntimeError(
                f"point {point}: fingerprint base index cambió antes de queries."
            )

        query_user_ids = sorted({
            int(uid_map[row[0]]) for row in primary_rows
        })
        expected_primary_users = p["dataset_plan"]["eval_points"][step_idx][
            "primary_users"
        ]
        if len(query_user_ids) != int(expected_primary_users):
            raise RuntimeError(
                f"point {point}: PRIMARY users={len(query_user_ids)}, "
                f"expected={expected_primary_users}"
            )

        seen_by_user = build_seen_by_user(post_update_rows, uid_map, iid_map)

        # Frozen exhaustive references.
        exact_online_base = exact_filtered_topk(
            online.U, base_v, query_user_ids, seen_by_user, top_k
        )
        exact_full_base = exact_filtered_topk(
            full_model.U, base_v, query_user_ids, seen_by_user, top_k
        )
        exact_full_full = exact_filtered_topk(
            full_model.U, full_model.V, query_user_ids, seen_by_user, top_k
        )

        # Frozen ANN branches.
        ann_online, search_k_online = ann_filtered_topk(
            base_index,
            online.U,
            query_user_ids,
            seen_by_user,
            top_k,
            ann_cfg["initial_search_k"],
        )
        ann_full_stale, search_k_full_stale = ann_filtered_topk(
            base_index,
            full_model.U,
            query_user_ids,
            seen_by_user,
            top_k,
            ann_cfg["initial_search_k"],
        )
        ann_full_rebuilt, search_k_full_rebuilt = ann_filtered_topk(
            full_index,
            full_model.U,
            query_user_ids,
            seen_by_user,
            top_k,
            ann_cfg["initial_search_k"],
        )

        comparison_rows = {
            "online_reused": [],
            "full_stale_index_consistency": [],
            "full_stale_model_fidelity": [],
            "full_rebuilt": [],
        }
        geometry_rows = []
        query_records = []

        for j, user_idx in enumerate(query_user_ids):
            comparisons = {
                "online_reused": (
                    exact_online_base[j], ann_online[j]
                ),
                "full_stale_index_consistency": (
                    exact_full_base[j], ann_full_stale[j]
                ),
                "full_stale_model_fidelity": (
                    exact_full_full[j], ann_full_stale[j]
                ),
                "full_rebuilt": (
                    exact_full_full[j], ann_full_rebuilt[j]
                ),
            }

            for reference_name, (ref, got) in comparisons.items():
                metrics = compare_lists(ref, got)
                comparison_rows[reference_name].append(metrics)
                query_records.append({
                    "seed": int(seed),
                    "point": point,
                    "reference": reference_name,
                    "user_idx": int(user_idx),
                    "raw_user_id": inverse_uid_map[int(user_idx)],
                    "n_observed_items": len(seen_by_user.get(user_idx, set())),
                    "exact_items": list(ref),
                    "retrieved_items": list(got),
                    **metrics,
                })

            geom = geometry_metrics(
                exact_full_base[j], exact_full_full[j], top_k
            )
            geometry_rows.append(geom)
            query_records.append({
                "seed": int(seed),
                "point": point,
                "reference": "exact_geometry_shift",
                "user_idx": int(user_idx),
                "raw_user_id": inverse_uid_map[int(user_idx)],
                "n_observed_items": len(seen_by_user.get(user_idx, set())),
                "exact_base_items": exact_full_base[j],
                "exact_full_items": exact_full_full[j],
                **geom,
            })

        retrieval = {
            name: aggregate_comparisons(rows)
            for name, rows in comparison_rows.items()
        }
        retrieval["exact_geometry_shift"] = aggregate_geometry(geometry_rows)

        # Frozen latency: same PRIMARY batch, raw and filtered; build separate.
        latency = {
            "online_reused": benchmark_branch(
                base_index, online.U, query_user_ids, seen_by_user, ann_cfg
            ),
            "full_stale": benchmark_branch(
                base_index, full_model.U, query_user_ids, seen_by_user, ann_cfg
            ),
            "full_rebuilt": benchmark_branch(
                full_index, full_model.U, query_user_ids, seen_by_user, ann_cfg
            ),
        }

        # Retrieval and latency must not mutate base index or model factors.
        base_fp_after = index_fingerprint(base_index)
        full_fp_after = index_fingerprint(full_index)

        if not fingerprint_equal(base_fp_initial, base_fp_after):
            raise RuntimeError(
                f"point {point}: base index cambió durante retrieval/latency."
            )
        if not fingerprint_equal(
            full_build_event["fingerprint"], full_fp_after
        ):
            raise RuntimeError(
                f"point {point}: full rebuilt index cambió durante queries."
            )
        if not np.array_equal(online.V, online_v_after_update):
            raise RuntimeError(
                f"point {point}: retrieval/latency mutó V_online después del update."
            )

        step = {
            "seed": int(seed),
            "point": point,
            "update_chunk": point,
            "eval_chunk": point + 1,
            "n_update_rows": len(update_rows),
            "n_primary_rows": len(primary_rows),
            "n_primary_users": len(query_user_ids),
            "online_update_time_s": float(online_update_time_s),
            "full_train_time_s": float(full_train_time_s),
            "structural": {
                "v_online_array_equal_previous": v_equal_previous,
                "v_online_array_equal_base": v_equal,
                "v_online_max_abs_diff_base": v_diff,
                "v_online_sha256": v_hash,
                "v_base_sha256": base_v_hash,
                "online_build_delta": int(online_build_delta),
                "base_item_mapping_equal_online": bool(
                    online.iid_map == iid_map
                ),
                "base_item_mapping_equal_full": bool(
                    full_model.iid_map == iid_map
                ),
            },
            "u_update_evidence": u_update_evidence,
            "base_index_initial_fingerprint": base_fp_initial,
            "base_index_before_queries_fingerprint": base_fp_before,
            "base_index_after_queries_fingerprint": base_fp_after,
            "base_index_fingerprint_unchanged": bool(
                fingerprint_equal(base_fp_initial, base_fp_before)
                and fingerprint_equal(base_fp_initial, base_fp_after)
            ),
            "base_index_object_identity_same": bool(
                id(base_index) == base_object_id
            ),
            "full_rebuilt_index_fingerprint": full_fp_after,
            "search_k_final": {
                "online_reused": int(search_k_online),
                "full_stale": int(search_k_full_stale),
                "full_rebuilt": int(search_k_full_rebuilt),
            },
            "builds_so_far": {
                "base": tracker.count("base"),
                "online": tracker.count("online"),
                "full_stale": tracker.count("full_stale"),
                "full_rebuilt": tracker.count("full_rebuilt"),
                "total": tracker.total(),
            },
            "build_time_ms": {
                "base": float(base_build_event["elapsed_ms"]),
                "full_rebuilt": float(full_build_event["elapsed_ms"]),
            },
            "retrieval": retrieval,
            "latency": latency,
            "query_records": query_records,
        }
        steps.append(step)

        print(
            f"point={point} | V_exact={v_equal} | dV={v_diff:.3g} | "
            f"dU>0={n_changed}/{len(current_update_user_ids)} | "
            f"online_build_delta={online_build_delta} | "
            f"online recall={retrieval['online_reused']['mean_set_recall_at_20']:.6f} | "
            f"full_stale model fidelity={retrieval['full_stale_model_fidelity']['mean_set_recall_at_20']:.6f} | "
            f"full_rebuilt recall={retrieval['full_rebuilt']['mean_set_recall_at_20']:.6f}"
        )

        observed_rows = post_update_rows

    expected = p["expected_build_counts"]["per_seed"]
    actual_builds = {
        "base": tracker.count("base"),
        "online": tracker.count("online"),
        "full_stale": tracker.count("full_stale"),
        "full_rebuilt": tracker.count("full_rebuilt"),
        "total": tracker.total(),
    }
    if actual_builds != expected:
        raise RuntimeError(
            f"seed {seed}: build counts {actual_builds} != {expected}"
        )

    structural_success = all(
        s["structural"]["v_online_array_equal_base"]
        and s["structural"]["v_online_max_abs_diff_base"] == 0.0
        and s["structural"]["v_online_sha256"] == s["structural"]["v_base_sha256"]
        and s["structural"]["online_build_delta"] == 0
        for s in steps
    )
    integrity_guard = all(
        s["base_index_fingerprint_unchanged"] for s in steps
    )
    update_guard = all(
        s["u_update_evidence"]["n_current_update_users_with_delta_gt_0"] >= 1
        for s in steps
    )

    # A false structural_success is a legitimate negative H4 outcome.
    # Integrity/update guards, by contrast, define execution validity.
    if not integrity_guard:
        raise RuntimeError(f"seed {seed}: integrity guard del base index falló.")
    if not update_guard:
        raise RuntimeError(f"seed {seed}: Online update nontriviality guard falló.")

    return {
        "status": "COMPLETE",
        "executor_version": EXECUTOR_VERSION,
        "frozen_protocol_version": FROZEN_PROTOCOL_VERSION,
        "frozen_protocol_hash": FROZEN_PROTOCOL_HASH,
        "freeze_manifest_sha256": FREEZE_SHA256,
        "seed": int(seed),
        "base_train_time_s": float(base_train_time),
        "base_v_sha256": base_v_hash,
        "build_counts": actual_builds,
        "build_events": tracker.events,
        "criteria": {
            "structural_success": structural_success,
            "operational_integrity_guard": integrity_guard,
            "online_update_nontriviality_guard": update_guard,
        },
        "steps": steps,
    }


# =============================================================================
# Aggregation / persistence
# =============================================================================

CONTINUOUS_RETRIEVAL_METRICS = [
    "mean_set_recall_at_20",
    "mean_positional_agreement",
    "exact_ordered_list_match_fraction",
    "candidate_shortfall_fraction",
]

REFERENCE_NAMES = [
    "online_reused",
    "full_stale_index_consistency",
    "full_stale_model_fidelity",
    "full_rebuilt",
]

LATENCY_BRANCHES = ["online_reused", "full_stale", "full_rebuilt"]


def seed_level_summary(seed_result):
    out = {"seed": seed_result["seed"]}

    base_events = [
        e["elapsed_ms"] for e in seed_result["build_events"]
        if e["kind"] == "base"
    ]
    full_events = [
        e["elapsed_ms"] for e in seed_result["build_events"]
        if e["kind"] == "full_rebuilt"
    ]
    if len(base_events) != 1 or len(full_events) != 3:
        raise RuntimeError(
            f"seed {seed_result['seed']}: eventos de build inesperados."
        )
    out["build.base_index_ms"] = float(base_events[0])
    out["build.mean_full_rebuilt_index_ms"] = mean(full_events)
    out["build.total_full_rebuilt_index_ms"] = float(sum(full_events))

    for reference in REFERENCE_NAMES:
        for metric in CONTINUOUS_RETRIEVAL_METRICS:
            values = [
                step["retrieval"][reference][metric]
                for step in seed_result["steps"]
            ]
            out[f"{reference}.{metric}"] = mean(values)

        out[f"{reference}.candidate_shortfall_total_3points"] = int(
            sum(
                step["retrieval"][reference]["candidate_shortfall_total"]
                for step in seed_result["steps"]
            )
        )
        out[f"{reference}.candidate_shortfall_affected_user_observations_3points"] = int(
            sum(
                step["retrieval"][reference]["candidate_shortfall_affected_users"]
                for step in seed_result["steps"]
            )
        )

    for metric in [
        "mean_set_overlap_at_20",
        "mean_positional_agreement",
        "exact_ordered_list_match_fraction",
    ]:
        values = [
            step["retrieval"]["exact_geometry_shift"][metric]
            for step in seed_result["steps"]
        ]
        out[f"exact_geometry_shift.{metric}"] = mean(values)

    for branch in LATENCY_BRANCHES:
        for mode in ["raw", "filtered"]:
            for metric in ["median_batch_ms", "p95_batch_ms", "median_ms_per_user"]:
                values = [
                    step["latency"][branch][mode][metric]
                    for step in seed_result["steps"]
                ]
                out[f"latency.{branch}.{mode}.{metric}"] = mean(values)

    return out


def build_final_summary(seed_results, freeze):
    if len(seed_results) != 5:
        raise RuntimeError("H4 final requiere exactamente 5 seeds completas.")

    seed_summaries = [seed_level_summary(r) for r in seed_results]
    keys = [k for k in seed_summaries[0] if k != "seed"]

    across_seed = {}
    for key in keys:
        values = [row[key] for row in seed_summaries]
        across_seed[key] = {
            "mean": mean(values),
            "sample_std": sample_std(values),
            "median": float(statistics.median(values)),
            "seed_values": values,
        }

    temporal = {}
    for point in [1, 2, 3]:
        point_summary = {}

        for reference in REFERENCE_NAMES:
            for metric in CONTINUOUS_RETRIEVAL_METRICS:
                values = [
                    next(s for s in result["steps"] if s["point"] == point)
                    ["retrieval"][reference][metric]
                    for result in seed_results
                ]
                point_summary[f"{reference}.{metric}"] = {
                    "mean": mean(values),
                    "sample_std": sample_std(values),
                    "seed_values": values,
                }

        for metric in [
            "mean_set_overlap_at_20",
            "mean_positional_agreement",
            "exact_ordered_list_match_fraction",
        ]:
            values = [
                next(s for s in result["steps"] if s["point"] == point)
                ["retrieval"]["exact_geometry_shift"][metric]
                for result in seed_results
            ]
            point_summary[f"exact_geometry_shift.{metric}"] = {
                "mean": mean(values),
                "sample_std": sample_std(values),
                "seed_values": values,
            }

        for branch in LATENCY_BRANCHES:
            for mode in ["raw", "filtered"]:
                for metric in [
                    "median_batch_ms",
                    "p95_batch_ms",
                    "median_ms_per_user",
                ]:
                    values = [
                        next(s for s in result["steps"] if s["point"] == point)
                        ["latency"][branch][mode][metric]
                        for result in seed_results
                    ]
                    point_summary[f"latency.{branch}.{mode}.{metric}"] = {
                        "mean": mean(values),
                        "sample_std": sample_std(values),
                        "seed_values": values,
                    }

        for reference in REFERENCE_NAMES:
            for metric in [
                "candidate_shortfall_total",
                "candidate_shortfall_affected_users",
            ]:
                values = [
                    next(s for s in result["steps"] if s["point"] == point)
                    ["retrieval"][reference][metric]
                    for result in seed_results
                ]
                point_summary[f"{reference}.{metric}"] = {
                    "mean": mean(values),
                    "sample_std": sample_std(values),
                    "sum_across_5_seeds": int(sum(values)),
                    "seed_values": values,
                }

        build_values = [
            next(s for s in result["steps"] if s["point"] == point)
            ["build_time_ms"]["full_rebuilt"]
            for result in seed_results
        ]
        point_summary["build.full_rebuilt_index_ms"] = {
            "mean": mean(build_values),
            "sample_std": sample_std(build_values),
            "seed_values": build_values,
        }

        temporal[f"p{point}"] = point_summary

    all_steps = [
        step for result in seed_results for step in result["steps"]
    ]
    all_build_counts = {
        "base": sum(r["build_counts"]["base"] for r in seed_results),
        "online": sum(r["build_counts"]["online"] for r in seed_results),
        "full_stale": sum(r["build_counts"]["full_stale"] for r in seed_results),
        "full_rebuilt": sum(r["build_counts"]["full_rebuilt"] for r in seed_results),
        "total": sum(r["build_counts"]["total"] for r in seed_results),
    }
    expected_builds = freeze["protocol"]["expected_build_counts"]["all_5_seeds"]

    structural_15_15 = sum(
        1
        for s in all_steps
        if (
            s["structural"]["v_online_array_equal_base"]
            and s["structural"]["v_online_max_abs_diff_base"] == 0.0
            and s["structural"]["v_online_sha256"]
            == s["structural"]["v_base_sha256"]
            and s["structural"]["online_build_delta"] == 0
        )
    )
    integrity_15_15 = sum(
        1 for s in all_steps if s["base_index_fingerprint_unchanged"]
    )
    update_15_15 = sum(
        1
        for s in all_steps
        if s["u_update_evidence"]["n_current_update_users_with_delta_gt_0"] >= 1
    )

    shortfall_totals = {}
    for reference in REFERENCE_NAMES:
        shortfall_totals[reference] = {
            "candidate_shortfall_total_all_seed_points": int(
                sum(
                    step["retrieval"][reference]["candidate_shortfall_total"]
                    for step in all_steps
                )
            ),
            "candidate_shortfall_affected_user_observations_all_seed_points": int(
                sum(
                    step["retrieval"][reference]["candidate_shortfall_affected_users"]
                    for step in all_steps
                )
            ),
        }

    criteria = {
        "h4_pre_specified_descriptive_criterion_met": bool(
            structural_15_15 == 15 and all_build_counts["online"] == 0
        ),
        "structural_seed_point_passes": structural_15_15,
        "structural_seed_point_total": 15,
        "operational_integrity_guard_passes": integrity_15_15,
        "operational_integrity_guard_total": 15,
        "online_update_nontriviality_guard_passes": update_15_15,
        "online_update_nontriviality_guard_total": 15,
        "actual_build_counts": all_build_counts,
        "expected_build_counts": expected_builds,
        "build_counts_exact": bool(all_build_counts == expected_builds),
    }

    # Do not abort on a negative H4 result. Preserve and report it.
    if integrity_15_15 != 15:
        raise RuntimeError("Operational integrity guard no pasó 15/15.")
    if update_15_15 != 15:
        raise RuntimeError("Online update guard no pasó 15/15.")
    if all_build_counts != expected_builds:
        raise RuntimeError("Build accounting final no coincide con freeze.")

    return {
        "executor_version": EXECUTOR_VERSION,
        "executor_sha256": sha256_file(SCRIPT_PATH),
        "frozen_protocol_version": FROZEN_PROTOCOL_VERSION,
        "frozen_protocol_hash": FROZEN_PROTOCOL_HASH,
        "freeze_manifest_sha256": FREEZE_SHA256,
        "aggregation_rule": freeze["protocol"]["aggregation_plan"],
        "seed_level": seed_summaries,
        "across_seed": across_seed,
        "temporal": temporal,
        "shortfall_totals": shortfall_totals,
        "criteria": criteria,
        "interpretation_boundary": {
            "allowed": freeze["protocol"]["h4_criteria"]["allowed_interpretation"],
            "forbidden": freeze["protocol"]["h4_criteria"]["forbidden_interpretation"],
            "ann_quality": freeze["protocol"]["h4_criteria"]["ann_quality"],
            "latency_vs_h2": freeze["protocol"]["h4_metrics"]["latency"][
                "cross_hypothesis_restriction"
            ],
        },
    }


def validate_seed_file(path, seed, executor_sha, freeze, dataset_sha):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = {
        "status": "COMPLETE",
        "executor_version": EXECUTOR_VERSION,
        "frozen_protocol_version": FROZEN_PROTOCOL_VERSION,
        "frozen_protocol_hash": FROZEN_PROTOCOL_HASH,
        "freeze_manifest_sha256": FREEZE_SHA256,
        "seed": int(seed),
        "executor_sha256": executor_sha,
        "dataset_sha256": dataset_sha,
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint {path.name} incompatible en {key}: "
                f"{data.get(key)!r} != {expected!r}"
            )

    if data.get("build_counts") != freeze["protocol"]["expected_build_counts"]["per_seed"]:
        raise RuntimeError(f"Checkpoint {path.name}: build_counts inválidos.")

    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) != 3:
        raise RuntimeError(f"Checkpoint {path.name}: deben existir 3 steps.")

    for expected_point, step in enumerate(steps, start=1):
        if step.get("seed") != int(seed) or step.get("point") != expected_point:
            raise RuntimeError(
                f"Checkpoint {path.name}: seed/point inconsistente en step."
            )
        if not step.get("base_index_fingerprint_unchanged", False):
            raise RuntimeError(
                f"Checkpoint {path.name}: integrity guard falso en p{expected_point}."
            )
        if (
            step.get("u_update_evidence", {})
            .get("n_current_update_users_with_delta_gt_0", 0)
            < 1
        ):
            raise RuntimeError(
                f"Checkpoint {path.name}: update guard falso en p{expected_point}."
            )

    criteria = data.get("criteria", {})
    if criteria.get("operational_integrity_guard") is not True:
        raise RuntimeError(f"Checkpoint {path.name}: integrity criterion inválido.")
    if criteria.get("online_update_nontriviality_guard") is not True:
        raise RuntimeError(f"Checkpoint {path.name}: update criterion inválido.")

    # structural_success is intentionally NOT required to be True:
    # a negative H4 result must remain resumable/reportable.
    if not isinstance(criteria.get("structural_success"), bool):
        raise RuntimeError(f"Checkpoint {path.name}: structural_success inválido.")

    return data


def revalidate_all_anchored_checkpoints(
    campaign_path,
    prefix,
    seeds,
    executor_sha,
    freeze,
    dataset_sha,
):
    with open(campaign_path, "r", encoding="utf-8") as f:
        campaign = json.load(f)

    anchors = campaign.get("completed_seed_checkpoints", {})
    if not isinstance(anchors, dict):
        raise RuntimeError(
            "Campaign manifest no contiene completed_seed_checkpoints válido."
        )

    verified = {}
    for seed in seeds:
        seed_path = prefix.with_name(prefix.name + f"_seed{seed}.json")
        anchor = anchors.get(str(seed))

        if not isinstance(anchor, dict):
            raise RuntimeError(
                f"Seed {seed}: falta anchor de checkpoint antes del cierre final."
            )
        if not seed_path.exists():
            raise RuntimeError(
                f"Seed {seed}: checkpoint anclado no existe en disco."
            )
        if anchor.get("filename") != seed_path.name:
            raise RuntimeError(
                f"Seed {seed}: filename de checkpoint no coincide con su anchor."
            )

        observed_sha = sha256_file(seed_path)
        expected_sha = anchor.get("sha256")
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"Seed {seed}: checkpoint cambió antes del cierre final:\n"
                f"  observed={observed_sha}\n"
                f"  expected={expected_sha}"
            )

        validate_seed_file(
            seed_path,
            seed,
            executor_sha,
            freeze,
            dataset_sha,
        )
        verified[str(seed)] = {
            "filename": seed_path.name,
            "sha256": observed_sha,
            "semantic_validation": "PASS",
        }

    if len(verified) != len(seeds):
        raise RuntimeError(
            "No se pudieron verificar todos los checkpoints antes del cierre."
        )

    return verified


def persist_flat_outputs(prefix, seed_results, summary):
    step_rows = []
    query_rows = []
    trial_rows = []

    for result in seed_results:
        trial = {
            "seed": result["seed"],
            "base_train_time_s": result["base_train_time_s"],
            **{f"build_counts.{k}": v for k, v in result["build_counts"].items()},
            **{f"criteria.{k}": v for k, v in result["criteria"].items()},
        }
        trial.update({
            k: v
            for k, v in seed_level_summary(result).items()
            if k != "seed"
        })
        trial_rows.append(trial)

        for step in result["steps"]:
            step_copy = {k: v for k, v in step.items() if k != "query_records"}
            step_rows.append(flatten_dict(step_copy))
            for q in step["query_records"]:
                query_rows.append(flatten_dict(q))

    if step_rows:
        fields = sorted({k for r in step_rows for k in r})
        write_csv(prefix.with_name(prefix.name + "_steps.csv"), fields, step_rows)
    if query_rows:
        fields = sorted({k for r in query_rows for k in r})
        write_csv(prefix.with_name(prefix.name + "_queries.csv"), fields, query_rows)
    if trial_rows:
        fields = sorted({k for r in trial_rows for k in r})
        write_csv(prefix.with_name(prefix.name + "_trials.csv"), fields, trial_rows)

    atomic_json(prefix.with_name(prefix.name + "_summary.json"), summary)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Final H4 executor V1.3 bound to frozen H4 V1.4 protocol."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate freeze/data/code/environment/helpers. No final training.",
    )
    mode.add_argument(
        "--run-final",
        action="store_true",
        help="Run/resume the frozen final H4 campaign.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Campaign timestamp YYYYMMDD_HHMMSS. In --run-final, reuse the exact "
            "timestamp to resume completed seeds."
        ),
    )
    return parser.parse_args()


def validate_all(run_synthetic_ann_self_test):
    paths = resolve_paths()
    freeze = load_and_validate_freeze(paths["freeze"])

    env = runtime_environment()
    validate_environment(env, freeze)

    source_files, source_hashes = validate_sources(freeze)
    validate_wrapper_contracts()

    if run_synthetic_ann_self_test:
        self_test_report = self_test_helpers(freeze["protocol"]["h4_ann"])
    else:
        self_test_report = {
            "status": "SKIPPED_IN_RUN_FINAL",
            "reason": (
                "F2 isolation: no synthetic ANN build before the first real "
                "H4 base-index build. Synthetic executor self-test is exercised "
                "only by --validate-only."
            ),
        }

    data = prepare_protocol_data()
    validate_protocol_data(data, freeze)

    return {
        "paths": paths,
        "freeze": freeze,
        "env": env,
        "source_files": source_files,
        "source_hashes": source_hashes,
        "self_test_report": self_test_report,
        "data": data,
    }


def print_validation(context):
    freeze = context["freeze"]
    data = context["data"]
    print("=" * 118)
    print("FINAL H4 EXECUTOR V1.3 -- VALIDATION")
    print("=" * 118)
    print(f"Freeze SHA256   : {FREEZE_SHA256}")
    print(f"Protocol hash   : {FROZEN_PROTOCOL_HASH}")
    print(f"Executor SHA256 : {sha256_file(SCRIPT_PATH)}")
    print(
        "Environment     : "
        f"Python={context['env']['python']} Cornac={context['env']['cornac']} "
        f"NumPy={context['env']['numpy']} Torch={context['env']['torch']} "
        f"FAISS={context['env']['faiss']}"
    )
    print(
        "Threads         : "
        f"torch={context['env']['torch_num_threads']} "
        f"interop={context['env']['torch_num_interop_threads']} "
        f"faiss={context['env']['faiss_threads']}"
    )
    print(f"Data SHA256     : {data['data_sha256']}")
    print(
        f"Base/future     : {len(data['base_rows'])}/{len(data['future_rows'])}"
    )
    print(
        "Warm chunks     : "
        + ", ".join(str(len(c)) for c in data["known_chunks"])
    )
    print(
        "PRIMARY users   : "
        + ", ".join(str(x["primary_users"]) for x in data["primary_eval_plan"])
    )
    print(
        "ANN             : "
        f"{freeze['protocol']['h4_ann']['index_class']}/IP "
        f"nlist={freeze['protocol']['h4_ann']['nlist']} "
        f"nprobe={freeze['protocol']['h4_ann']['nprobe']} "
        f"seed={freeze['protocol']['h4_ann']['ann_clustering_seed']}"
    )
    print(f"ANN self-test   : {context['self_test_report']}")
    print("Validation      : PASS")


def run_final(context, timestamp):
    freeze = context["freeze"]
    data = context["data"]
    executor_sha = sha256_file(SCRIPT_PATH)

    if timestamp is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not re.fullmatch(r"\d{8}_\d{6}", timestamp):
        raise RuntimeError("--timestamp debe tener formato YYYYMMDD_HHMMSS.")

    prefix = RESULTS_DIR / f"final_h4_online_ibpr_mejorado_v1_3_{timestamp}"
    campaign_path = prefix.with_name(prefix.name + "_manifest.json")

    campaign_contract = {
        "status": "RUNNING",
        "timestamp": timestamp,
        "executor_version": EXECUTOR_VERSION,
        "executor_sha256": executor_sha,
        "freeze_manifest_filename": FREEZE_FILENAME,
        "freeze_manifest_sha256": FREEZE_SHA256,
        "frozen_protocol_version": FROZEN_PROTOCOL_VERSION,
        "frozen_protocol_hash": FROZEN_PROTOCOL_HASH,
        "dataset_sha256": data["data_sha256"],
        "runtime_environment_signature": environment_signature(context["env"]),
        "source_hashes": context["source_hashes"],
        "executor_validation_self_test": context["self_test_report"],
        "final_seeds": freeze["protocol"]["final_seeds"],
        "retuning_after_freeze": "PROHIBITED",
    }

    if campaign_path.exists():
        with open(campaign_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for key, expected in campaign_contract.items():
            if key == "status":
                continue
            if existing.get(key) != expected:
                raise RuntimeError(
                    f"Campaign manifest incompatible en {key}: "
                    f"{existing.get(key)!r} != {expected!r}"
                )
        if existing.get("status") == "COMPLETE":
            print(f"Campaign {timestamp} ya está COMPLETE.")
            return
    else:
        atomic_json(
            campaign_path,
            {
                **campaign_contract,
                "completed_seed_checkpoints": {},
            },
        )

    seed_results = []

    for seed in freeze["protocol"]["final_seeds"]:
        seed_path = prefix.with_name(prefix.name + f"_seed{seed}.json")

        if seed_path.exists():
            print(f"Resuming seed={seed}: validating completed checkpoint...")

            with open(campaign_path, "r", encoding="utf-8") as f:
                current_campaign = json.load(f)

            checkpoint_hashes = current_campaign.get(
                "completed_seed_checkpoints", {}
            )
            anchor = checkpoint_hashes.get(str(seed))
            if not isinstance(anchor, dict):
                raise RuntimeError(
                    f"Checkpoint seed={seed} existe pero no está anclado "
                    "criptográficamente en el campaign manifest. "
                    "No se reutiliza automáticamente."
                )

            observed_checkpoint_sha = sha256_file(seed_path)
            expected_checkpoint_sha = anchor.get("sha256")
            if observed_checkpoint_sha != expected_checkpoint_sha:
                raise RuntimeError(
                    f"Checkpoint seed={seed} cambió desde que fue anclado:\n"
                    f"  observed={observed_checkpoint_sha}\n"
                    f"  expected={expected_checkpoint_sha}"
                )

            if anchor.get("filename") != seed_path.name:
                raise RuntimeError(
                    f"Checkpoint seed={seed}: filename anclado incompatible."
                )

            result = validate_seed_file(
                seed_path, seed, executor_sha, freeze, data["data_sha256"]
            )
            seed_results.append(result)
            print(
                f"seed={seed}: checkpoint PASS, SHA anchored, skip."
            )
            continue

        result = run_seed(int(seed), data, freeze)
        result["executor_sha256"] = executor_sha
        result["dataset_sha256"] = data["data_sha256"]
        atomic_json(seed_path, result)

        # Re-open and validate the persisted checkpoint before advancing.
        result = validate_seed_file(
            seed_path, seed, executor_sha, freeze, data["data_sha256"]
        )

        checkpoint_sha = sha256_file(seed_path)
        with open(campaign_path, "r", encoding="utf-8") as f:
            current_campaign = json.load(f)

        checkpoint_hashes = dict(
            current_campaign.get("completed_seed_checkpoints", {})
        )
        checkpoint_hashes[str(seed)] = {
            "filename": seed_path.name,
            "sha256": checkpoint_sha,
        }
        current_campaign["completed_seed_checkpoints"] = checkpoint_hashes
        atomic_json(campaign_path, current_campaign)

        seed_results.append(result)
        print(
            f"seed={seed}: COMPLETE | checkpoint={seed_path.name} | "
            f"sha256={checkpoint_sha} | anchored=YES"
        )

    seed_results.sort(
        key=lambda x: freeze["protocol"]["final_seeds"].index(x["seed"])
    )

    final_provenance = assert_final_provenance_unchanged(
        context, executor_sha
    )

    final_checkpoint_verification = revalidate_all_anchored_checkpoints(
        campaign_path=campaign_path,
        prefix=prefix,
        seeds=freeze["protocol"]["final_seeds"],
        executor_sha=executor_sha,
        freeze=freeze,
        dataset_sha=data["data_sha256"],
    )

    summary = build_final_summary(seed_results, freeze)
    persist_flat_outputs(prefix, seed_results, summary)

    output_files = {}
    for suffix in ["_steps.csv", "_queries.csv", "_trials.csv", "_summary.json"]:
        path = prefix.with_name(prefix.name + suffix)
        output_files[path.name] = sha256_file(path)

    for seed in freeze["protocol"]["final_seeds"]:
        path = prefix.with_name(prefix.name + f"_seed{seed}.json")
        output_files[path.name] = sha256_file(path)

    with open(campaign_path, "r", encoding="utf-8") as f:
        current_campaign = json.load(f)

    h4_conclusion = (
        "H4_CRITERION_MET"
        if summary["criteria"]["h4_pre_specified_descriptive_criterion_met"]
        else "H4_CRITERION_NOT_MET"
    )

    final_campaign = {
        **campaign_contract,
        "status": "COMPLETE",
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "h4_conclusion": h4_conclusion,
        "completed_seed_checkpoints": current_campaign.get(
            "completed_seed_checkpoints", {}
        ),
        "final_provenance_unchanged": final_provenance,
        "final_checkpoint_verification": final_checkpoint_verification,
        "criteria": summary["criteria"],
        "output_files_sha256": output_files,
    }
    atomic_json(campaign_path, final_campaign)

    print()
    print("=" * 118)
    print("FINAL H4 CAMPAIGN: COMPLETE")
    print("=" * 118)
    print(f"Timestamp       : {timestamp}")
    print(f"Protocol hash   : {FROZEN_PROTOCOL_HASH}")
    print(f"Executor SHA256 : {executor_sha}")
    print(
        "H4 criterion    : "
        f"{summary['criteria']['structural_seed_point_passes']}/15 structural "
        f"| online rebuilds={summary['criteria']['actual_build_counts']['online']}"
    )
    print(
        "Integrity guard : "
        f"{summary['criteria']['operational_integrity_guard_passes']}/15"
    )
    print(
        "U update guard  : "
        f"{summary['criteria']['online_update_nontriviality_guard_passes']}/15"
    )
    print(
        "Build counts    : "
        f"{summary['criteria']['actual_build_counts']}"
    )
    print(f"Manifest        : {campaign_path}")
    print(f"Manifest SHA256 : {sha256_file(campaign_path)}")

    if summary["criteria"]["h4_pre_specified_descriptive_criterion_met"]:
        interpretation = (
            "H4 criterion met: functional ANN index reuse was observed in "
            "the evaluated MovieLens 1M scenario only."
        )
    else:
        interpretation = (
            "H4 criterion NOT met: the pre-specified structural reuse "
            "criterion failed in at least one seed/point. No positive H4 "
            "claim is permitted."
        )

    print(f"Interpretation  : {interpretation}")


def main():
    args = parse_args()

    # Validation occurs before any final model fit/index build.
    context = validate_all(
        run_synthetic_ann_self_test=bool(args.validate_only)
    )
    print_validation(context)

    if args.validate_only:
        print()
        print("VALIDATE-ONLY COMPLETE: no final H4 model/index/outcome was generated.")
        return

    run_final(context, args.timestamp)


if __name__ == "__main__":
    main()
