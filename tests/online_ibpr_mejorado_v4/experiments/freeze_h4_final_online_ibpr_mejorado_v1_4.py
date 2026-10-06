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
import datetime as dt
import hashlib
import importlib
import inspect
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
import cornac

try:
    import faiss
except ImportError as exc:
    raise RuntimeError(
        "FAISS no está disponible. El freeze H4 requiere faiss funcional."
    ) from exc

from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.models.recommender import ANNMixin, MEASURE_DOT


# =============================================================================
# H4 FINAL FREEZE V1.4 -- PLAN ONLY
# -----------------------------------------------------------------------------
# This artifact freezes H4 BEFORE any final H4 outcome is generated.
#
# It DOES NOT:
#   * train H4 final models;
#   * call partial_fit_recent() on MovieLens final data;
#   * build final MovieLens ANN indices;
#   * evaluate final H4 retrieval outcomes.
#
# It DOES:
#   * validate the exact H1-H3 parent manifest structurally;
#   * bind the freeze to the actual current source code and runtime environment;
#   * validate source continuity for the training cores;
#   * freeze model chronology / seed policies;
#   * freeze exact-vs-ANN filtering and adaptive retrieval policy;
#   * freeze branch definitions, metrics, build accounting, and interpretation;
#   * require a persisted, hashable PASS log from synthetic preflight V1.5;
#   * decode that log robustly across UTF-8/UTF-16 PowerShell redirection;
#   * recompute parent protocol/realized-plan hashes instead of trusting fields;
#   * freeze non-trivial Online U-update evidence;
#   * harden base-index immutability as an operational-integrity guard;
#   * preserve the ORIGINAL pre-specified H4 descriptive criterion;
#   * freeze seed-level aggregation for repeated temporal points;
#   * freeze explicit retrieval/geometry metric definitions;
#   * require native numerical threadpools to be single-threaded;
#   * require ALL four model source files (cores + wrappers) to remain byte-identical
#     to the FINAL H1-H3 source hashes, while separately validating ANN semantics.
# =============================================================================

PROTOCOL_VERSION = "h4_final_v1_4_freeze_20261002"

# -----------------------------------------------------------------------------
# Frozen H1-H3 parent identity.
# -----------------------------------------------------------------------------
H1H3_TIMESTAMP = "20261001_145409"
H1H3_PROTOCOL_VERSION = "final_h1_h3_v3_3_freeze_20261001"
H1H3_PROTOCOL_HASH = (
    "47a0f4bc499097669647316a455abe81563d7c09fccf26cc916ef1cfc8f58e7b"
)
H1H3_SCRIPT_SHA256 = (
    "0684e5d5f91e494dd497d9a243f4de2aac5f6b267b5c1120869c51815c3fdd5b"
)
H1H3_REALIZED_PLAN_SHA256 = (
    "cbf76393b6241771c9d91c18418b444abd7f4c8a977f7f8b1864dabba4131ec2"
)

DATA_SHA256 = (
    "29da5346c5bcf37dc927771d8ffd7ec3323dc7857ed4b0f6a45278b666954d3e"
)
C1_DATA_SHA256 = (
    "3112900cf5f4966e117bbc0ccfb33cb2e0128ac8b36c2b79a154019651d1ba42"
)

C1_PARENT = {
    "timestamp": "20260929_231439",
    "candidate": "R900",
    "protocol_hash": (
        "7adec0037dba1cd31c9100942d5fa9dfb0aa00dbb9a5bf8dc7b8ff4b51154112"
    ),
}
C2_PARENT = {
    "timestamp": "20260930_225103",
    "candidate": "O019",
    "protocol_hash": (
        "61215894da0d3cadf5966b3cc503558884955ee6314b01e6f95a914e6f0b9621"
    ),
}

FINAL_SEEDS = [777, 999, 2026, 31415, 271828]

R900 = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-5,
    "batch_size": 512,
}

O019 = {
    "learning_rate": 0.0025,
    "lamda": 1e-6,
    "batch_size": 256,
    "n_epochs": 3,
    "loss_mode": "angular",
    "update_V": False,
    "neg_sampling": "uniform",
    "normalize": True,
    "max_steps": None,
}

# Top-level source hashes recorded by the FINAL H1-H3 manifest.
# The training cores are required to remain exact for H4.
PARENT_SOURCE_HASHES = {
    "ibpr_core": (
        "e33e61a20862e8dc3d6bb3270998ba75557b8eefd48a52dcaac63271b6b38059"
    ),
    "online_core": (
        "e27630a4b7fc4b819a8a4ef1757eea7d1cb21f8bc25efc028a0954964b16b801"
    ),
    "ibpr_wrapper": (
        "ec525d4dfc21dfc9d0389033cddbfdd600aed25122ff0e78630e9c08fab6fb1e"
    ),
    "online_wrapper": (
        "3650a9a035e5baac492263af71c4572198375d97a7bb178a453281d3e01e8450"
    ),
}

# -----------------------------------------------------------------------------
# Exact temporal plan inherited from the frozen H1-H3 realized plan.
# -----------------------------------------------------------------------------
DATASET_PLAN = {
    "dataset": "MovieLens 1M",
    "variant": "1M",
    "positive_threshold": 3.0,
    "implicit_positive": 1.0,
    "total_implicit_positive_rows": 836478,
    "target_base_fraction": 0.60,
    "target_base_rows": 501886,
    "effective_base_rows": 501887,
    "effective_future_rows": 334591,
    "base_users": 4037,
    "base_items": 3505,
    "future_warm_rows": 52467,
    "future_warm_fraction": 0.15680935829116743,
    "base_timestamp_range": [956703932, 974678223],
    "future_timestamp_range": [974678227, 1046454590],
    "warm_chunk_boundaries": [0, 13118, 26236, 39350, 52467],
    "chunks": [
        {
            "chunk": 1,
            "warm_rows": 13118,
            "users": 363,
            "items": 2458,
            "ts_min": 974678227,
            "ts_max": 977183160,
        },
        {
            "chunk": 2,
            "warm_rows": 13118,
            "users": 463,
            "items": 2408,
            "ts_min": 977183184,
            "ts_max": 986263783,
        },
        {
            "chunk": 3,
            "warm_rows": 13114,
            "users": 438,
            "items": 2372,
            "ts_min": 986263809,
            "ts_max": 1007365896,
        },
        {
            "chunk": 4,
            "warm_rows": 13117,
            "users": 376,
            "items": 2424,
            "ts_min": 1007365934,
            "ts_max": 1046454590,
        },
    ],
    "eval_points": [
        {
            "point": 1,
            "update_chunk": 1,
            "eval_chunk": 2,
            "users_exposed_to_update": 363,
            "primary_rows": 5300,
            "primary_users": 212,
            "primary_items": 1957,
            "all_warm_rows": 13118,
            "all_warm_users": 463,
            "unadapted_eval_users": 251,
        },
        {
            "point": 2,
            "update_chunk": 2,
            "eval_chunk": 3,
            "users_exposed_to_update": 614,
            "primary_rows": 8546,
            "primary_users": 315,
            "primary_items": 2161,
            "all_warm_rows": 13114,
            "all_warm_users": 438,
            "unadapted_eval_users": 123,
        },
        {
            "point": 3,
            "update_chunk": 3,
            "eval_chunk": 4,
            "users_exposed_to_update": 737,
            "primary_rows": 9143,
            "primary_users": 306,
            "primary_items": 2193,
            "all_warm_rows": 13117,
            "all_warm_users": 376,
            "unadapted_eval_users": 70,
        },
    ],
}

# -----------------------------------------------------------------------------
# Model chronology and seed policies.
# -----------------------------------------------------------------------------
MODEL_CHRONOLOGY = {
    "base": {
        "definition": (
            "For each final trial seed, fit a fresh IBPR R900 on the effective "
            "60% base only."
        ),
        "seed_policy": (
            "Reset NumPy/Torch/model seed to the final trial seed before base fit."
        ),
    },
    "online": {
        "initialization": (
            "Warm-start OnlineIBPRMejorado from exact copies of that seed's "
            "U_base and V_base."
        ),
        "sequence": [
            "update warm chunk 1 -> H4 point 1",
            "update warm chunk 2 -> H4 point 2",
            "update warm chunk 3 -> H4 point 3",
        ],
        "history_policy": (
            "At point t, partial_fit_recent receives history_csr equal to "
            "base positives plus accumulated WARM chunks 1..t, including the "
            "current update chunk and excluding the future evaluation chunk."
        ),
        "online_seed_policy": "trial_seed + partial_update_count",
        "partial_update_count": "0 before point1, then 1,2,3 after non-empty updates",
    },
    "full": {
        "definition": (
            "At each point t, fit a FRESH IBPR R900 from scratch on base plus "
            "accumulated WARM chunks 1..t."
        ),
        "seed_policy": (
            "Reset to the SAME final trial seed before each Full fit, exactly "
            "as frozen in H1-H3."
        ),
    },
    "item_mapping": (
        "The base user/item mapping is fixed for all branches. Only future rows "
        "whose user AND item exist in the effective base enter the WARM universe. "
        "All U/V comparisons and ANN item ids must use that exact internal item order."
    ),
}

# -----------------------------------------------------------------------------
# ANN configuration.
# ANN clustering seed is fixed across model seeds/branches to isolate geometry.
# -----------------------------------------------------------------------------
H4_ANN = {
    "backend": "faiss",
    "index_class": "IndexIVFFlat",
    "metric": "inner_product",
    "ann_clustering_seed": 42,
    "top_k": 20,
    "nlist": 16,
    "nprobe": 8,
    "initial_search_k": 80,
    "raw_faiss_k": 80,
    "adaptive_growth": "double search_k until K valid items or ntotal",
    "latency_warmups": 10,
    "latency_repeats": 50,
    "threads": 1,
}

# -----------------------------------------------------------------------------
# Exact/ANN retrieval policy -- frozen BEFORE final H4 results.
# -----------------------------------------------------------------------------
RETRIEVAL_POLICY = {
    "score": (
        "Use the model factors exactly as stored; score is dot product U @ V.T. "
        "No extra query-time normalization is applied."
    ),
    "exact_reference": (
        "Exhaustively score every base-known item and sort descending by score; "
        "break exact-score ties deterministically by ascending internal item id."
    ),
    "observed_exclusion": (
        "At H4 point t, both exact and ANN exclude exactly the user's observed "
        "positive items in base plus accumulated WARM update chunks 1..t. "
        "Items from evaluation chunk t+1 are NOT excluded unless they were "
        "already present in that prior accumulated history."
    ),
    "ann_candidate_policy": (
        "Start with search_k=min(ntotal,80). Remove negative ids, duplicates, and "
        "observed items. If fewer than K=20 valid items remain, double search_k "
        "and query again, capped at ntotal."
    ),
    "shortfall": (
        "shortfall=max(0,K-number_of_valid_items_after_searching_up_to_ntotal). "
        "Report it; do not silently pad it with observed or invalid items."
    ),
    "population_primary": (
        "At each point use exactly the PRIMARY H1-H3 evaluation users: eval users "
        "with prior warm-update opportunity. PRIMARY rows define membership; "
        "ALL-WARM is supplementary only."
    ),
    "population_supplementary": (
        "ALL-WARM retrieval may be reported separately but cannot replace PRIMARY."
    ),
}

RETRIEVAL_METRIC_DEFINITIONS = {
    "set_recall_at_20": (
        "For each PRIMARY user, let E be the filtered exhaustive top-20 list and "
        "A the filtered ANN output truncated to at most 20 valid unique items. "
        "Compute |set(A) intersection set(E)| / |E|. Do not pad ANN shortfalls. "
        "Average this user-level value within each seed x point."
    ),
    "positional_agreement": (
        "For each PRIMARY user, compare aligned ranks j=1..|E|. Agreement is the "
        "fraction of ranks where ANN[j] exists and ANN[j] == E[j]. Missing ANN "
        "positions caused by shortfall count as disagreement. Average within point."
    ),
    "exact_ordered_list_match": (
        "Per user binary indicator equal to 1 only when ANN returns the same number "
        "of items as E and the complete ordered ANN list equals E exactly. Report "
        "the fraction of users equal to 1 within seed x point."
    ),
    "candidate_shortfall": (
        "Per user max(0, |E|-|A|) after adaptive ANN retrieval up to ntotal. "
        "Report both total shortfall and number/fraction of affected PRIMARY users."
    ),
    "full_stale_model_fidelity": (
        "For full_stale, compare ANN(U_full,index[V_base]) directly against the "
        "model-faithful exhaustive exact(U_full,V_full) using the same set_recall@20, "
        "positional-agreement, exact-list, and shortfall definitions."
    ),
    "exact_geometry_shift": (
        "For each PRIMARY user compare exhaustive top-20 from exact(U_full,V_base) "
        "against exhaustive top-20 from exact(U_full,V_full). Primary geometry "
        "diagnostic is symmetric set_overlap@20 = |A intersection B| / 20. Also "
        "report positional agreement and exact ordered-list match fraction. "
        "This is descriptive and is not an additive decomposition."
    ),
}

AGGREGATION_PLAN = {
    "experimental_unit": (
        "The final trial seed is the independent replication unit. The three "
        "temporal H4 points within a seed are repeated measures, not independent "
        "replicates."
    ),
    "point_level": (
        "For every seed x point, compute retrieval metrics by first computing the "
        "user-level quantity on that point's PRIMARY users and then taking the "
        "unweighted mean across those PRIMARY users, except shortfall counts which "
        "are additionally reported as counts/fractions."
    ),
    "seed_level": (
        "For each seed, compute the unweighted arithmetic mean of its three "
        "point-level values for each continuous retrieval diagnostic. Do not weight "
        "points by number of PRIMARY rows/users."
    ),
    "across_seed_summary": (
        "The principal aggregate descriptive summary uses the five seed-level "
        "replicates: report mean and sample standard deviation across the 5 seeds. "
        "Median may be reported additionally, but cannot replace the frozen mean/std."
    ),
    "temporal_summary": (
        "Also report p1, p2, and p3 separately by summarizing the five seed-specific "
        "values at that point. This preserves temporal structure."
    ),
    "no_pseudoreplication": (
        "The 15 seed x point observations MUST NOT be treated as 15 independent "
        "replicates for inferential or uncertainty summaries."
    ),
    "structural_h4": (
        "The pre-specified H4 structural criterion is evaluated directly over all "
        "15 seed x point checks because it is an invariant requirement, not a "
        "statistical replication count."
    ),
    "latency": (
        "Latency remains descriptive at seed x point and temporal summaries; no "
        "cross-H2 pooling or formal significance test is introduced."
    ),
}

U_UPDATE_EVIDENCE = {
    "scope": (
        "At each H4 point, compare U immediately before vs immediately after "
        "partial_fit_recent for users present in the CURRENT warm update chunk."
    ),
    "row_delta": (
        "For each current-update user u, compute L2 norm "
        "||U_after[u]-U_before[u]||_2."
    ),
    "reported": [
        "n_current_update_users",
        "n_current_update_users_with_delta_gt_0",
        "fraction_current_update_users_with_delta_gt_0",
        "median_current_update_user_delta_l2",
        "max_current_update_user_delta_l2",
    ],
    "nontriviality_guard": (
        "At every one of the 15 seed/point observations, at least one user in "
        "the current update chunk must have delta_l2 > 0. No arbitrary positive "
        "magnitude threshold beyond strict > 0 is introduced."
    ),
    "interpretation": (
        "This guard only establishes that the Online update was non-trivial. "
        "It is not an effectiveness criterion and does not replace H1."
    ),
}

BUILD_ACCOUNTING = {
    "single_builder_rule": (
        "The final H4 executor must route every IVF construction used by the "
        "three H4 retrieval scenarios through one instrumented H4 index-builder "
        "helper. Direct scenario-level FAISS IVF construction outside that helper "
        "is prohibited."
    ),
    "scope": (
        "Build counts refer to all IVF builds in the H4 retrieval pipeline: "
        "one base build per seed, zero Online rebuilds, zero full_stale rebuilds, "
        "and one full_rebuilt build at each of three points per seed."
    ),
    "claim_boundary": (
        "The claim is zero Online rebuilds in the instrumented H4 retrieval "
        "pipeline, not interception of unrelated FAISS objects elsewhere in the process."
    ),
}

H4_BRANCHES = {
    "online_reused": (
        "U_online(t) + the SAME IVF index built once over V_base before the first "
        "online update; no online rebuild is allowed."
    ),
    "full_stale": (
        "U_full(t) + the unreconstructed base IVF index over V_base."
    ),
    "full_rebuilt": (
        "U_full(t) + a newly built IVF index over V_full(t)."
    ),
}

H4_REFERENCES = {
    "online_reused": "ANN(U_online,index[V_base]) vs exact(U_online,V_base)",
    "full_stale_index_consistency": (
        "ANN(U_full,index[V_base]) vs exact(U_full,V_base)"
    ),
    "full_stale_model_fidelity": (
        "ANN(U_full,index[V_base]) vs exact(U_full,V_full)"
    ),
    "full_rebuilt": "ANN(U_full,index[V_full]) vs exact(U_full,V_full)",
    "exact_geometry_shift": "exact(U_full,V_base) vs exact(U_full,V_full)",
}

H4_METRICS = {
    "primary_retrieval": "mean set recall@20 ANN vs exhaustive on PRIMARY users",
    "additional_retrieval": [
        "mean positional agreement",
        "fraction of exact ordered-list matches",
        "candidate shortfall total",
    ],
    "latency": {
        "raw": (
            "Same PRIMARY query batch and raw_faiss_k=80 within each point/branch; "
            "10 warmups + 50 repetitions; report median and p95 batch ms."
        ),
        "filtered": (
            "Same PRIMARY query batch through adaptive retrieval + filtering; "
            "10 warmups + 50 repetitions; report median and p95 batch ms."
        ),
        "derived": (
            "Also report median batch-ms / number of queried PRIMARY users as "
            "descriptive ms-per-user normalization."
        ),
        "build": "Report IVF build time separately from query latency.",
        "cross_hypothesis_restriction": (
            "H4 latency belongs to the H4 runtime environment and MUST NOT be "
            "numerically merged with or interpreted as H2 timing."
        ),
    },
    "structural": [
        "V_online == V_base by np.array_equal",
        "max_abs_diff(V_online,V_base) == 0",
        "SHA256(V_online) == SHA256(V_base)",
        "base index object identity recorded within process",
        "serialized SHA256(index_base_after) == SHA256(index_base_before)",
        "base index ntotal unchanged",
        "base index dimension d unchanged",
        "base index metric_type unchanged",
        "base index nlist unchanged",
        "base index nprobe unchanged",
        "actual online build delta == 0",
        "base/item mapping equality across branches",
        "non-trivial U-update evidence on current-update users",
    ],
}

H4_CRITERIA = {
    "pre_specified_descriptive_criterion": (
        "H4 is consistent only when V_base == V_online bit-for-bit at all "
        "5 seeds x 3 points (15/15) AND the Online branch performs zero rebuilds "
        "of the base index. This is the original frozen criterion from the roadmap; "
        "V1.3 does not redefine it."
    ),
    "operational_integrity_guard": (
        "Independently of the H4 descriptive criterion, a valid execution must keep "
        "the reused base-index fingerprint unchanged at every Online point: "
        "serialized SHA256, ntotal, d, metric_type, nlist, and nprobe must equal "
        "their pre-update base values. Failure invalidates the H4 execution and "
        "requires investigation; it does not create a new H4 success criterion."
    ),
    "online_update_nontriviality_guard": (
        "A valid execution must show at least one current-update user with "
        "||U_after-U_before||_2 > 0 at every one of the 15 seed/point observations. "
        "This demonstrates the Online update was not a no-op; it is not an "
        "effectiveness criterion and does not replace H1."
    ),
    "base_index_object_identity_role": (
        "Object identity is recorded as within-process forensic evidence only. "
        "Serialized hash and structural metadata provide the mandatory integrity guard."
    ),
    "ann_quality": (
        "ANN retrieval fidelity is reported descriptively. No post-hoc recall "
        "threshold is introduced as an H4 success criterion."
    ),
    "shortfall_role": "Report descriptively and investigate any non-zero value.",
    "allowed_interpretation": (
        "Functional reuse of the item ANN index in the evaluated MovieLens 1M scenario."
    ),
    "forbidden_interpretation": (
        "No general industrial-scale ANN scalability claim."
    ),
    "retuning": (
        "R900, O019, temporal split, populations, ANN parameters, retrieval policy, "
        "and success criterion are frozen. No retuning after this manifest."
    ),
}

EXPECTED_BUILD_COUNTS = {
    "per_seed": {
        "base": 1,
        "online": 0,
        "full_stale": 0,
        "full_rebuilt": 3,
        "total": 4,
    },
    "all_5_seeds": {
        "base": 5,
        "online": 0,
        "full_stale": 0,
        "full_rebuilt": 15,
        "total": 20,
    },
}

# Environment known from the PASSED V1.5 preflight terminal output.
EXPECTED_PREFLIGHT_ENV = {
    "python": "3.12.0",
    "cornac": "2.3.5",
    "numpy": "2.4.2",
    "torch": "2.10.0+cpu",
    "faiss": "1.15.0",
}

PREFLIGHT_FILENAME = "preflight_h4_ann_reuse_online_ibpr_mejorado_v1_5.py"
PREFLIGHT_SHA256 = (
    "38b8da0d5e9a53e0c10933b4b7af1de503a163b01cd557f90824d15bdad9f866"
)
DEFAULT_PREFLIGHT_LOG = "preflight_h4_ann_reuse_online_ibpr_mejorado_v1_5_PASS.txt"

DEFAULT_PARENT_MANIFEST = (
    "final_h1_h3_online_ibpr_mejorado_v3_3_manifest_"
    + H1H3_TIMESTAMP
    + ".json"
)


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
        "python_full": sys.version,
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
        env["native_threadpools_error"] = repr(exc)

    return env


def environment_signature(env):
    """
    Stable runtime signature bound into the protocol hash. Absolute library paths
    from threadpool_info remain provenance-only and are intentionally excluded.
    """
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


def assert_expected_preflight_environment(env):
    mismatches = []

    for key, expected in EXPECTED_PREFLIGHT_ENV.items():
        observed = env.get(key)
        if observed != expected:
            mismatches.append(
                f"{key}: observed={observed!r}, expected={expected!r}"
            )

    for name, value in env["thread_env"].items():
        if value != "1":
            mismatches.append(
                f"{name}: observed={value!r}, expected='1'"
            )

    if env["torch_num_threads"] != 1:
        mismatches.append(
            f"torch_num_threads={env['torch_num_threads']} expected=1"
        )

    if env["torch_num_interop_threads"] != 1:
        mismatches.append(
            f"torch_num_interop_threads={env['torch_num_interop_threads']} expected=1"
        )

    if env["faiss_threads"] not in (None, 1):
        mismatches.append(
            f"faiss_threads={env['faiss_threads']} expected=1"
        )

    for pool in env.get("native_threadpools", []):
        n_threads = pool.get("num_threads")
        if n_threads is not None and int(n_threads) != 1:
            mismatches.append(
                "native_threadpool "
                f"{pool.get('user_api')}/{pool.get('internal_api')} "
                f"prefix={pool.get('prefix')} num_threads={n_threads} expected=1"
            )

    if mismatches:
        raise RuntimeError(
            "El entorno no coincide con el entorno que pasó preflight H4 V1.5:\n  - "
            + "\n  - ".join(mismatches)
        )


def resolve_repo_paths():
    script_path = Path(__file__).resolve()
    experiments_dir = script_path.parent
    results_dir = experiments_dir / "results"

    if experiments_dir.name != "experiments":
        raise RuntimeError(
            "El script debe permanecer dentro de "
            "tests/online_ibpr_mejorado_v4/experiments."
        )

    repo_root = experiments_dir.parents[2]

    return {
        "script": script_path,
        "experiments_dir": experiments_dir,
        "results_dir": results_dir,
        "repo_root": repo_root,
    }


def discover_model_source_files():
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

    core_specs = [
        ("ibpr_core", ibpr_wrapper_module, "ibpr.py"),
        ("online_core", online_wrapper_module, "online_ibpr_mejorado.py"),
    ]

    for label, wrapper_module, sibling_name in core_specs:
        wrapper_source = inspect.getsourcefile(wrapper_module)
        if wrapper_source is None:
            raise RuntimeError(f"No se pudo resolver módulo wrapper para {label}.")

        candidate = Path(wrapper_source).resolve().with_name(sibling_name)
        if not candidate.exists():
            raise RuntimeError(
                f"No se encontró source core esperado para {label}: {candidate}"
            )
        files[label] = candidate

    return files


def source_method_sha256(method):
    source = inspect.getsource(method).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def assert_wrapper_semantics():
    """
    Guards the H4-relevant wrapper behavior independently of byte identity.
    V1.4 separately requires both wrappers to be byte-identical to FINAL H1-H3.
    """
    ibpr_fit = inspect.getsource(IBPR.fit)
    online_partial = inspect.getsource(OnlineIBPRMejorado.partial_fit_recent)

    required_ibpr_tokens = [
        "from .ibpr import ibpr",
        "n_epochs=self.max_iter",
        "lamda=self.lamda",
        "learning_rate=self.learning_rate",
        "batch_size=self.batch_size",
        'init_params={"U": self.U, "V": self.V}',
    ]

    required_online_tokens = [
        "from .online_ibpr_mejorado import online_ibpr_mejorado",
        "recent_pairs=recent_pairs",
        "history_csr=history_csr",
        "update_V=self.update_V",
        "normalize=self.normalize",
        "loss_mode=self.loss_mode",
        "random_seed=self.seed + self._partial_update_count",
        "self._partial_update_count += 1",
    ]

    missing = []

    for token in required_ibpr_tokens:
        if token not in ibpr_fit:
            missing.append(f"IBPR.fit missing token: {token}")

    for token in required_online_tokens:
        if token not in online_partial:
            missing.append(f"Online.partial_fit_recent missing token: {token}")

    if missing:
        raise RuntimeError(
            "El wrapper actual no satisface invariantes semánticos esperados:\n  - "
            + "\n  - ".join(missing)
        )

    return {
        "ibpr_fit_method_sha256": source_method_sha256(IBPR.fit),
        "online_partial_fit_recent_method_sha256": source_method_sha256(
            OnlineIBPRMejorado.partial_fit_recent
        ),
        "semantic_contract_pass": True,
    }


def assert_ann_contracts():
    if not issubclass(IBPR, ANNMixin):
        raise RuntimeError("IBPR no hereda ANNMixin.")

    if not issubclass(OnlineIBPRMejorado, ANNMixin):
        raise RuntimeError("OnlineIBPRMejorado no hereda ANNMixin.")

    U = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    V = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

    ibpr = IBPR(
        k=2,
        trainable=False,
        init_params={"U": U.copy(), "V": V.copy()},
        name="h4_freeze_ibpr_contract",
    )
    ibpr.num_users = U.shape[0]
    ibpr.num_items = V.shape[0]

    online = OnlineIBPRMejorado(
        k=2,
        trainable=False,
        init_params={"U": U.copy(), "V": V.copy()},
        update_V=False,
        normalize=True,
        name="h4_freeze_online_contract",
    )
    online.num_users = U.shape[0]
    online.num_items = V.shape[0]

    for name, model in [("IBPR", ibpr), ("OnlineIBPRMejorado", online)]:
        if model.get_vector_measure() != MEASURE_DOT:
            raise RuntimeError(f"{name} no declara MEASURE_DOT.")
        if not np.array_equal(model.get_user_vectors(), U):
            raise RuntimeError(f"{name}.get_user_vectors() no coincide.")
        if not np.array_equal(model.get_item_vectors(), V):
            raise RuntimeError(f"{name}.get_item_vectors() no coincide.")

        expected = V.dot(U[0])
        observed = np.asarray(model.score(0))

        if not np.allclose(observed, expected, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"{name}.score() no coincide con V @ U.")

    return {
        "IBPR_is_ANNMixin": True,
        "OnlineIBPRMejorado_is_ANNMixin": True,
        "measure": "MEASURE_DOT",
        "score_contract": "V @ U[user]",
    }


def assert_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(
            f"Manifest H1-H3 inconsistente en {label}: "
            f"observed={actual!r}, expected={expected!r}"
        )


def normalize_parent_primary_plan(parent):
    plan = parent["realized_plan"]["primary_eval_plan"]
    return [
        {
            "point": row["eval_point"],
            "update_chunk": row["update_chunk"],
            "eval_chunk": row["eval_chunk"],
            "users_exposed_to_update": row["n_users_exposed_to_update"],
            "primary_rows": row["n_primary_rows"],
            "primary_users": row["n_primary_users"],
            "primary_items": row["n_primary_items"],
            "all_warm_rows": row["n_allwarm_rows"],
            "all_warm_users": row["n_allwarm_users"],
            "unadapted_eval_users": row["n_allwarm_unadapted_users"],
        }
        for row in plan
    ]


def normalize_parent_chunks(parent):
    return [
        {
            "chunk": row["chunk"],
            "warm_rows": row["rows"],
            "users": row["users"],
            "items": row["items"],
            "ts_min": row["min_timestamp"],
            "ts_max": row["max_timestamp"],
        }
        for row in parent["realized_plan"]["warm_chunks"]
    ]


def validate_parent_manifest(path):
    if not path.exists():
        raise RuntimeError(
            "No se encontró el manifest H1-H3 congelado:\n"
            f"  {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        parent = json.load(f)

    protocol_payload_parent = parent.get("protocol_payload")
    if not isinstance(protocol_payload_parent, dict):
        raise RuntimeError(
            "Manifest H1-H3 no contiene protocol_payload válido."
        )

    realized_parent = parent.get("realized_plan")
    if not isinstance(realized_parent, dict):
        raise RuntimeError(
            "Manifest H1-H3 no contiene realized_plan válido."
        )

    recomputed_parent_protocol_hash = canonical_sha256(protocol_payload_parent)
    recomputed_parent_realized_hash = canonical_sha256(realized_parent)

    # Exact top-level identity/config assertions, including recomputed hashes.
    assert_equal(
        parent.get("protocol_version"),
        H1H3_PROTOCOL_VERSION,
        "protocol_version",
    )
    assert_equal(
        parent.get("protocol_hash"),
        H1H3_PROTOCOL_HASH,
        "protocol_hash",
    )
    assert_equal(
        recomputed_parent_protocol_hash,
        H1H3_PROTOCOL_HASH,
        "recomputed protocol_payload SHA256",
    )
    assert_equal(
        parent.get("dataset_sha256"),
        DATA_SHA256,
        "dataset_sha256",
    )
    assert_equal(
        parent.get("c1_compatible_dataset_sha256"),
        C1_DATA_SHA256,
        "c1_compatible_dataset_sha256",
    )
    assert_equal(
        parent.get("final_seeds"),
        FINAL_SEEDS,
        "final_seeds",
    )
    assert_equal(
        parent.get("frozen_ibpr_config"),
        R900,
        "frozen_ibpr_config",
    )
    assert_equal(
        parent.get("frozen_online_config"),
        O019,
        "frozen_online_config",
    )
    assert_equal(
        parent.get("script_sha256"),
        H1H3_SCRIPT_SHA256,
        "script_sha256",
    )
    assert_equal(
        parent.get("realized_plan_sha256"),
        H1H3_REALIZED_PLAN_SHA256,
        "realized_plan_sha256",
    )

    # Parent chain.
    assert_equal(
        parent.get("parent_c1_timestamp"),
        C1_PARENT["timestamp"],
        "parent_c1_timestamp",
    )
    assert_equal(
        parent.get("parent_c1_config_id"),
        C1_PARENT["candidate"],
        "parent_c1_config_id",
    )
    assert_equal(
        parent.get("parent_c1_protocol_hash"),
        C1_PARENT["protocol_hash"],
        "parent_c1_protocol_hash",
    )
    assert_equal(
        parent.get("parent_c2_timestamp"),
        C2_PARENT["timestamp"],
        "parent_c2_timestamp",
    )
    assert_equal(
        parent.get("parent_c2_config_id"),
        C2_PARENT["candidate"],
        "parent_c2_config_id",
    )
    assert_equal(
        parent.get("parent_c2_protocol_hash"),
        C2_PARENT["protocol_hash"],
        "parent_c2_protocol_hash",
    )

    assert_equal(
        recomputed_parent_realized_hash,
        H1H3_REALIZED_PLAN_SHA256,
        "recomputed realized_plan SHA256",
    )

    realized = realized_parent

    assert_equal(
        realized.get("effective_base_rows"),
        DATASET_PLAN["effective_base_rows"],
        "realized_plan.effective_base_rows",
    )
    assert_equal(
        realized.get("n_future_raw"),
        DATASET_PLAN["effective_future_rows"],
        "realized_plan.n_future_raw",
    )
    assert_equal(
        realized.get("n_future_warm"),
        DATASET_PLAN["future_warm_rows"],
        "realized_plan.n_future_warm",
    )
    assert_equal(
        realized.get("base_users"),
        DATASET_PLAN["base_users"],
        "realized_plan.base_users",
    )
    assert_equal(
        realized.get("base_items"),
        DATASET_PLAN["base_items"],
        "realized_plan.base_items",
    )
    assert_equal(
        realized.get("warm_chunk_boundaries"),
        DATASET_PLAN["warm_chunk_boundaries"],
        "realized_plan.warm_chunk_boundaries",
    )
    assert_equal(
        normalize_parent_chunks(parent),
        DATASET_PLAN["chunks"],
        "realized_plan.warm_chunks",
    )
    assert_equal(
        normalize_parent_primary_plan(parent),
        DATASET_PLAN["eval_points"],
        "realized_plan.primary_eval_plan",
    )

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "protocol_version": parent["protocol_version"],
        "protocol_hash": parent["protocol_hash"],
        "realized_plan_sha256": parent["realized_plan_sha256"],
        "recomputed_protocol_hash": recomputed_parent_protocol_hash,
        "recomputed_realized_plan_sha256": recomputed_parent_realized_hash,
        "script_sha256": parent["script_sha256"],
        "parent_source_hashes": {
            "ibpr_core": parent["ibpr_core_file_sha256"],
            "online_core": parent["online_core_file_sha256"],
            "ibpr_wrapper": parent["ibpr_wrapper_file_sha256"],
            "online_wrapper": parent["online_wrapper_file_sha256"],
        },
    }


def validate_source_continuity(source_files, parent_info):
    observed = {
        label: sha256_file(path)
        for label, path in source_files.items()
    }

    # V1.4 requires the complete model source surface used by H4 to remain exactly
    # the one recorded by FINAL H1-H3. The original wrappers already expose the
    # ANNMixin/vector contracts needed by H4, which are checked independently below.
    for label in [
        "ibpr_core",
        "online_core",
        "ibpr_wrapper",
        "online_wrapper",
    ]:
        expected = PARENT_SOURCE_HASHES[label]
        parent_recorded = parent_info["parent_source_hashes"][label]

        assert_equal(
            parent_recorded,
            expected,
            f"parent source hash {label}",
        )

        if observed[label] != expected:
            raise RuntimeError(
                f"Source drift detectado en {label}:\n"
                f"  observed={observed[label]}\n"
                f"  H1-H3  ={expected}"
            )

    wrapper_status = {
        label: {
            "parent_h1h3_sha256": PARENT_SOURCE_HASHES[label],
            "current_h4_sha256": observed[label],
            "byte_identical_to_h1h3": True,
        }
        for label in ["ibpr_wrapper", "online_wrapper"]
    }

    return {
        "observed_hashes": observed,
        "wrapper_status": wrapper_status,
        "all_four_source_files_byte_identical_to_h1h3": True,
        "source_continuity_policy": (
            "IBPR core, Online core, IBPR wrapper, and Online wrapper must all be "
            "byte-identical to the FINAL H1-H3 source hashes. ANNMixin/vector/score "
            "behavior is validated separately by semantic and ANN contract checks."
        ),
    }


def read_text_log_robust(path):
    """
    Decode persisted terminal logs deterministically across common PowerShell
    redirection encodings. The raw file SHA256 remains the provenance identity.
    """
    raw = path.read_bytes()

    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"

    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le"), "utf-16-le-bom"

    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be"), "utf-16-be-bom"

    # Detect BOM-less UTF-16 BEFORE accepting UTF-8, because ASCII-heavy UTF-16
    # can decode as UTF-8 without an exception while leaving interleaved NULs.
    sample = raw[: min(len(raw), 8192)]
    odd_nuls = sum(sample[i] == 0 for i in range(1, len(sample), 2))
    even_nuls = sum(sample[i] == 0 for i in range(0, len(sample), 2))
    odd_slots = max(1, len(sample) // 2)
    even_slots = max(1, (len(sample) + 1) // 2)

    if odd_nuls / odd_slots > 0.20:
        return raw.decode("utf-16-le"), "utf-16-le-heuristic"

    if even_nuls / even_slots > 0.20:
        return raw.decode("utf-16-be"), "utf-16-be-heuristic"

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "No se pudo decodificar el log de preflight como UTF-8/UTF-16 "
            "de forma determinista."
        ) from exc

    if "\x00" in decoded:
        raise RuntimeError(
            "El log decodificado como UTF-8 contiene NULs inesperados; "
            "se rechaza para evitar una clasificación de encoding ambigua."
        )

    return decoded, "utf-8"


def validate_preflight_log(path):
    if not path.exists():
        raise RuntimeError(
            "No se encontró un log persistente del preflight H4 V1.5:\n"
            f"  {path}\n"
            "Reejecuta V1.5 y guarda stdout/stderr antes de congelar H4."
        )

    text, detected_encoding = read_text_log_robust(path)

    required_markers = [
        "H4 ANN REUSE PREFLIGHT V1.5",
        "PRE-FLIGHT H4 V1.5: PASS",
        "V bit-exact at every step    : True",
        "V SHA equal at every step    : True",
        "Base index unchanged         : True",
        "Online rebuilds              : 0",
        "Tracked index builds         : 4 (base=1, online=0, full_rebuilt=3)",
        "Evidence status : synthetic preflight only; NOT final H4 evidence.",
    ]

    missing = [
        marker for marker in required_markers
        if marker not in text
    ]

    if missing:
        raise RuntimeError(
            "El log de preflight no contiene marcadores PASS requeridos:\n  - "
            + "\n  - ".join(missing)
        )

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "detected_encoding": detected_encoding,
        "required_markers_confirmed": True,
    }


def protocol_payload(
    env_signature,
    parent_info,
    source_continuity,
    wrapper_semantics,
    ann_contract,
    preflight_log_info,
):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "parent": {
            "h1h3_timestamp": H1H3_TIMESTAMP,
            "h1h3_protocol_version": H1H3_PROTOCOL_VERSION,
            "h1h3_protocol_hash": H1H3_PROTOCOL_HASH,
            "h1h3_manifest_sha256": parent_info["sha256"],
            "h1h3_script_sha256": H1H3_SCRIPT_SHA256,
            "h1h3_realized_plan_sha256": H1H3_REALIZED_PLAN_SHA256,
            "data_sha256": DATA_SHA256,
            "c1_data_sha256": C1_DATA_SHA256,
            "c1_parent": C1_PARENT,
            "c2_parent": C2_PARENT,
        },
        "final_seeds": FINAL_SEEDS,
        "r900": R900,
        "o019": O019,
        "dataset_plan": DATASET_PLAN,
        "model_chronology": MODEL_CHRONOLOGY,
        "h4_ann": H4_ANN,
        "retrieval_policy": RETRIEVAL_POLICY,
        "retrieval_metric_definitions": RETRIEVAL_METRIC_DEFINITIONS,
        "aggregation_plan": AGGREGATION_PLAN,
        "u_update_evidence": U_UPDATE_EVIDENCE,
        "build_accounting": BUILD_ACCOUNTING,
        "h4_branches": H4_BRANCHES,
        "h4_references": H4_REFERENCES,
        "h4_metrics": H4_METRICS,
        "h4_criteria": H4_CRITERIA,
        "expected_build_counts": EXPECTED_BUILD_COUNTS,
        "runtime_environment_signature": env_signature,
        "source_continuity": source_continuity,
        "wrapper_semantics": wrapper_semantics,
        "ann_contract": ann_contract,
        "preflight": {
            "script_filename": PREFLIGHT_FILENAME,
            "script_sha256": PREFLIGHT_SHA256,
            "execution_log_sha256": preflight_log_info["sha256"],
            "status": "PASS",
            "evidence_role": "synthetic preflight only; not final H4 evidence",
        },
        "retuning_after_freeze": "PROHIBITED",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Freeze H4 final protocol without running final H4 outcomes."
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Required. Validate/freeze protocol only.",
    )
    parser.add_argument(
        "--h1h3-manifest",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to FINAL H1-H3 V3.3 manifest. "
            "Default: experiments/results/<frozen manifest filename>."
        ),
    )
    parser.add_argument(
        "--preflight-log",
        type=Path,
        default=None,
        help=(
            "Persisted stdout/stderr log from PASSED preflight H4 V1.5. "
            "Default: experiments/results/"
            + DEFAULT_PREFLIGHT_LOG
        ),
    )
    args = parser.parse_args()

    if not args.plan_only:
        raise RuntimeError(
            "Este artefacto es exclusivamente de freeze. "
            "Ejecuta con --plan-only."
        )

    paths = resolve_repo_paths()
    paths["results_dir"].mkdir(parents=True, exist_ok=True)

    parent_manifest_path = (
        args.h1h3_manifest.resolve()
        if args.h1h3_manifest is not None
        else (paths["results_dir"] / DEFAULT_PARENT_MANIFEST).resolve()
    )

    preflight_log_path = (
        args.preflight_log.resolve()
        if args.preflight_log is not None
        else (paths["results_dir"] / DEFAULT_PREFLIGHT_LOG).resolve()
    )

    print("=" * 124)
    print("H4 FINAL V1.4 -- PLAN ONLY / PROTOCOL FREEZE")
    print("=" * 124)
    print(
        "No final H4 model training, Online update, MovieLens ANN build, "
        "or H4 outcome evaluation will be executed."
    )
    print()

    env = runtime_environment()
    assert_expected_preflight_environment(env)
    env_sig = environment_signature(env)

    print("H4 runtime environment: PASS")
    print(f"  Python : {env['python']}")
    print(f"  Cornac : {env['cornac']}")
    print(f"  NumPy  : {env['numpy']}")
    print(f"  SciPy  : {env['scipy']}  [bound into H4 protocol hash]")
    print(f"  PyTorch: {env['torch']}")
    print(f"  FAISS  : {env['faiss']}")
    print(
        f"  threads: torch={env['torch_num_threads']} | "
        f"interop={env['torch_num_interop_threads']} | "
        f"faiss={env['faiss_threads']}"
    )

    parent_info = validate_parent_manifest(parent_manifest_path)

    print()
    print("Frozen H1-H3 parent manifest: PASS")
    print(f"  path   : {parent_info['path']}")
    print(f"  sha256 : {parent_info['sha256']}")
    print(f"  protocol: {parent_info['protocol_hash']}")
    print(f"  recomputed protocol: {parent_info['recomputed_protocol_hash']}")
    print(f"  realized plan: {parent_info['realized_plan_sha256']}")
    print(
        "  recomputed realized plan: "
        f"{parent_info['recomputed_realized_plan_sha256']}"
    )

    preflight_script_path = paths["experiments_dir"] / PREFLIGHT_FILENAME

    if not preflight_script_path.exists():
        raise RuntimeError(
            "No se encontró el preflight H4 V1.5 junto al freeze:\n"
            f"  {preflight_script_path}"
        )

    observed_preflight_sha = sha256_file(preflight_script_path)

    if observed_preflight_sha != PREFLIGHT_SHA256:
        raise RuntimeError(
            "SHA256 inesperado para preflight H4 V1.5:\n"
            f"  observed={observed_preflight_sha}\n"
            f"  expected={PREFLIGHT_SHA256}"
        )

    preflight_log_info = validate_preflight_log(preflight_log_path)

    print()
    print("H4 preflight provenance: PASS")
    print(f"  script sha256 : {observed_preflight_sha}")
    print(f"  log           : {preflight_log_info['path']}")
    print(f"  log sha256    : {preflight_log_info['sha256']}")
    print(f"  log encoding  : {preflight_log_info['detected_encoding']}")

    source_files = discover_model_source_files()
    source_continuity = validate_source_continuity(
        source_files,
        parent_info,
    )
    wrapper_semantics = assert_wrapper_semantics()
    ann_contract = assert_ann_contracts()

    print()
    print("Source continuity:")
    for label, source in source_files.items():
        observed_hash = source_continuity["observed_hashes"][label]
        print(f"  {label:<18}: {observed_hash}")
        print(f"    {source}")

    print(
        "  full source continuity: PASS "
        "(cores + wrappers byte-identical to FINAL H1-H3)"
    )

    for label, status in source_continuity["wrapper_status"].items():
        print(f"  {label:<18}: IDENTICAL TO H1-H3")

    print()
    print("Wrapper/ANN semantic contracts: PASS")
    print(f"  wrapper semantics: {wrapper_semantics}")
    print(f"  ANN contract     : {ann_contract}")

    payload = protocol_payload(
        env_signature=env_sig,
        parent_info=parent_info,
        source_continuity=source_continuity,
        wrapper_semantics=wrapper_semantics,
        ann_contract=ann_contract,
        preflight_log_info=preflight_log_info,
    )
    protocol_hash = canonical_sha256(payload)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    freeze_script_sha = sha256_file(paths["script"])

    freeze_manifest = {
        "status": "FROZEN_PLAN_ONLY",
        "timestamp": timestamp,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "protocol": payload,
        "runtime_environment_full": env,
        "parent_h1h3_manifest": parent_info,
        "source_files": {
            label: {
                "path": str(source.resolve()),
                "sha256": source_continuity["observed_hashes"][label],
            }
            for label, source in source_files.items()
        },
        "freeze_script": {
            "path": str(paths["script"]),
            "sha256": freeze_script_sha,
        },
        "preflight_script": {
            "path": str(preflight_script_path.resolve()),
            "sha256": observed_preflight_sha,
        },
        "preflight_log": preflight_log_info,
        "execution_guard": {
            "plan_only": True,
            "h4_final_base_training_executed": False,
            "h4_final_online_update_executed": False,
            "h4_final_full_training_executed": False,
            "h4_final_indices_built": False,
            "h4_final_metrics_evaluated": False,
            "retuning_after_freeze": "PROHIBITED",
        },
    }

    out = (
        paths["results_dir"]
        / f"h4_final_v1_4_freeze_manifest_{timestamp}.json"
    )

    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            freeze_manifest,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        f.write("\n")

    print()
    print("Frozen H4 design:")
    print(f"  seeds                 : {FINAL_SEEDS}")
    print("  temporal points       : 3")
    print(
        "  PRIMARY users         : "
        + ", ".join(
            str(p["primary_users"])
            for p in DATASET_PLAN["eval_points"]
        )
    )
    print(
        "  ANN                   : "
        f"IndexIVFFlat/IP nlist={H4_ANN['nlist']} "
        f"nprobe={H4_ANN['nprobe']} "
        f"ANN-seed={H4_ANN['ann_clustering_seed']}"
    )
    print(
        "  retrieval             : initial_k=80, adaptive doubling, "
        "same observed filtering exact/ANN"
    )
    print(
        "  H4 frozen criterion   : 15/15 V bit-exact + zero Online rebuilds"
    )
    print(
        "  integrity guard       : unchanged base-index SHA/ntotal/d/metric/nlist/nprobe"
    )
    print(
        "  Online validity guard : 15/15 points with >=1 current-update user "
        "having delta_U_l2 > 0"
    )
    print(
        "  aggregation           : 3 repeated points -> seed mean -> 5-seed mean/std"
    )
    print(
        "  expected builds       : "
        "5 base + 0 online + 0 full_stale + 15 full_rebuilt = 20"
    )
    print("  ANN fidelity           : descriptive; no post-hoc success threshold")
    print("  H4 latency vs H2       : MUST NOT be merged")
    print("  retuning               : PROHIBITED")

    print()
    print("=" * 124)
    print("H4 FINAL V1.4 FREEZE: PASS")
    print("=" * 124)
    print(f"Protocol hash : {protocol_hash}")
    print(f"Manifest      : {out}")
    print(f"Manifest SHA  : {sha256_file(out)}")
    print()
    print(
        "PLAN ONLY complete. No final H4 model state, final ANN result, "
        "or final H4 metric was generated."
    )


if __name__ == "__main__":
    main()
