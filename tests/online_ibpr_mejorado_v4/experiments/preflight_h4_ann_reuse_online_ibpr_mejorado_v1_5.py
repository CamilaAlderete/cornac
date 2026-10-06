import os

# Freeze numerical thread pools BEFORE importing NumPy / Torch / SciPy / FAISS.
for _name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ[_name] = "1"

import hashlib
import platform
import statistics
import sys
import time

import numpy as np
import torch
import cornac
from scipy.sparse import csr_matrix

try:
    import faiss
except ImportError as exc:
    raise RuntimeError(
        "FAISS no está disponible. H4 requiere una instalación funcional de faiss."
    ) from exc

from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.models.recommender import ANNMixin, MEASURE_DOT


# =============================================================================
# H4 SYNTHETIC PREFLIGHT V1.5
# -----------------------------------------------------------------------------
# Purpose:
#   Validate the mechanics that the final H4 experiment will use:
#     * one base index built BEFORE any online update;
#     * three REAL, cumulative partial_fit_recent() calls;
#     * V_online == V_base bit-for-bit after every update;
#     * zero online rebuilds across the complete update sequence;
#     * same base-index identity/hash across all online points;
#     * paired PRIMARY query population across online/full_stale/full_rebuilt;
#     * exact-vs-ANN references, build accounting and timing methodology.
#
# This file is NOT final H4 evidence and does NOT use the final MovieLens holdout.
# =============================================================================

SEED = 42
DIM = 20
N_ITEMS = 1024
N_USERS = 32
TOP_K = 20

N_ONLINE_STEPS = 3
NLIST = 16
NPROBE = 8

BASE_OBSERVED_PER_USER = 35
RECENT_USERS_PER_STEP = 12
RECENT_PAIRS_PER_USER = 3

LATENCY_WARMUPS = 10
LATENCY_REPEATS = 50

# Fixed candidate budget for apples-to-apples RAW FAISS latency.
RAW_FAISS_K = 80

# Frozen Online configuration used by the final H1-H3 campaign.
ONLINE_CONFIG = {
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


def l2_normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Vector sintético de norma cero.")
    return np.ascontiguousarray(x / norms, dtype=np.float32)


def array_sha256(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.view(np.uint8).tobytes()).hexdigest()


def index_sha256(index):
    return hashlib.sha256(bytes(faiss.serialize_index(index))).hexdigest()


def make_base_observed_sets(rng, n_users, n_items, n_observed):
    if n_observed >= n_items:
        raise ValueError("n_observed debe ser menor que n_items.")

    observed = []
    for _ in range(n_users):
        items = rng.choice(n_items, size=n_observed, replace=False)
        observed.append(set(map(int, items)))
    return observed


def observed_sets_to_csr(observed_sets, n_items):
    rows = []
    cols = []

    for u, items in enumerate(observed_sets):
        for i in sorted(items):
            rows.append(u)
            cols.append(i)

    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix(
        (
            data,
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=(len(observed_sets), n_items),
        dtype=np.float32,
    )


def update_users_for_step(step):
    """
    Deterministic overlapping user windows:
      step 1:  0..11
      step 2:  8..19
      step 3: 16..27

    Cumulative PRIMARY sizes become 12, 20, 28 users. This mirrors the idea
    that the population exposed to online updates grows across temporal points.
    """
    if step < 1 or step > N_ONLINE_STEPS:
        raise ValueError("step fuera de rango.")

    start = (step - 1) * 8
    stop = start + RECENT_USERS_PER_STEP

    if stop > N_USERS:
        raise RuntimeError("Ventana de usuarios sintéticos excede N_USERS.")

    return np.arange(start, stop, dtype=np.int64)


def make_recent_chunk(rng, observed_sets, n_items, user_ids):
    """
    Create one chronological synthetic update chunk. Every pair is warm-start
    and absent from the history observed before this chunk.
    """
    pairs = []

    for u in user_ids:
        u = int(u)
        available = np.asarray(
            [i for i in range(n_items) if i not in observed_sets[u]],
            dtype=np.int64,
        )

        if len(available) < RECENT_PAIRS_PER_USER:
            raise RuntimeError(
                f"Usuario {u} sin suficientes ítems nuevos para el preflight."
            )

        chosen = rng.choice(
            available,
            size=RECENT_PAIRS_PER_USER,
            replace=False,
        )

        for i in chosen:
            pairs.append((u, int(i)))

    pairs = np.asarray(pairs, dtype=np.int64)

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise RuntimeError("recent_pairs sintético inválido.")

    for u, i in pairs:
        if int(i) in observed_sets[int(u)]:
            raise RuntimeError(
                "recent_pairs contiene un positivo ya observado antes del chunk."
            )

    return pairs


def add_recent_pairs_to_observed(observed_sets, recent_pairs):
    updated = [set(s) for s in observed_sets]
    for u, i in recent_pairs:
        updated[int(u)].add(int(i))
    return updated


def subset_observed_sets(observed_sets, user_ids):
    return [observed_sets[int(u)] for u in user_ids]


def exact_filtered_topk(U, V, observed_sets, k):
    """
    Exhaustive reference:
      score = U @ V.T
      remove already-observed items
      deterministic tie-break by item id
    """
    U = np.asarray(U, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)

    if len(observed_sets) != U.shape[0]:
        raise ValueError("observed_sets y U tienen distinto número de usuarios.")

    scores = U @ V.T
    item_ids = np.arange(V.shape[0], dtype=np.int64)

    out = np.full((U.shape[0], k), -1, dtype=np.int64)
    shortfalls = np.zeros(U.shape[0], dtype=np.int64)

    for u in range(U.shape[0]):
        order = np.lexsort((item_ids, -scores[u]))
        kept = [int(i) for i in order if int(i) not in observed_sets[u]]

        take = min(k, len(kept))
        if take:
            out[u, :take] = kept[:take]
        shortfalls[u] = k - take

    return out, shortfalls


def ann_filtered_topk(index, U, observed_sets, k):
    """
    ANN retrieval with adaptive over-retrieval and the SAME observed-item policy
    as the exhaustive reference.
    """
    U = np.ascontiguousarray(U, dtype=np.float32)

    if len(observed_sets) != U.shape[0]:
        raise ValueError("observed_sets y U tienen distinto número de usuarios.")

    n_users = U.shape[0]
    ntotal = int(index.ntotal)

    result = np.full((n_users, k), -1, dtype=np.int64)
    shortfalls = np.full(n_users, k, dtype=np.int64)

    initial_extra = BASE_OBSERVED_PER_USER + N_ONLINE_STEPS * RECENT_PAIRS_PER_USER + 10
    search_k = min(ntotal, max(k * 4, k + initial_extra))

    while True:
        _, candidates = index.search(U, search_k)

        all_complete = True

        for u in range(n_users):
            seen = set()
            kept = []

            for item in candidates[u]:
                item = int(item)

                if item < 0:
                    continue
                if item in observed_sets[u] or item in seen:
                    continue

                seen.add(item)
                kept.append(item)

                if len(kept) == k:
                    break

            result[u, :] = -1
            take = min(k, len(kept))

            if take:
                result[u, :take] = kept[:take]

            shortfalls[u] = k - take

            if shortfalls[u] > 0:
                all_complete = False

        if all_complete or search_k >= ntotal:
            break

        search_k = min(ntotal, max(search_k + 1, search_k * 2))

    return result, shortfalls, search_k


def valid_items(row):
    return [int(x) for x in row if int(x) >= 0]


def mean_set_recall(reference, retrieved):
    vals = []

    for ref, got in zip(reference, retrieved):
        ref_valid = valid_items(ref)
        got_valid = valid_items(got)

        if not ref_valid:
            continue

        vals.append(
            len(set(ref_valid) & set(got_valid)) / len(ref_valid)
        )

    return float(np.mean(vals)) if vals else float("nan")


def mean_positional_agreement(reference, retrieved):
    vals = []

    for ref, got in zip(reference, retrieved):
        mask = ref >= 0
        if not np.any(mask):
            continue

        vals.append(float(np.mean(ref[mask] == got[mask])))

    return float(np.mean(vals)) if vals else float("nan")


def exact_list_match_fraction(reference, retrieved):
    return float(
        np.mean([
            bool(np.array_equal(ref, got))
            for ref, got in zip(reference, retrieved)
        ])
    )


def retrieval_metrics(reference, retrieved, shortfalls):
    return {
        "set_recall": mean_set_recall(reference, retrieved),
        "positional_agreement": mean_positional_agreement(reference, retrieved),
        "exact_list_fraction": exact_list_match_fraction(reference, retrieved),
        "shortfall_total": int(np.sum(shortfalls)),
    }


def print_retrieval(label, reference, retrieved, shortfalls):
    m = retrieval_metrics(reference, retrieved, shortfalls)

    print(
        f"{label}: "
        f"set_recall@{TOP_K}={m['set_recall']:.4f} | "
        f"positional_agreement={m['positional_agreement']:.4f} | "
        f"exact_list_fraction={m['exact_list_fraction']:.4f} | "
        f"shortfall_total={m['shortfall_total']}"
    )

    return m


class IndexBuildTracker:
    """
    Every IVF construction made by this experimental protocol MUST pass through
    this object. Events are therefore actual build calls by this preflight.
    """

    def __init__(self):
        self.events = []

    def build_ivf(self, label, V, seed):
        started_ns = time.perf_counter_ns()

        V = np.ascontiguousarray(V, dtype=np.float32)
        quantizer = faiss.IndexFlatIP(V.shape[1])

        index = faiss.IndexIVFFlat(
            quantizer,
            V.shape[1],
            NLIST,
            faiss.METRIC_INNER_PRODUCT,
        )

        if hasattr(index, "cp") and hasattr(index.cp, "seed"):
            index.cp.seed = int(seed)

        index.nprobe = NPROBE
        index.train(V)

        if not index.is_trained:
            raise RuntimeError(f"IndexIVFFlat '{label}' no quedó entrenado.")

        index.add(V)

        if int(index.ntotal) != V.shape[0]:
            raise RuntimeError(
                f"IndexIVFFlat '{label}' ntotal={index.ntotal}; "
                f"esperado={V.shape[0]}."
            )

        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6

        self.events.append({
            "label": str(label),
            "elapsed_ms": float(elapsed_ms),
            "ntotal": int(index.ntotal),
            "hash": index_sha256(index),
            "object_id": id(index),
        })

        return index

    def count(self, label):
        return sum(event["label"] == label for event in self.events)

    def total_count(self):
        return len(self.events)


def latency_summary(search_fn):
    for _ in range(LATENCY_WARMUPS):
        search_fn()

    samples = []

    for _ in range(LATENCY_REPEATS):
        start_ns = time.perf_counter_ns()
        search_fn()
        samples.append((time.perf_counter_ns() - start_ns) / 1e6)

    samples.sort()

    median_ms = float(statistics.median(samples))
    p95_index = max(0, int(np.ceil(0.95 * len(samples))) - 1)
    p95_ms = float(samples[p95_index])

    return {
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "n": len(samples),
    }


def print_latency(label, summary):
    print(
        f"  {label:<22}: "
        f"median={summary['median_ms']:.4f} ms | "
        f"p95={summary['p95_ms']:.4f} ms | "
        f"n={summary['n']}"
    )


def assert_wrapper_ann_contract(model, expected_U, expected_V, name):
    if not isinstance(model, ANNMixin):
        raise RuntimeError(f"{name} no implementa ANNMixin.")

    if model.get_vector_measure() != MEASURE_DOT:
        raise RuntimeError(f"{name} no declara MEASURE_DOT.")

    if not np.array_equal(model.get_user_vectors(), expected_U):
        raise RuntimeError(f"{name}.get_user_vectors() no coincide con U.")

    if not np.array_equal(model.get_item_vectors(), expected_V):
        raise RuntimeError(f"{name}.get_item_vectors() no coincide con V.")

    user_idx = 0
    expected_scores = expected_V.dot(expected_U[user_idx, :])
    actual_scores = np.asarray(model.score(user_idx))

    if not np.array_equal(actual_scores, expected_scores):
        max_diff = float(np.max(np.abs(actual_scores - expected_scores)))

        if not np.allclose(
            actual_scores,
            expected_scores,
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"{name}.score() no coincide con V @ U. "
                f"max_abs_diff={max_diff:.3e}"
            )


def row_l2_deltas(after, before):
    return np.linalg.norm(
        np.asarray(after, dtype=np.float64)
        - np.asarray(before, dtype=np.float64),
        axis=1,
    )


def synthetic_full_state(rng, U_base, V_base, step):
    """
    Synthetic analogue only: creates a state in which both U and V differ from
    the base, with a slightly increasing perturbation across temporal points.
    The final H4 campaign will use actual IBPR full retraining instead.
    """
    scale = 0.06 + 0.02 * step

    U_full = l2_normalize_rows(
        U_base
        + rng.normal(scale=scale, size=U_base.shape).astype(np.float32)
    )

    V_full = l2_normalize_rows(
        V_base
        + rng.normal(scale=scale, size=V_base.shape).astype(np.float32)
    )

    if np.array_equal(V_full, V_base):
        raise RuntimeError(f"step {step}: V_full sintética no cambió.")

    return U_full, V_full


def main():
    # -------------------------------------------------------------------------
    # Reproducibility controls.
    # -------------------------------------------------------------------------
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(1)

    print("=" * 122)
    print("H4 ANN REUSE PREFLIGHT V1.5 - 3 CUMULATIVE REAL ONLINE UPDATES / SAME BASE INDEX")
    print("=" * 122)
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Platform      : {platform.platform()}")
    print(f"Cornac        : {getattr(cornac, '__version__', 'unavailable')}")
    print(f"NumPy         : {np.__version__}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"FAISS         : {getattr(faiss, '__version__', 'unavailable')}")

    if hasattr(faiss, "omp_get_max_threads"):
        print(f"FAISS threads : {faiss.omp_get_max_threads()}")

    print()

    # Independent deterministic RNG streams:
    #   rng_data drives base/history/recent online chunks;
    #   rng_full drives only synthetic Full diagnostic states.
    # This prevents the Full diagnostic branch from changing later Online inputs.
    rng_data = np.random.default_rng(SEED)
    rng_full = np.random.default_rng(SEED + 10_000)

    # -------------------------------------------------------------------------
    # 1. Synthetic normalized base state.
    # -------------------------------------------------------------------------
    U_base = l2_normalize_rows(
        rng_data.normal(size=(N_USERS, DIM)).astype(np.float32)
    )
    V_base = l2_normalize_rows(
        rng_data.normal(size=(N_ITEMS, DIM)).astype(np.float32)
    )

    current_observed = make_base_observed_sets(
        rng_data,
        n_users=N_USERS,
        n_items=N_ITEMS,
        n_observed=BASE_OBSERVED_PER_USER,
    )

    # -------------------------------------------------------------------------
    # 2. Wrapper ANN contract on the base state.
    # -------------------------------------------------------------------------
    ibpr = IBPR(
        k=DIM,
        trainable=False,
        init_params={
            "U": U_base.copy(),
            "V": V_base.copy(),
        },
        name="h4_preflight_ibpr",
    )
    ibpr.num_users = N_USERS
    ibpr.num_items = N_ITEMS

    if not issubclass(IBPR, ANNMixin):
        raise RuntimeError("IBPR no hereda ANNMixin.")

    assert_wrapper_ann_contract(
        ibpr,
        expected_U=U_base,
        expected_V=V_base,
        name="IBPR",
    )

    print("IBPR ANN wrapper contract: OK")

    # -------------------------------------------------------------------------
    # 3. Build the ONE base ANN index BEFORE any online update.
    # -------------------------------------------------------------------------
    tracker = IndexBuildTracker()

    base_index = tracker.build_ivf(
        label="base",
        V=V_base,
        seed=SEED,
    )

    base_object_id = id(base_index)
    base_hash = index_sha256(base_index)
    base_ntotal = int(base_index.ntotal)

    if tracker.count("base") != 1 or tracker.total_count() != 1:
        raise RuntimeError(
            "El índice base no fue construido exactamente una vez."
        )

    print(
        "Base index constructed BEFORE all online updates: OK | "
        f"object_id={base_object_id} | sha256={base_hash}"
    )

    # -------------------------------------------------------------------------
    # 4. Exact-vs-IndexFlatIP policy reference on the base state.
    # -------------------------------------------------------------------------
    exact_base, exact_base_shortfall = exact_filtered_topk(
        U_base,
        V_base,
        current_observed,
        TOP_K,
    )

    if int(np.sum(exact_base_shortfall)) != 0:
        raise RuntimeError(
            "Ground truth exhaustivo base tiene shortfall inesperado."
        )

    flat_base = faiss.IndexFlatIP(DIM)
    flat_base.add(np.ascontiguousarray(V_base, dtype=np.float32))

    flat_retrieved, flat_shortfall, _ = ann_filtered_topk(
        flat_base,
        U_base,
        current_observed,
        TOP_K,
    )

    if int(np.sum(flat_shortfall)) != 0:
        raise RuntimeError("IndexFlatIP produjo shortfall inesperado.")

    flat_metrics = retrieval_metrics(
        exact_base,
        flat_retrieved,
        flat_shortfall,
    )

    if flat_metrics["set_recall"] != 1.0:
        raise RuntimeError(
            "IndexFlatIP no recuperó exactamente el mismo top-k SET que NumPy. "
            f"set_recall={flat_metrics['set_recall']:.6f}"
        )

    print(
        "Observed-item filtering + exact/IndexFlatIP agreement: OK | "
        f"set_recall={flat_metrics['set_recall']:.4f} | "
        f"positional={flat_metrics['positional_agreement']:.4f}"
    )

    base_ann, base_shortfall, _ = ann_filtered_topk(
        base_index,
        U_base,
        current_observed,
        TOP_K,
    )

    print_retrieval(
        "base_ivf vs exact(U_base,V_base)",
        exact_base,
        base_ann,
        base_shortfall,
    )

    if id(base_index) != base_object_id:
        raise RuntimeError("Cambió la identidad del índice base.")
    if index_sha256(base_index) != base_hash:
        raise RuntimeError("Cambió el hash del índice base tras consultas base.")
    if int(base_index.ntotal) != base_ntotal:
        raise RuntimeError("Cambió ntotal del índice base tras consultas base.")

    # -------------------------------------------------------------------------
    # 5. Create the REAL Online wrapper once and update it cumulatively.
    # -------------------------------------------------------------------------
    online = OnlineIBPRMejorado(
        k=DIM,
        learning_rate=ONLINE_CONFIG["learning_rate"],
        lamda=ONLINE_CONFIG["lamda"],
        batch_size=ONLINE_CONFIG["batch_size"],
        trainable=True,
        init_params={
            "U": U_base.copy(),
            "V": V_base.copy(),
        },
        update_V=ONLINE_CONFIG["update_V"],
        neg_sampling=ONLINE_CONFIG["neg_sampling"],
        normalize=ONLINE_CONFIG["normalize"],
        loss_mode=ONLINE_CONFIG["loss_mode"],
        seed=SEED,
        name="h4_preflight_online",
    )

    if not issubclass(OnlineIBPRMejorado, ANNMixin):
        raise RuntimeError("OnlineIBPRMejorado no hereda ANNMixin.")

    if not hasattr(online, "_partial_update_count"):
        raise RuntimeError(
            "OnlineIBPRMejorado no expone _partial_update_count; "
            "no puede auditarse la progresión de seeds de actualización."
        )

    if int(online._partial_update_count) != 0:
        raise RuntimeError(
            "_partial_update_count debe iniciar en 0."
        )

    cumulative_explicit_users = set()
    step_records = []

    # -------------------------------------------------------------------------
    # 6. Three cumulative prequential-style online updates.
    # -------------------------------------------------------------------------
    for step in range(1, N_ONLINE_STEPS + 1):
        print()
        print("-" * 122)
        print(f"ONLINE STEP {step}/{N_ONLINE_STEPS}")
        print("-" * 122)

        step_users = update_users_for_step(step)

        recent_pairs = make_recent_chunk(
            rng_data,
            current_observed,
            N_ITEMS,
            step_users,
        )

        # Mirror the final H1-H3 protocol:
        # history_csr contains the post-update history (history + current chunk).
        post_update_observed = add_recent_pairs_to_observed(
            current_observed,
            recent_pairs,
        )
        history_post_update_csr = observed_sets_to_csr(
            post_update_observed,
            N_ITEMS,
        )

        U_before = np.asarray(online.U).copy()
        V_before = np.asarray(online.V).copy()

        index_id_pre_update = id(base_index)
        index_hash_pre_update = index_sha256(base_index)
        index_ntotal_pre_update = int(base_index.ntotal)
        builds_pre_update = tracker.total_count()

        partial_count_before = int(online._partial_update_count)

        if partial_count_before != step - 1:
            raise RuntimeError(
                f"step {step}: _partial_update_count previo={partial_count_before}, "
                f"esperado={step - 1}."
            )

        online.partial_fit_recent(
            recent_pairs=recent_pairs,
            history_csr=history_post_update_csr,
            max_steps=ONLINE_CONFIG["max_steps"],
            n_epochs=ONLINE_CONFIG["n_epochs"],
        )

        U_online = np.asarray(online.U)
        V_online = np.asarray(online.V)

        # The REAL wrapper must advance exactly one update seed per non-empty call.
        partial_count_after = int(online._partial_update_count)

        if partial_count_after != step:
            raise RuntimeError(
                f"step {step}: _partial_update_count posterior={partial_count_after}, "
                f"esperado={step}."
            )

        # Actual observed build delta for this Online update. Prior Full-rebuilt
        # indices from earlier steps are allowed to exist; what must be zero is
        # the number of NEW builds caused during this partial update.
        builds_after_update = tracker.total_count()
        online_build_delta = builds_after_update - builds_pre_update

        if online_build_delta != 0:
            raise RuntimeError(
                f"step {step}: partial_fit_recent produjo "
                f"{online_build_delta} build(s)/rebuild(s) de índice."
            )

        # Same already-built base index must survive every update unchanged.
        if id(base_index) != index_id_pre_update:
            raise RuntimeError(
                f"step {step}: cambió la identidad del índice base."
            )

        if index_sha256(base_index) != index_hash_pre_update:
            raise RuntimeError(
                f"step {step}: cambió el hash del índice base."
            )

        if int(base_index.ntotal) != index_ntotal_pre_update:
            raise RuntimeError(
                f"step {step}: cambió ntotal del índice base."
            )

        # Current explicitly updated users must actually move in U.
        deltas = row_l2_deltas(U_online, U_before)
        current_deltas = deltas[step_users]

        if not np.all(current_deltas > 0.0):
            bad = step_users[current_deltas <= 0.0]
            raise RuntimeError(
                f"step {step}: usuarios explícitamente actualizados sin ΔU: "
                f"{bad.tolist()}"
            )

        # V must remain exactly equal to both prior V and original V_base.
        if not np.array_equal(V_online, V_before):
            max_diff = float(np.max(np.abs(V_online - V_before)))
            raise RuntimeError(
                f"step {step}: V cambió respecto del estado Online anterior. "
                f"max_abs_diff={max_diff:.3e}"
            )

        if not np.array_equal(V_online, V_base):
            max_diff = float(np.max(np.abs(V_online - V_base)))
            raise RuntimeError(
                f"step {step}: V_online != V_base. max_abs_diff={max_diff:.3e}"
            )

        if array_sha256(V_online) != array_sha256(V_base):
            raise RuntimeError(
                f"step {step}: SHA256(V_online) != SHA256(V_base)."
            )

        assert_wrapper_ann_contract(
            online,
            expected_U=U_online,
            expected_V=V_online,
            name=f"OnlineIBPRMejorado step={step}",
        )

        cumulative_explicit_users.update(map(int, step_users))
        primary_users = np.asarray(
            sorted(cumulative_explicit_users),
            dtype=np.int64,
        )
        primary_observed = subset_observed_sets(
            post_update_observed,
            primary_users,
        )

        # ---------------------------------------------------------------------
        # online_reused PRIMARY at this temporal point.
        # ---------------------------------------------------------------------
        exact_online_primary, exact_online_shortfall = exact_filtered_topk(
            U_online[primary_users],
            V_online,
            primary_observed,
            TOP_K,
        )

        if int(np.sum(exact_online_shortfall)) != 0:
            raise RuntimeError(
                f"step {step}: ground truth Online PRIMARY tiene shortfall."
            )

        online_ann, online_shortfall, _ = ann_filtered_topk(
            base_index,
            U_online[primary_users],
            primary_observed,
            TOP_K,
        )

        online_metrics = print_retrieval(
            f"step {step} online_reused PRIMARY",
            exact_online_primary,
            online_ann,
            online_shortfall,
        )

        # ---------------------------------------------------------------------
        # Synthetic Full state for the SAME temporal point.
        # PRIMARY user population is exactly the same as online_reused.
        # ---------------------------------------------------------------------
        U_full, V_full = synthetic_full_state(
            rng_full,
            U_base,
            V_base,
            step,
        )

        exact_full_baseV, full_baseV_shortfall = exact_filtered_topk(
            U_full[primary_users],
            V_base,
            primary_observed,
            TOP_K,
        )

        exact_full_true, full_true_shortfall = exact_filtered_topk(
            U_full[primary_users],
            V_full,
            primary_observed,
            TOP_K,
        )

        if int(np.sum(full_baseV_shortfall)) != 0:
            raise RuntimeError(
                f"step {step}: exact(U_full,V_base) tiene shortfall."
            )

        if int(np.sum(full_true_shortfall)) != 0:
            raise RuntimeError(
                f"step {step}: exact(U_full,V_full) tiene shortfall."
            )

        # Pure geometry shift; no ANN approximation.
        geometry_metrics = print_retrieval(
            f"step {step} EXACT geometry shift",
            exact_full_true,
            exact_full_baseV,
            np.zeros(len(primary_users), dtype=np.int64),
        )

        # ---------------------------------------------------------------------
        # full_stale: same PRIMARY users, old base index, NO rebuild.
        # ---------------------------------------------------------------------
        full_stale_ann, full_stale_shortfall, _ = ann_filtered_topk(
            base_index,
            U_full[primary_users],
            primary_observed,
            TOP_K,
        )

        full_stale_index_metrics = print_retrieval(
            f"step {step} full_stale INDEX-CONSISTENCY",
            exact_full_baseV,
            full_stale_ann,
            full_stale_shortfall,
        )

        full_stale_model_metrics = print_retrieval(
            f"step {step} full_stale MODEL-FAITHFULNESS",
            exact_full_true,
            full_stale_ann,
            full_stale_shortfall,
        )

        if tracker.total_count() != 1 + (step - 1):
            raise RuntimeError(
                f"step {step}: full_stale provocó un build inesperado."
            )

        if id(base_index) != base_object_id:
            raise RuntimeError(
                f"step {step}: full_stale perdió identidad del índice base."
            )

        if index_sha256(base_index) != base_hash:
            raise RuntimeError(
                f"step {step}: full_stale mutó el índice base."
            )

        # ---------------------------------------------------------------------
        # full_rebuilt: one NEW index over V_full for this temporal point.
        # ---------------------------------------------------------------------
        full_label = f"full_rebuilt_step_{step}"
        full_index = tracker.build_ivf(
            label=full_label,
            V=V_full,
            seed=SEED,
        )

        full_hash = index_sha256(full_index)

        if tracker.count(full_label) != 1:
            raise RuntimeError(
                f"step {step}: full_rebuilt no produjo exactamente un build."
            )

        expected_total_builds = 1 + step
        if tracker.total_count() != expected_total_builds:
            raise RuntimeError(
                f"step {step}: builds={tracker.total_count()}, "
                f"esperado={expected_total_builds}."
            )

        if id(full_index) == base_object_id:
            raise RuntimeError(
                f"step {step}: full_rebuilt reutilizó el objeto base."
            )

        if full_hash == base_hash:
            raise RuntimeError(
                f"step {step}: índice V_full tiene el mismo hash que base."
            )

        full_rebuilt_ann, full_rebuilt_shortfall, _ = ann_filtered_topk(
            full_index,
            U_full[primary_users],
            primary_observed,
            TOP_K,
        )

        full_rebuilt_metrics = print_retrieval(
            f"step {step} full_rebuilt",
            exact_full_true,
            full_rebuilt_ann,
            full_rebuilt_shortfall,
        )

        # ---------------------------------------------------------------------
        # Latency at this point, paired on SAME PRIMARY users and SAME raw_k.
        # ---------------------------------------------------------------------
        raw_k = min(
            RAW_FAISS_K,
            int(base_index.ntotal),
            int(full_index.ntotal),
        )

        U_online_primary_c = np.ascontiguousarray(
            U_online[primary_users],
            dtype=np.float32,
        )
        U_full_primary_c = np.ascontiguousarray(
            U_full[primary_users],
            dtype=np.float32,
        )

        raw_online_latency = latency_summary(
            lambda: base_index.search(U_online_primary_c, raw_k)
        )
        raw_full_stale_latency = latency_summary(
            lambda: base_index.search(U_full_primary_c, raw_k)
        )
        raw_full_rebuilt_latency = latency_summary(
            lambda: full_index.search(U_full_primary_c, raw_k)
        )

        filtered_online_latency = latency_summary(
            lambda: ann_filtered_topk(
                base_index,
                U_online[primary_users],
                primary_observed,
                TOP_K,
            )
        )
        filtered_full_stale_latency = latency_summary(
            lambda: ann_filtered_topk(
                base_index,
                U_full[primary_users],
                primary_observed,
                TOP_K,
            )
        )
        filtered_full_rebuilt_latency = latency_summary(
            lambda: ann_filtered_topk(
                full_index,
                U_full[primary_users],
                primary_observed,
                TOP_K,
            )
        )

        print(
            f"step {step} RAW FAISS latency "
            f"(same PRIMARY users, same k={raw_k}):"
        )
        print_latency("online_reused", raw_online_latency)
        print_latency("full_stale", raw_full_stale_latency)
        print_latency("full_rebuilt", raw_full_rebuilt_latency)

        print(
            f"step {step} FILTERED end-to-end latency "
            f"(same PRIMARY users):"
        )
        print_latency("online_reused", filtered_online_latency)
        print_latency("full_stale", filtered_full_stale_latency)
        print_latency("full_rebuilt", filtered_full_rebuilt_latency)

        step_records.append({
            "step": step,
            "current_update_users": len(step_users),
            "primary_users": len(primary_users),
            "min_current_delta": float(np.min(current_deltas)),
            "median_current_delta": float(np.median(current_deltas)),
            "v_exact": bool(np.array_equal(V_online, V_base)),
            "v_sha_equal": bool(array_sha256(V_online) == array_sha256(V_base)),
            "base_index_hash_equal": bool(index_sha256(base_index) == base_hash),
            "online_rebuilds": int(online_build_delta),
            "online_recall": online_metrics["set_recall"],
            "online_shortfall": online_metrics["shortfall_total"],
            "geometry_fidelity": geometry_metrics["set_recall"],
            "full_stale_index_recall": full_stale_index_metrics["set_recall"],
            "full_stale_model_fidelity": full_stale_model_metrics["set_recall"],
            "full_rebuilt_recall": full_rebuilt_metrics["set_recall"],
            "full_rebuilt_shortfall": full_rebuilt_metrics["shortfall_total"],
            "raw_online_median_ms": raw_online_latency["median_ms"],
            "raw_full_stale_median_ms": raw_full_stale_latency["median_ms"],
            "raw_full_rebuilt_median_ms": raw_full_rebuilt_latency["median_ms"],
            "filtered_online_median_ms": filtered_online_latency["median_ms"],
            "filtered_full_stale_median_ms": filtered_full_stale_latency["median_ms"],
            "filtered_full_rebuilt_median_ms": filtered_full_rebuilt_latency["median_ms"],
        })

        # Accumulate history only after the step is fully validated.
        current_observed = post_update_observed

        # The base index must STILL be unchanged after all step work.
        if id(base_index) != base_object_id:
            raise RuntimeError(
                f"step {step}: guard final de identidad del índice base falló."
            )

        if index_sha256(base_index) != base_hash:
            raise RuntimeError(
                f"step {step}: guard final de hash del índice base falló."
            )

        if int(base_index.ntotal) != base_ntotal:
            raise RuntimeError(
                f"step {step}: guard final de ntotal del índice base falló."
            )

    # -------------------------------------------------------------------------
    # 7. Final all-user Online diagnostic after the third cumulative update.
    # -------------------------------------------------------------------------
    exact_online_all, exact_online_all_shortfall = exact_filtered_topk(
        np.asarray(online.U),
        np.asarray(online.V),
        current_observed,
        TOP_K,
    )

    if int(np.sum(exact_online_all_shortfall)) != 0:
        raise RuntimeError(
            "Ground truth Online ALL-USERS final tiene shortfall inesperado."
        )

    online_all_ann, online_all_shortfall, _ = ann_filtered_topk(
        base_index,
        np.asarray(online.U),
        current_observed,
        TOP_K,
    )

    online_all_metrics = print_retrieval(
        "FINAL online_reused ALL-USERS supplementary",
        exact_online_all,
        online_all_ann,
        online_all_shortfall,
    )

    # -------------------------------------------------------------------------
    # 8. Final build accounting and invariants.
    # -------------------------------------------------------------------------
    expected_total_builds = 1 + N_ONLINE_STEPS

    if tracker.count("base") != 1:
        raise RuntimeError(
            "Preflight requiere exactamente un build base."
        )

    for step in range(1, N_ONLINE_STEPS + 1):
        label = f"full_rebuilt_step_{step}"
        if tracker.count(label) != 1:
            raise RuntimeError(
                f"Preflight requiere exactamente un build '{label}'."
            )

    if tracker.total_count() != expected_total_builds:
        raise RuntimeError(
            f"Builds totales={tracker.total_count()}, "
            f"esperado={expected_total_builds}."
        )

    if not np.array_equal(np.asarray(online.V), V_base):
        raise RuntimeError(
            "Guard final: V_online final != V_base."
        )

    if array_sha256(np.asarray(online.V)) != array_sha256(V_base):
        raise RuntimeError(
            "Guard final: SHA256(V_online final) != SHA256(V_base)."
        )

    if index_sha256(base_index) != base_hash:
        raise RuntimeError(
            "Guard final: el índice base cambió."
        )

    if id(base_index) != base_object_id:
        raise RuntimeError(
            "Guard final: cambió la identidad del índice base."
        )

    if int(base_index.ntotal) != base_ntotal:
        raise RuntimeError(
            "Guard final: cambió ntotal del índice base."
        )

    if not hasattr(online, "_partial_update_count"):
        raise RuntimeError(
            "Guard final: falta _partial_update_count."
        )

    if int(online._partial_update_count) != N_ONLINE_STEPS:
        raise RuntimeError(
            "_partial_update_count final no coincide con el número de updates."
        )

    # -------------------------------------------------------------------------
    # 9. Summary.
    # -------------------------------------------------------------------------
    print()
    print("=" * 122)
    print("PRE-FLIGHT H4 V1.5: PASS")
    print("=" * 122)
    print(f"Online steps                 : {N_ONLINE_STEPS}")
    print(
        f"Final explicit PRIMARY users : "
        f"{len(cumulative_explicit_users)}/{N_USERS}"
    )
    print(
        f"V bit-exact at every step    : "
        f"{all(r['v_exact'] for r in step_records)}"
    )
    print(
        f"V SHA equal at every step    : "
        f"{all(r['v_sha_equal'] for r in step_records)}"
    )
    print(
        f"Base index unchanged         : "
        f"{all(r['base_index_hash_equal'] for r in step_records)}"
    )
    print(
        f"Online rebuilds              : "
        f"{sum(r['online_rebuilds'] for r in step_records)}"
    )
    print(
        f"Tracked index builds         : {tracker.total_count()} "
        f"(base=1, online=0, full_rebuilt={N_ONLINE_STEPS})"
    )
    print(f"Base index SHA256            : {base_hash}")
    print(f"V SHA256 base                : {array_sha256(V_base)}")
    print(f"V SHA256 online final        : {array_sha256(np.asarray(online.V))}")
    print(
        f"Final ALL-USERS recall@{TOP_K}   : "
        f"{online_all_metrics['set_recall']:.4f}"
    )

    print()
    print("Per-step PRIMARY summary:")
    for r in step_records:
        print(
            f"  step={r['step']} | "
            f"PRIMARY users={r['primary_users']} | "
            f"V_exact={r['v_exact']} | "
            f"online_recall={r['online_recall']:.4f} | "
            f"online_shortfall={r['online_shortfall']} | "
            f"full_stale_index={r['full_stale_index_recall']:.4f} | "
            f"full_stale_model={r['full_stale_model_fidelity']:.4f} | "
            f"full_rebuilt={r['full_rebuilt_recall']:.4f} | "
            f"geometry={r['geometry_fidelity']:.4f}"
        )

    print()
    print("Tracked IVF build events:")
    for idx, event in enumerate(tracker.events, start=1):
        print(
            f"  #{idx} label={event['label']} | "
            f"time={event['elapsed_ms']:.4f} ms | "
            f"ntotal={event['ntotal']} | "
            f"object_id={event['object_id']} | "
            f"sha256={event['hash']}"
        )

    print()
    print(
        "Evidence status : synthetic preflight only; NOT final H4 evidence."
    )
    print(
        "Interpretation  : validates three cumulative real Online updates while "
        "reusing one already-built base index, with observed zero Online rebuilds, "
        "paired PRIMARY populations, independent RNG streams, and explicit "
        "full_stale/full_rebuilt controls."
    )


if __name__ == "__main__":
    main()
