import hashlib
import inspect
import platform
import sys

import numpy as np
import cornac


N_ITEMS = 3505
N_USERS = 32
DIM = 20
NLIST = 80
NPROBE = 40
USE_GPU = False
NUM_THREADS = 1
SEED = 42
K = 20


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_source(obj):
    try:
        return sha256_bytes(inspect.getsource(obj).encode("utf-8"))
    except Exception:
        return "unavailable"


def load_symbols():
    try:
        from cornac.models import IBPR, OnlineIBPRMejorado
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron importar IBPR/OnlineIBPRMejorado: "
            f"{exc!r}"
        )

    try:
        from cornac.models.recommender import ANNMixin, MEASURE_DOT
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo importar ANNMixin/MEASURE_DOT: {exc!r}"
        )

    errors = []
    FaissANN = None
    faiss_path = None

    try:
        from cornac.models import FaissANN as _FaissANN
        FaissANN = _FaissANN
        faiss_path = "cornac.models.FaissANN"
    except Exception as exc:
        errors.append(f"cornac.models.FaissANN: {exc!r}")

    if FaissANN is None:
        try:
            from cornac.models.ann import FaissANN as _FaissANN
            FaissANN = _FaissANN
            faiss_path = "cornac.models.ann.FaissANN"
        except Exception as exc:
            errors.append(f"cornac.models.ann.FaissANN: {exc!r}")

    if FaissANN is None:
        raise RuntimeError(
            "FaissANN no pudo importarse.\n" + "\n".join(errors)
        )

    BaseANN = None
    for module_name in (
        "cornac.models.ann.recom_ann_base",
        "cornac.models.ann.recom_ann",
    ):
        try:
            module = __import__(module_name, fromlist=["BaseANN"])
            BaseANN = getattr(module, "BaseANN")
            break
        except Exception:
            pass

    return (
        IBPR,
        OnlineIBPRMejorado,
        ANNMixin,
        MEASURE_DOT,
        FaissANN,
        BaseANN,
        faiss_path,
    )


def assert_model_contract(model_cls, ANNMixin, label):
    if not issubclass(model_cls, ANNMixin):
        raise RuntimeError(f"{label} no hereda ANNMixin.")

    required = (
        "get_vector_measure",
        "get_user_vectors",
        "get_item_vectors",
        "score",
    )
    missing = [name for name in required if not hasattr(model_cls, name)]
    if missing:
        raise RuntimeError(f"{label}: faltan métodos ANN: {missing}")


def instantiate_faiss_ann(FaissANN, model):
    sig = inspect.signature(FaissANN.__init__)
    params = set(sig.parameters)

    required = {
        "model",
        "nlist",
        "nprobe",
        "use_gpu",
        "num_threads",
    }

    missing = sorted(required - params)
    if missing:
        raise RuntimeError(
            "FaissANN local no cumple la API congelada para H4. "
            f"Faltan parámetros obligatorios: {missing}. "
            f"Firma local: {sig}"
        )

    kwargs = {
        "model": model,
        "nlist": NLIST,
        "nprobe": NPROBE,
        "use_gpu": USE_GPU,
        "num_threads": NUM_THREADS,
    }

    seed_supported = "seed" in params
    if seed_supported:
        kwargs["seed"] = SEED

    ann = FaissANN(**kwargs)
    return ann, sig, seed_supported


def call_build_index(ann):
    if not hasattr(ann, "build_index"):
        raise RuntimeError("FaissANN local no expone build_index().")
    return ann.build_index()


def call_knn_query(ann, query, k):
    if not hasattr(ann, "knn_query"):
        raise RuntimeError("FaissANN local no expone knn_query().")

    result = ann.knn_query(
        np.asarray(query, dtype=np.float32),
        int(k),
    )

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "FaissANN.knn_query() no devolvió una tupla de 2 elementos."
        )

    a, b = result
    a = np.asarray(a)
    b = np.asarray(b)

    if np.issubdtype(a.dtype, np.integer):
        neighbors, distances = a, b
    elif np.issubdtype(b.dtype, np.integer):
        neighbors, distances = b, a
    else:
        raise RuntimeError(
            "No se pudo identificar la salida de IDs de knn_query()."
        )

    return neighbors, distances


def get_faiss_index_object(ann):
    for name in ("index", "_index", "ann_index", "_ann_index"):
        obj = getattr(ann, name, None)
        if obj is not None:
            return obj, name

    for name, value in vars(ann).items():
        if value is None:
            continue
        module = getattr(value.__class__, "__module__", "")
        if module.startswith("faiss"):
            return value, name

    return None, None


def serialize_index(faiss, ann):
    index, attr_name = get_faiss_index_object(ann)
    if index is None:
        return None, attr_name

    try:
        raw = faiss.serialize_index(index)
        raw = np.asarray(raw, dtype=np.uint8).tobytes()
        return raw, attr_name
    except Exception:
        return None, attr_name


def index_hash(faiss, ann):
    raw, attr_name = serialize_index(faiss, ann)
    if raw is None:
        return None, attr_name
    return sha256_bytes(raw), attr_name


def exact_numpy_topk(V, query, k):
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    scores = np.asarray(V, dtype=np.float32).dot(query)

    item_ids = np.arange(V.shape[0], dtype=np.int64)
    order = np.lexsort((item_ids, -scores))
    top = order[:k]

    return item_ids[top], scores[top]


def check_flatip_against_numpy(faiss, V, queries, k):
    flat = faiss.IndexFlatIP(V.shape[1])
    flat.add(np.asarray(V, dtype=np.float32))

    for q_idx, q in enumerate(queries):
        q2 = np.asarray(q, dtype=np.float32).reshape(1, -1)

        distances, neighbors = flat.search(q2, k)
        faiss_ids = neighbors[0]

        if len(faiss_ids) != k or np.any(faiss_ids < 0):
            raise RuntimeError(
                f"IndexFlatIP no devolvió {k} IDs válidos para query {q_idx}."
            )

        np_ids, np_scores = exact_numpy_topk(V, q, k)

        if not np.array_equal(faiss_ids, np_ids):
            faiss_scores = V[faiss_ids].dot(q.reshape(-1))

            if not np.allclose(
                np.sort(faiss_scores)[::-1],
                np.sort(np_scores)[::-1],
                rtol=1e-6,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"IndexFlatIP != NumPy exhaustive para query {q_idx}."
                )


def make_synthetic_factors():
    rng = np.random.default_rng(SEED)

    V = (0.05 * rng.normal(size=(N_ITEMS, DIM))).astype(np.float32)
    U = (0.05 * rng.normal(size=(N_USERS, DIM))).astype(np.float32)

    U[0] = 0.0
    U[0, 0] = 1.0

    U[1] = 0.0
    U[1, 1] = 1.0

    for i in range(40):
        V[i] = 0.0
        V[i, 0] = 10.0 - i * 0.01

    for offset, i in enumerate(range(40, 80)):
        V[i] = 0.0
        V[i, 1] = 10.0 - offset * 0.01

    return U, V


def run_real_model_ann_smoke(
    label,
    model,
    FaissANN,
    faiss,
    threadpool_limits,
):
    ann, sig, seed_supported = instantiate_faiss_ann(
        FaissANN=FaissANN,
        model=model,
    )

    with threadpool_limits(limits=1):
        call_build_index(ann)

    q0 = np.asarray(model.U[0], dtype=np.float32).reshape(1, -1)
    q1 = np.asarray(model.U[1], dtype=np.float32).reshape(1, -1)

    with threadpool_limits(limits=1):
        n0, d0 = call_knn_query(ann, q0, K)
        n1, d1 = call_knn_query(ann, q1, K)

    n0 = np.asarray(n0).reshape(-1)
    n1 = np.asarray(n1).reshape(-1)

    if np.any(n0 < -1) or np.any(n1 < -1):
        raise RuntimeError(f"{label}: IDs inválidos menores que -1.")

    valid0 = n0[n0 >= 0]
    valid1 = n1[n1 >= 0]

    if len(valid0) != K:
        raise RuntimeError(
            f"{label}: query U[0] devolvió {len(valid0)} vecinos válidos; "
            f"se esperaban {K}."
        )

    if len(valid1) != K:
        raise RuntimeError(
            f"{label}: query U[1] devolvió {len(valid1)} vecinos válidos; "
            f"se esperaban {K}."
        )

    if np.array_equal(valid0[:10], valid1[:10]):
        raise RuntimeError(
            f"{label}: queries deliberadamente distintas devolvieron "
            "el mismo top-10; no se valida query dinámica."
        )

    h_before, attr_name = index_hash(faiss, ann)

    ann_identity_before = id(ann)

    with threadpool_limits(limits=1):
        _n_again, _d_again = call_knn_query(ann, q0, K)

    ann_identity_after = id(ann)
    if ann_identity_before != ann_identity_after:
        raise RuntimeError(f"{label}: cambió la identidad del objeto ANN.")

    h_after, _ = index_hash(faiss, ann)

    if h_before is not None and h_after is not None and h_before != h_after:
        raise RuntimeError(
            f"{label}: el índice cambió después de una simple consulta."
        )

    print(f"{label}:")
    print(f"  constructor              : {sig}")
    print(f"  seed supported           : {seed_supported}")
    print(f"  build_index              : OK")
    print(f"  query U[0]               : {len(valid0)}/{K} valid")
    print(f"  query U[1]               : {len(valid1)}/{K} valid")
    print(f"  dynamic query differs    : OK")
    print(f"  same ANN object          : OK")
    print(f"  index attr               : {attr_name}")
    print(f"  index SHA before         : {h_before}")
    print(
        f"  index SHA unchanged      : "
        f"{h_before == h_after if h_before is not None else 'unavailable'}"
    )

    return ann, valid0, d0, h_before, seed_supported


def main():
    print("=" * 110)
    print("H4 ANN PREFLIGHT V3.1")
    print("=" * 110)

    print(f"Python                     : {sys.version.split()[0]}")
    print(f"Platform                   : {platform.platform()}")
    print(f"Cornac                     : {getattr(cornac, '__version__', 'unavailable')}")
    print(f"NumPy                      : {np.__version__}")

    try:
        import scipy
        print(f"SciPy                      : {scipy.__version__}")
    except Exception:
        print("SciPy                      : unavailable")

    try:
        import torch
        print(f"PyTorch                    : {torch.__version__}")
    except Exception:
        print("PyTorch                    : unavailable")

    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except Exception as exc:
        raise RuntimeError(
            f"threadpoolctl es obligatorio para H4: {exc!r}"
        )

    try:
        import faiss
    except Exception as exc:
        print()
        print("FAISS IMPORT: FAIL")
        print(repr(exc))
        raise SystemExit(2)

    print(f"FAISS                      : {getattr(faiss, '__version__', 'unavailable')}")

    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(NUM_THREADS)

    (
        IBPR,
        OnlineIBPRMejorado,
        ANNMixin,
        MEASURE_DOT,
        FaissANN,
        BaseANN,
        faiss_path,
    ) = load_symbols()

    print(f"FaissANN                   : {faiss_path}")

    assert_model_contract(IBPR, ANNMixin, "IBPR")
    assert_model_contract(OnlineIBPRMejorado, ANNMixin, "OnlineIBPRMejorado")

    print()
    print("Frozen synthetic ANN config:")
    print(f"  n_items                  : {N_ITEMS}")
    print(f"  n_users                  : {N_USERS}")
    print(f"  dim                      : {DIM}")
    print(f"  nlist                    : {NLIST}")
    print(f"  nprobe                   : {NPROBE}")
    print(f"  use_gpu                  : {USE_GPU}")
    print(f"  num_threads              : {NUM_THREADS}")
    print(f"  seed                     : {SEED}")

    sig = inspect.signature(FaissANN.__init__)
    required = {"model", "nlist", "nprobe", "use_gpu", "num_threads"}
    missing = sorted(required - set(sig.parameters))

    if missing:
        raise RuntimeError(
            f"FaissANN local no cumple API H4. Faltan: {missing}. "
            f"Firma: {sig}"
        )

    print()
    print("Source fingerprints:")
    print(f"  FaissANN                 : {sha256_source(FaissANN)}")
    print(f"  FaissANN.__init__        : {sha256_source(FaissANN.__init__)}")
    print(
        f"  FaissANN.build_index     : "
        f"{sha256_source(getattr(FaissANN, 'build_index', None))}"
    )
    print(
        f"  FaissANN.knn_query       : "
        f"{sha256_source(getattr(FaissANN, 'knn_query', None))}"
    )
    print(
        f"  BaseANN                  : "
        f"{sha256_source(BaseANN) if BaseANN is not None else 'unavailable'}"
    )

    U, V = make_synthetic_factors()

    print()
    print("Exact retrieval sanity check:")
    check_flatip_against_numpy(
        faiss=faiss,
        V=V,
        queries=(U[0], U[1], U[2]),
        k=K,
    )
    print("  NumPy vs IndexFlatIP     : OK")

    ibpr_model = IBPR(
        k=DIM,
        trainable=False,
        init_params={
            "U": U.copy(),
            "V": V.copy(),
        },
    )

    online_model = OnlineIBPRMejorado(
        k=DIM,
        trainable=False,
        init_params={
            "U": U.copy(),
            "V": V.copy(),
        },
        update_V=False,
        seed=SEED,
    )

    if ibpr_model.get_vector_measure() != MEASURE_DOT:
        raise RuntimeError("IBPR no declara MEASURE_DOT.")

    if online_model.get_vector_measure() != MEASURE_DOT:
        raise RuntimeError("OnlineIBPRMejorado no declara MEASURE_DOT.")

    print()
    print("Real model -> Cornac FaissANN integration:")

    _, _, _, _, ibpr_seed_support = run_real_model_ann_smoke(
        label="IBPR",
        model=ibpr_model,
        FaissANN=FaissANN,
        faiss=faiss,
        threadpool_limits=threadpool_limits,
    )

    _, _, _, _, online_seed_support = run_real_model_ann_smoke(
        label="OnlineIBPRMejorado",
        model=online_model,
        FaissANN=FaissANN,
        faiss=faiss,
        threadpool_limits=threadpool_limits,
    )

    print()
    if not ibpr_seed_support or not online_seed_support:
        print(
            "WARNING: FaissANN local no expone parámetro seed; "
            "reproducibilidad se reportará descriptivamente."
        )

    ann_a, _, _ = instantiate_faiss_ann(FaissANN, online_model)
    ann_b, _, _ = instantiate_faiss_ann(FaissANN, online_model)

    with threadpool_limits(limits=1):
        call_build_index(ann_a)
        call_build_index(ann_b)

        q = np.asarray(
            online_model.U[0],
            dtype=np.float32,
        ).reshape(1, -1)

        n_a, d_a = call_knn_query(ann_a, q, K)
        n_b, d_b = call_knn_query(ann_b, q, K)

    n_a = np.asarray(n_a)
    n_b = np.asarray(n_b)
    d_a = np.asarray(d_a, dtype=np.float64)
    d_b = np.asarray(d_b, dtype=np.float64)

    h_a, _ = index_hash(faiss, ann_a)
    h_b, _ = index_hash(faiss, ann_b)

    print()
    print("Independent-build reproducibility (diagnostic only):")
    print(f"  neighbors equal          : {np.array_equal(n_a, n_b)}")
    print(
        f"  max distance diff        : "
        f"{float(np.max(np.abs(d_a - d_b))):.12g}"
    )
    print(
        f"  index SHA equal          : "
        f"{(h_a == h_b) if h_a and h_b else 'unavailable'}"
    )

    print()
    print("Threadpool fingerprint after all imports:")
    info = threadpool_info()

    if not info:
        print("  (none reported)")
    else:
        for entry in info:
            print(
                f"  user_api={entry.get('user_api')} | "
                f"internal_api={entry.get('internal_api')} | "
                f"num_threads={entry.get('num_threads')} | "
                f"prefix={entry.get('prefix')}"
            )

    print()
    print("Direct FAISS IVFFlat diagnostic:")

    quantizer = faiss.IndexFlatIP(DIM)
    direct = faiss.IndexIVFFlat(
        quantizer,
        DIM,
        NLIST,
        faiss.METRIC_INNER_PRODUCT,
    )

    if hasattr(direct, "cp") and hasattr(direct.cp, "seed"):
        direct.cp.seed = SEED

    with threadpool_limits(limits=1):
        direct.train(V)
        direct.add(V)
        direct.nprobe = NPROBE

        dist, neigh = direct.search(
            np.asarray(
                U[0],
                dtype=np.float32,
            ).reshape(1, -1),
            K,
        )

    if neigh.shape != (1, K):
        raise RuntimeError(
            f"Direct IVFFlat shape inesperado: {neigh.shape}"
        )

    if np.any(neigh < 0):
        raise RuntimeError(
            "Direct IVFFlat no devolvió 20 vecinos válidos."
        )

    print("  train                     : OK")
    print("  add                       : OK")
    print(f"  search                    : {K}/{K} valid")

    print()
    print("=" * 110)
    print("PREFLIGHT RESULT            : PASS")
    print("=" * 110)
    print()
    print("Este preflight es sintético.")
    print("No utiliza MovieLens ni genera evidencia H4.")
    print(
        "La reproducibilidad entre builds independientes es diagnóstica "
        "y no condiciona PASS."
    )


if __name__ == "__main__":
    main()
