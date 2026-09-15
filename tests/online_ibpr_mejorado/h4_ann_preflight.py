import inspect
import platform
import sys

import numpy as np
import cornac

from cornac.models import IBPR, OnlineIBPRMejorado
from cornac.models.recommender import ANNMixin, MEASURE_DOT


def load_faiss_ann():
    errors = []
    try:
        from cornac.models import FaissANN
        return FaissANN, "cornac.models.FaissANN"
    except Exception as exc:
        errors.append(f"cornac.models.FaissANN: {exc!r}")

    try:
        from cornac.models.ann import FaissANN
        return FaissANN, "cornac.models.ann.FaissANN"
    except Exception as exc:
        errors.append(f"cornac.models.ann.FaissANN: {exc!r}")

    raise RuntimeError(
        "FaissANN no pudo importarse desde Cornac.\n" + "\n".join(errors)
    )


def check_model_contract(model_cls, label):
    if not issubclass(model_cls, ANNMixin):
        raise RuntimeError(f"{label} no hereda ANNMixin.")

    required = {
        "get_vector_measure",
        "get_user_vectors",
        "get_item_vectors",
        "score",
    }
    missing = [name for name in required if not hasattr(model_cls, name)]
    if missing:
        raise RuntimeError(f"{label}: faltan métodos ANN {missing}.")

    obj = model_cls
    source = inspect.getsource(obj)
    for token in ("MEASURE_DOT", "get_user_vectors", "get_item_vectors"):
        if token not in source:
            raise RuntimeError(f"{label}: contrato ANN esperado no contiene {token}.")


def main():
    print("=" * 100)
    print("H4 ANN PREFLIGHT")
    print("=" * 100)
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"Cornac   : {getattr(cornac, '__version__', 'unavailable')}")
    print()

    check_model_contract(IBPR, "IBPR")
    check_model_contract(OnlineIBPRMejorado, "OnlineIBPRMejorado")
    print("IBPR ANN contract                 : OK")
    print("OnlineIBPRMejorado ANN contract   : OK")

    # Instantiate only to validate the declared metric.
    ibpr_probe = IBPR(k=2, trainable=False, init_params={
        "U": np.ones((2, 2), dtype=np.float32),
        "V": np.ones((4, 2), dtype=np.float32),
    })
    if ibpr_probe.get_vector_measure() != MEASURE_DOT:
        raise RuntimeError("IBPR no declara MEASURE_DOT.")

    online_probe = OnlineIBPRMejorado(
        k=2,
        trainable=False,
        init_params={
            "U": np.ones((2, 2), dtype=np.float32),
            "V": np.ones((4, 2), dtype=np.float32),
        },
        update_V=False,
        seed=42,
    )
    if online_probe.get_vector_measure() != MEASURE_DOT:
        raise RuntimeError("OnlineIBPRMejorado no declara MEASURE_DOT.")

    print("MEASURE_DOT                      : OK")

    try:
        import faiss
    except Exception as exc:
        print()
        print("FAISS IMPORT: FAIL")
        print(repr(exc))
        print()
        print("El preflight no modifica el experimento.")
        print("Instala una distribución faiss-cpu compatible y vuelve a ejecutar.")
        raise SystemExit(2)

    print(f"FAISS version                    : {getattr(faiss, '__version__', 'unavailable')}")

    FaissANN, import_path = load_faiss_ann()
    print(f"FaissANN import                  : OK ({import_path})")

    sig = inspect.signature(FaissANN.__init__)
    params = set(sig.parameters)
    required_params = {"model", "nlist", "nprobe", "use_gpu", "num_threads", "seed"}
    missing = required_params - params
    if missing:
        raise RuntimeError(
            f"FaissANN local no cumple contrato esperado. Faltan: {sorted(missing)}"
        )
    print("FaissANN constructor contract    : OK")

    # Direct FAISS smoke test independent of MovieLens/final models.
    rng = np.random.default_rng(42)
    item_vectors = rng.normal(size=(500, 20)).astype(np.float32)
    query_vectors = rng.normal(size=(8, 20)).astype(np.float32)

    nlist = 10
    quantizer = faiss.IndexFlatIP(item_vectors.shape[1])
    index = faiss.IndexIVFFlat(
        quantizer,
        item_vectors.shape[1],
        nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    if hasattr(index, "cp"):
        index.cp.seed = 42

    index.train(item_vectors)
    index.add(item_vectors)
    index.nprobe = 5

    distances, neighbors = index.search(query_vectors, 20)

    if neighbors.shape != (8, 20):
        raise RuntimeError(f"FAISS neighbors shape inesperado: {neighbors.shape}")
    if distances.shape != (8, 20):
        raise RuntimeError(f"FAISS distances shape inesperado: {distances.shape}")
    if np.any(neighbors < 0):
        raise RuntimeError("FAISS devolvió vecinos inválidos en smoke test.")

    print("FAISS IndexIVFFlat train/add     : OK")
    print("FAISS inner-product query        : OK")
    print()
    print("PREFLIGHT RESULT                 : PASS")
    print()
    print("Este test es sintético: no entrena MovieLens ni genera evidencia H4.")


if __name__ == "__main__":
    main()
