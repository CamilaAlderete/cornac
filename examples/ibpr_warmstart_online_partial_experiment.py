import time
from collections import OrderedDict

import numpy as np
import cornac
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR, OnlineIBPRMejorado


SEED = 42
RATING_THRESHOLD = 3.0
VARIANT = "1M"
BASE_FRAC = 0.70
ADAPT_FRAC = 0.10
TEST_FRAC = 0.20
TOP_K = 20


def load_positive_chrono_movielens_1m(rating_threshold=RATING_THRESHOLD):
    """Load MovieLens 1M in chronological order and binarize it for implicit training."""
    data = movielens.load_feedback(fmt="UIRT", variant=VARIANT)
    positive = [
        (str(u), str(i), 1.0, int(ts))
        for u, i, r, ts in data
        if float(r) >= rating_threshold
    ]
    positive.sort(key=lambda x: x[3])
    return positive


def split_base_adapt_test(data, base_frac=BASE_FRAC, adapt_frac=ADAPT_FRAC):
    n = len(data)
    base_end = int(n * base_frac)
    adapt_end = int(n * (base_frac + adapt_frac))
    base_raw = data[:base_end]
    adapt_raw = data[base_end:adapt_end]
    test_raw = data[adapt_end:]
    return base_raw, adapt_raw, test_raw


def filter_to_base_known(base_raw, adapt_raw, test_raw):
    known_users = {u for u, _, _, _ in base_raw}
    known_items = {i for _, i, _, _ in base_raw}

    adapt_filtered = [x for x in adapt_raw if x[0] in known_users and x[1] in known_items]
    test_filtered = [x for x in test_raw if x[0] in known_users and x[1] in known_items]
    return adapt_filtered, test_filtered


def build_datasets(base_raw, adapt_raw, test_raw):
    all_rows = base_raw + adapt_raw + test_raw
    if len(all_rows) == 0:
        raise ValueError("No hay datos luego del filtrado para construir el experimento.")

    global_ds = Dataset.build(all_rows, fmt="UIRT", seed=SEED)
    uid_map = global_ds.uid_map
    iid_map = global_ds.iid_map

    base_train_set = Dataset.build(
        base_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=False,
    )
    adapt_train_set = Dataset.build(
        adapt_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=False,
    )
    combined_train_set = Dataset.build(
        base_raw + adapt_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=False,
    )
    test_set = Dataset.build(
        test_raw,
        fmt="UIRT",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=SEED,
        exclude_unknowns=True,
    )

    return base_train_set, adapt_train_set, combined_train_set, test_set


def build_metrics(k=TOP_K):
    return [
        cornac.metrics.AUC(),
        cornac.metrics.MAP(),
        cornac.metrics.NDCG(k=k),
        cornac.metrics.Precision(k=k),
        cornac.metrics.Recall(k=k),
    ]


def evaluate_ranking(model, train_set, test_set, metrics):
    start = time.perf_counter()
    avg_results, _ = ranking_eval(
        model=model,
        metrics=metrics,
        train_set=train_set,
        test_set=test_set,
        val_set=None,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - start
    return OrderedDict((m.name, r) for m, r in zip(metrics, avg_results)), elapsed


def print_segment_summary(base_raw, adapt_raw, test_raw):
    print("=" * 80)
    print("SEGMENT SUMMARY")
    print("=" * 80)
    print(f"Base segment   : {len(base_raw):,}")
    print(f"Adapt segment  : {len(adapt_raw):,}")
    print(f"Test segment   : {len(test_raw):,}")
    print(f"Users in base  : {len({u for u, _, _, _ in base_raw}):,}")
    print(f"Items in base  : {len({i for _, i, _, _ in base_raw}):,}")
    print()


def print_result_block(title, train_time, test_time, metrics_dict):
    print("-" * 80)
    print(title)
    print("-" * 80)
    print(f"Train time: {train_time:.4f} s")
    print(f"Test time : {test_time:.4f} s")
    for name, value in metrics_dict.items():
        print(f"{name:12s}: {value:.4f}")
    print()


def main():
    data = load_positive_chrono_movielens_1m()
    base_raw, adapt_raw_all, test_raw_all = split_base_adapt_test(data)
    adapt_raw, test_raw = filter_to_base_known(base_raw, adapt_raw_all, test_raw_all)

    if len(adapt_raw) == 0 or len(test_raw) == 0:
        raise ValueError("Los segmentos adapt/test quedaron vacíos luego del filtrado a entidades conocidas.")

    print_segment_summary(base_raw, adapt_raw, test_raw)

    base_train_set, adapt_train_set, combined_train_set, test_set = build_datasets(
        base_raw, adapt_raw, test_raw
    )
    metrics = build_metrics()

    # 1) Train base IBPR on the base segment only
    ibpr = IBPR(
        k=50,
        max_iter=20,
        learning_rate=0.01,
        lamda=0.001,
        batch_size=512,
        verbose=True,
        name="IBPR (base only)",
    )

    t0 = time.perf_counter()
    ibpr.fit(base_train_set)
    ibpr_train_time = time.perf_counter() - t0
    ibpr_metrics, ibpr_test_time = evaluate_ranking(ibpr, combined_train_set, test_set, metrics)

    # 2) Warm-start OnlineIBPRMejorado with U/V from IBPR, then do partial retraining on the adapt segment
    online_partial = OnlineIBPRMejorado(
        k=50,
        max_iter=10,
        learning_rate=0.01,
        lamda=0.001,
        batch_size=512,
        init_params={"U": ibpr.U.copy(), "V": ibpr.V.copy()},
        update_V=False,
        neg_sampling="uniform",
        normalize=True,
        loss_mode="cosine_bpr",
        verbose=True,
        name="OnlineIBPRMejorado (warm-start + partial fit)",
    )

    recent_pairs = np.column_stack(
        (adapt_train_set.uir_tuple[0], adapt_train_set.uir_tuple[1])
    ).astype(np.int64)

    t0 = time.perf_counter()
    online_partial.partial_fit_recent(
        recent_pairs=recent_pairs,
        history_csr=combined_train_set.csr_matrix,
        max_steps=None,
        n_epochs=1,
    )
    online_train_time = time.perf_counter() - t0
    online_metrics, online_test_time = evaluate_ranking(
        online_partial, combined_train_set, test_set, metrics
    )

    print("=" * 80)
    print("RESULTS - WARM-START + PARTIAL RETRAINING ON ADAPT SEGMENT")
    print("=" * 80)
    print_result_block("IBPR trained on base segment", ibpr_train_time, ibpr_test_time, ibpr_metrics)
    print_result_block(
        "OnlineIBPRMejorado warm-started from IBPR and partially retrained on adapt segment",
        online_train_time,
        online_test_time,
        online_metrics,
    )


if __name__ == "__main__":
    main()
