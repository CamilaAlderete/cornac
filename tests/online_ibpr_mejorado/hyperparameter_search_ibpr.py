import argparse
import csv
import itertools
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import torch
import cornac
from cornac.data import Dataset
from cornac.datasets import movielens
from cornac.eval_methods.base_method import ranking_eval
from cornac.models import IBPR
from cornac.utils.Tee import Tee


# ============================================================
# General experimental configuration
# ============================================================

# This HPO script works ONLY on implicit positive feedback.
# MovieLens explicit ratings >= RATING_THRESHOLD are converted to 1.0.
RATING_THRESHOLD = 3.0
VARIANT = "1M"
TOP_K = 20

# Only the first 60% of the chronological positive interactions is used
# during hyperparameter selection. The later 40% remains untouched here.
HPO_END_FRAC = 0.60

# Warm-start validation sufficiency checks.
# We do NOT force a high warm-start percentage, because the thesis scope
# intentionally excludes cold-start. What matters is that the retained
# warm-start evaluation population is still large enough to be meaningful.
MIN_KNOWN_VALIDATION_ROWS = 3000
MIN_VALIDATION_USERS = 100
MIN_VALIDATION_ITEMS = 100

# HPO uses only the first 60% of the globally chronological positive-feedback
# stream. Inside that development horizon, each user's own history is split
# temporally using an expanding-window protocol.
#
# For an eligible user with history h = [i1, i2, ..., in] ordered by time:
#   Fold 1: train first 50%, validate next segment up to 65%
#   Fold 2: train first 65%, validate next segment up to 80%
#   Fold 3: train first 80%, validate remaining interactions
#
# This is intentionally a PER-USER temporal validation protocol, not a global
# prequential protocol. Global prequential evaluation is reserved for H1-H4.
FOLDS = [
    {
        "name": "fold_1",
        "train_end_ratio": 0.50,
        "val_end_ratio": 0.65,
    },
    {
        "name": "fold_2",
        "train_end_ratio": 0.65,
        "val_end_ratio": 0.80,
    },
    {
        "name": "fold_3",
        "train_end_ratio": 0.80,
        "val_end_ratio": 1.00,
    },
]

# A user needs at least seven implicit-positive interactions inside the HPO
# development horizon so that floor-based boundaries leave at least one
# future interaction in every validation segment.
MIN_USER_INTERACTIONS_FOR_VALIDATION = 7

# Search-space rationale:
# - paper/reference region: k=20, lr=0.05, lambda=0.001
# - Cornac defaults: k=20, lr=0.05, lambda=0.001, batch=100
# - pilot region: k=50, lr=0.01, lambda=0.001, batch=512
K_VALUES = [20, 50, 100]
LEARNING_RATE_VALUES = [0.005, 0.01, 0.02, 0.05]
LAMDA_VALUES = [0.0001, 0.001, 0.01]
BATCH_SIZE_VALUES = [100, 256, 512]

# Successive-halving-like budget schedule.
SCREENING_MAX_ITER = 20
MAX_ITER_VALUES = [20, 50, 100]

# Stage A: 20 core configurations total.
# Two are mandatory anchors; the remaining 18 are sampled reproducibly.
DEFAULT_SCREENING_CONFIGS = 20
TOP_STAGE_A = 5
TOP_STAGE_B = 3
TOP_STAGE_C = 2

# These reference configurations are methodological anchors.
# They are always carried through Stage C so they are evaluated under
# max_iter = [20, 50, 100], even if they do not rank among the early top runs.
PROTECTED_CONFIG_SOURCES = {
    "paper_region_plus_cornac_defaults",
    "pilot",
}

# Search / confirmation seeds.
HPO_RANDOM_SEED = 2026
SCREENING_SEED = 42
CONFIRMATION_SEEDS = [42, 123, 2024]

# The primary model-selection metric is fixed before running the HPO.
PRIMARY_METRIC = f"NDCG@{TOP_K}"

QUALITY_METRICS = [
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

TRIAL_FIELDS = [
    "origin_stage",
    "config_id",
    "config_source",
    "seed",
    "fold",
    "k",
    "learning_rate",
    "lamda",
    "batch_size",
    "max_iter",
    "n_train",
    "n_train_users",
    "n_train_items",
    "n_validation_original",
    "n_validation_known",
    "known_validation_fraction",
    "n_validation_users",
    "n_validation_items",
    "train_time_s",
    "eval_time_s",
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
]

SUMMARY_FIELDS = [
    "stage",
    "rank",
    "config_id",
    "config_source",
    "k",
    "learning_rate",
    "lamda",
    "batch_size",
    "max_iter",
    "seeds",
    "folds",
    "n_runs",
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
    "mean_train_time_s",
    "std_train_time_s",
    "mean_eval_time_s",
    "std_eval_time_s",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Temporal hyperparameter search for the implicit-feedback "
            "IBPR base model."
        )
    )

    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Timestamp YYYYMMDD_HHMMSS. If omitted, a new timestamp is "
            "created. Reusing an existing timestamp resumes/skips completed "
            "physical runs."
        ),
    )

    parser.add_argument(
        "--screening-configs",
        type=int,
        default=DEFAULT_SCREENING_CONFIGS,
        help=(
            "Total number of core configurations in Stage A. "
            "Must be >= 2. Default: 20."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Print the temporal folds and sampled Stage-A configurations "
            "without training models."
        ),
    )

    return parser.parse_args()


# ============================================================
# Protocol validation
# ============================================================

def validate_hpo_protocol():
    """
    Validate that:
      - HPO never uses data outside the first HPO_END_FRAC globally;
      - per-user folds are expanding and contiguous;
      - all ratios are valid.
    """
    if not (0.0 < HPO_END_FRAC < 1.0):
        raise ValueError("HPO_END_FRAC debe estar entre 0 y 1.")

    previous_val_end = None

    for fold in FOLDS:
        train_end = float(fold["train_end_ratio"])
        val_end = float(fold["val_end_ratio"])

        if not (0.0 < train_end < val_end <= 1.0):
            raise ValueError(
                f"Fold temporal inválido ({fold['name']}): "
                f"train_end_ratio={train_end}, "
                f"val_end_ratio={val_end}."
            )

        if previous_val_end is not None:
            if not np.isclose(train_end, previous_val_end):
                raise ValueError(
                    "Los folds por usuario deben ser contiguos y expansivos: "
                    "el train_end_ratio de un fold debe coincidir con el "
                    "val_end_ratio del fold anterior."
                )

        previous_val_end = val_end

    if not np.isclose(float(FOLDS[-1]["val_end_ratio"]), 1.0):
        raise ValueError(
            "El último fold debe terminar en el 100% de la historia "
            "individual disponible dentro del horizonte HPO."
        )


def protected_config_ids(configs):
    return [
        config["config_id"]
        for config in configs
        if config["config_source"] in PROTECTED_CONFIG_SOURCES
    ]


def unique_preserving_order(values):
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


# ============================================================
# Data preparation: implicit + chronological
# ============================================================

def load_positive_chrono_movielens(
    variant=VARIANT,
    rating_threshold=RATING_THRESHOLD,
):
    """
    Convert explicit MovieLens feedback into implicit positive feedback.

    Every rating >= rating_threshold becomes an observed positive interaction
    with value 1.0. Lower ratings are not treated as explicit negatives; they
    are simply absent from the positive-feedback dataset.
    """
    data = movielens.load_feedback(fmt="UIRT", variant=variant)

    positive = [
        (str(user_id), str(item_id), 1.0, int(timestamp))
        for user_id, item_id, rating, timestamp in data
        if float(rating) >= rating_threshold
    ]

    positive.sort(key=lambda row: row[3])

    return positive


def build_hpo_pool(all_positive_rows):
    """
    Reserve the globally latest 40% of implicit-positive interactions.

    Only the first HPO_END_FRAC of the globally chronological stream is ever
    visible to hyperparameter selection.
    """
    n_total = len(all_positive_rows)
    hpo_end = int(n_total * HPO_END_FRAC)

    if hpo_end <= 0:
        raise ValueError("El horizonte HPO quedó vacío.")

    return list(all_positive_rows[:hpo_end])


def build_user_histories(hpo_pool_rows):
    histories = {}

    for row in hpo_pool_rows:
        user_id = row[0]
        histories.setdefault(user_id, []).append(row)

    # Explicit ordering makes the per-user temporal assumption auditable,
    # even though hpo_pool_rows is already globally chronological.
    for user_id in histories:
        histories[user_id].sort(key=lambda row: row[3])

    return histories


def user_boundaries(n_interactions):
    """
    Shared integer boundaries for the three folds.

    floor() is intentionally used for all intermediate boundaries so that
    Fold 2 training includes exactly the segment evaluated by Fold 1, and
    Fold 3 training includes exactly the segment evaluated by Fold 2.
    """
    boundaries = [
        int(np.floor(n_interactions * 0.50)),
        int(np.floor(n_interactions * 0.65)),
        int(np.floor(n_interactions * 0.80)),
        n_interactions,
    ]

    return boundaries


def prepare_fold_rows(user_histories, fold):
    """
    Build one per-user temporal warm-start fold.

    Training may include users with short histories, because those interactions
    are still legitimate implicit training evidence. Validation, however, is
    restricted to users with enough history to contribute at least one future
    positive interaction to every fold.

    User warm-start is therefore guaranteed by construction. Item warm-start
    is enforced afterwards by filtering candidate validation interactions to
    items already present in the fold's training set.
    """
    fold_index = next(
        index
        for index, candidate in enumerate(FOLDS)
        if candidate["name"] == fold["name"]
    )

    train_rows = []
    validation_candidate = []

    eligible_users = set()

    for user_id, history in user_histories.items():
        n_interactions = len(history)

        if n_interactions >= MIN_USER_INTERACTIONS_FOR_VALIDATION:
            boundaries = user_boundaries(n_interactions)

            # Defensive guarantee: every eligible user must contribute at
            # least one interaction to each validation interval.
            if not all(
                boundaries[index + 1] > boundaries[index]
                for index in range(3)
            ):
                raise ValueError(
                    f"El usuario {user_id} tiene {n_interactions} "
                    "interacciones pero no produce tres segmentos temporales "
                    "no vacíos con las fronteras configuradas."
                )

            eligible_users.add(user_id)

            train_end = boundaries[fold_index]
            val_end = boundaries[fold_index + 1]

            train_rows.extend(history[:train_end])
            validation_candidate.extend(
                history[train_end:val_end]
            )

        else:
            # Sparse users are never part of validation. They may contribute
            # only past interactions to training, using the same fold ratio.
            train_end = int(
                np.floor(
                    n_interactions
                    * float(fold["train_end_ratio"])
                )
            )

            if n_interactions > 0:
                train_end = max(1, train_end)

            train_rows.extend(history[:train_end])

    if not train_rows:
        raise ValueError(f"{fold['name']}: train quedó vacío.")

    if not validation_candidate:
        raise ValueError(
            f"{fold['name']}: no se generaron interacciones candidatas "
            "de validación por usuario."
        )

    # Sort only for reproducibility/readability. IBPR itself does not require
    # chronological ordering once the fold has been defined.
    train_rows.sort(key=lambda row: row[3])
    validation_candidate.sort(key=lambda row: row[3])

    known_users = {u for u, _, _, _ in train_rows}
    known_items = {i for _, i, _, _ in train_rows}

    # Users should already be known by construction, but keep both conditions
    # explicit so the experiment remains robust to future edits.
    validation_known = [
        row
        for row in validation_candidate
        if row[0] in known_users and row[1] in known_items
    ]

    if not validation_known:
        raise ValueError(
            f"{fold['name']}: validation quedó vacío luego del filtrado "
            "warm-start de ítems."
        )

    validation_users = {
        u
        for u, _, _, _ in validation_known
    }

    validation_items = {
        i
        for _, i, _, _ in validation_known
    }

    train_users = {
        u
        for u, _, _, _ in train_rows
    }

    train_items = {
        i
        for _, i, _, _ in train_rows
    }

    # Per-user temporal integrity check.
    max_train_ts_by_user = {}

    for u, _, _, ts in train_rows:
        previous = max_train_ts_by_user.get(u)
        if previous is None or ts > previous:
            max_train_ts_by_user[u] = ts

    min_val_ts_by_user = {}

    for u, _, _, ts in validation_known:
        previous = min_val_ts_by_user.get(u)
        if previous is None or ts < previous:
            min_val_ts_by_user[u] = ts

    for user_id in validation_users:
        if max_train_ts_by_user[user_id] > min_val_ts_by_user[user_id]:
            raise ValueError(
                f"{fold['name']}: se detectó fuga temporal para el "
                f"usuario {user_id}."
            )

    return {
        "train_rows": train_rows,
        "validation_original": validation_candidate,
        "validation_known": validation_known,
        "eligible_users": eligible_users,
        "validation_users": validation_users,
        "validation_items": validation_items,
        "train_users": train_users,
        "train_items": train_items,
    }


def build_fold_datasets(fold_rows, seed):
    """
    Build fold-local mappings from training only.

    This guarantees that the validation set cannot introduce new users/items
    into the model vocabulary. Validation has already been filtered to
    warm-start known entities.
    """
    train_rows = fold_rows["train_rows"]
    validation_known = fold_rows["validation_known"]

    train_set = Dataset.build(
        train_rows,
        fmt="UIRT",
        seed=seed,
    )

    validation_set = Dataset.build(
        validation_known,
        fmt="UIRT",
        global_uid_map=train_set.uid_map,
        global_iid_map=train_set.iid_map,
        seed=seed,
        exclude_unknowns=True,
    )

    return train_set, validation_set


def build_metrics():
    return [
        cornac.metrics.AUC(),
        cornac.metrics.MAP(),
        cornac.metrics.NDCG(k=TOP_K),
        cornac.metrics.Precision(k=TOP_K),
        cornac.metrics.Recall(k=TOP_K),
    ]


# ============================================================
# Search-space construction
# ============================================================

def config_signature(config):
    return (
        int(config["k"]),
        float(config["learning_rate"]),
        float(config["lamda"]),
        int(config["batch_size"]),
    )


def build_stage_a_configs(total_configs):
    if total_configs < 2:
        raise ValueError("--screening-configs debe ser >= 2.")

    paper_cornac_anchor = {
        "k": 20,
        "learning_rate": 0.05,
        "lamda": 0.001,
        "batch_size": 100,
        "config_source": "paper_region_plus_cornac_defaults",
    }

    pilot_anchor = {
        "k": 50,
        "learning_rate": 0.01,
        "lamda": 0.001,
        "batch_size": 512,
        "config_source": "pilot",
    }

    mandatory = [
        paper_cornac_anchor,
        pilot_anchor,
    ]

    all_configs = []

    for k, learning_rate, lamda, batch_size in itertools.product(
        K_VALUES,
        LEARNING_RATE_VALUES,
        LAMDA_VALUES,
        BATCH_SIZE_VALUES,
    ):
        config = {
            "k": int(k),
            "learning_rate": float(learning_rate),
            "lamda": float(lamda),
            "batch_size": int(batch_size),
            "config_source": "random_search",
        }
        all_configs.append(config)

    mandatory_signatures = {
        config_signature(config)
        for config in mandatory
    }

    available = [
        config
        for config in all_configs
        if config_signature(config) not in mandatory_signatures
    ]

    needed_random = total_configs - len(mandatory)

    if needed_random > len(available):
        raise ValueError(
            f"Se solicitaron {total_configs} configuraciones, pero el "
            f"espacio permite como máximo {len(all_configs)} distintas."
        )

    rng = np.random.default_rng(HPO_RANDOM_SEED)
    selected_indices = rng.choice(
        len(available),
        size=needed_random,
        replace=False,
    )

    selected = mandatory + [
        available[int(index)]
        for index in selected_indices
    ]

    # Stable identifiers make resume and result analysis straightforward.
    output = []

    for index, config in enumerate(selected, start=1):
        row = dict(config)
        row["config_id"] = f"C{index:03d}"
        output.append(row)

    return output


# ============================================================
# Trial persistence / resume
# ============================================================

def append_csv(path, fieldnames, row):
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                field: row.get(field, "")
                for field in fieldnames
            }
        )


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )


def load_existing_trials(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def physical_trial_key(
    config_id,
    k,
    learning_rate,
    lamda,
    batch_size,
    seed,
    fold_name,
    max_iter,
):
    """
    Identify a physical run using its actual hyperparameters as well as
    its human-readable config_id. This prevents unsafe reuse if the sampled
    search plan changes while an old timestamp is reused.
    """
    return (
        str(config_id),
        int(k),
        float(learning_rate),
        float(lamda),
        int(batch_size),
        int(seed),
        str(fold_name),
        int(max_iter),
    )


def config_trial_key(
    config,
    seed,
    fold_name,
    max_iter,
):
    return physical_trial_key(
        config["config_id"],
        config["k"],
        config["learning_rate"],
        config["lamda"],
        config["batch_size"],
        seed,
        fold_name,
        max_iter,
    )


def row_trial_key(row):
    return physical_trial_key(
        row["config_id"],
        row["k"],
        row["learning_rate"],
        row["lamda"],
        row["batch_size"],
        row["seed"],
        row["fold"],
        row["max_iter"],
    )


def existing_trial_lookup(rows):
    lookup = {}

    for row in rows:
        key = row_trial_key(row)

        if key in lookup:
            raise ValueError(
                "Se encontró un trial físico duplicado en el CSV de "
                f"reanudación: {key}"
            )

        lookup[key] = row

    return lookup


# ============================================================
# Training / evaluation
# ============================================================

def evaluate_ranking(model, train_set, validation_set):
    metrics = build_metrics()

    start = time.perf_counter()

    avg_results, _ = ranking_eval(
        model=model,
        metrics=metrics,
        train_set=train_set,
        test_set=validation_set,
        val_set=None,
        # Feedback is implicit: every observed validation interaction is 1.0.
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
    )

    elapsed = time.perf_counter() - start

    return (
        {
            metric.name: float(result)
            for metric, result in zip(metrics, avg_results)
        },
        elapsed,
    )


def run_physical_trial(
    origin_stage,
    config,
    max_iter,
    seed,
    fold,
    fold_rows,
):
    # Make the stochastic training repeatable for this physical run.
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_set, validation_set = build_fold_datasets(
        fold_rows,
        seed=seed,
    )

    model = IBPR(
        name=(
            f"IBPR_HPO_{config['config_id']}_"
            f"{fold['name']}_seed_{seed}_iter_{max_iter}"
        ),
        k=config["k"],
        max_iter=max_iter,
        learning_rate=config["learning_rate"],
        lamda=config["lamda"],
        batch_size=config["batch_size"],
        verbose=False,
    )

    train_start = time.perf_counter()
    model.fit(train_set)
    train_time = time.perf_counter() - train_start

    metric_values, eval_time = evaluate_ranking(
        model,
        train_set,
        validation_set,
    )

    n_original = len(fold_rows["validation_original"])
    n_known = len(fold_rows["validation_known"])

    row = {
        "origin_stage": origin_stage,
        "config_id": config["config_id"],
        "config_source": config["config_source"],
        "seed": seed,
        "fold": fold["name"],
        "k": config["k"],
        "learning_rate": config["learning_rate"],
        "lamda": config["lamda"],
        "batch_size": config["batch_size"],
        "max_iter": max_iter,
        "n_train": len(fold_rows["train_rows"]),
        "n_train_users": len(fold_rows["train_users"]),
        "n_train_items": len(fold_rows["train_items"]),
        "n_validation_original": n_original,
        "n_validation_known": n_known,
        "known_validation_fraction": (
            float(n_known / n_original)
            if n_original > 0
            else float("nan")
        ),
        "n_validation_users": len(
            fold_rows["validation_users"]
        ),
        "n_validation_items": len(
            fold_rows["validation_items"]
        ),
        "train_time_s": train_time,
        "eval_time_s": eval_time,
    }

    row.update(metric_values)

    return row


def ensure_trial(
    origin_stage,
    config,
    max_iter,
    seed,
    fold,
    fold_rows,
    trials_path,
    trial_rows,
    trial_lookup,
):
    key = config_trial_key(
        config,
        seed,
        fold["name"],
        max_iter,
    )

    if key in trial_lookup:
        print(
            "REUSE  "
            f"{config['config_id']} | "
            f"{fold['name']} | seed={seed} | iter={max_iter}"
        )
        return trial_lookup[key]

    print(
        "RUN    "
        f"{config['config_id']} | "
        f"k={config['k']} lr={config['learning_rate']} "
        f"lamda={config['lamda']} batch={config['batch_size']} | "
        f"{fold['name']} | seed={seed} | iter={max_iter}"
    )

    row = run_physical_trial(
        origin_stage=origin_stage,
        config=config,
        max_iter=max_iter,
        seed=seed,
        fold=fold,
        fold_rows=fold_rows,
    )

    append_csv(
        trials_path,
        TRIAL_FIELDS,
        row,
    )

    trial_rows.append(row)
    trial_lookup[key] = row

    print(
        f"       {PRIMARY_METRIC}={float(row[PRIMARY_METRIC]):.6f} | "
        f"train={float(row['train_time_s']):.2f}s | "
        f"known_val={100.0 * float(row['known_validation_fraction']):.2f}%"
    )

    return row


# ============================================================
# Aggregation / ranking
# ============================================================

def numeric_value(row, field):
    return float(row[field])


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return float("nan"), float("nan")

    mean_value = float(np.mean(values))

    if values.size < 2:
        return mean_value, 0.0

    return mean_value, float(np.std(values, ddof=1))


def matching_rows(
    trial_rows,
    config,
    max_iter,
    seeds,
    fold_names,
):
    expected = {
        config_trial_key(
            config,
            seed,
            fold_name,
            max_iter,
        )
        for seed in seeds
        for fold_name in fold_names
    }

    selected = []

    for row in trial_rows:
        key = row_trial_key(row)

        if key in expected:
            selected.append(row)

    if len(selected) != len(expected):
        missing = len(expected) - len(selected)
        raise ValueError(
            f"Faltan {missing} ejecuciones para "
            f"{config['config_id']} / max_iter={max_iter}."
        )

    return selected


def aggregate_configuration(
    stage_name,
    config,
    max_iter,
    seeds,
    fold_names,
    trial_rows,
):
    rows = matching_rows(
        trial_rows=trial_rows,
        config=config,
        max_iter=max_iter,
        seeds=seeds,
        fold_names=fold_names,
    )

    summary = {
        "stage": stage_name,
        "config_id": config["config_id"],
        "config_source": config["config_source"],
        "k": config["k"],
        "learning_rate": config["learning_rate"],
        "lamda": config["lamda"],
        "batch_size": config["batch_size"],
        "max_iter": max_iter,
        "seeds": ",".join(str(seed) for seed in seeds),
        "folds": ",".join(fold_names),
        "n_runs": len(rows),
    }

    for metric in QUALITY_METRICS:
        metric_values = [
            numeric_value(row, metric)
            for row in rows
        ]

        mean_value, std_value = mean_std(metric_values)

        summary[f"mean_{metric}"] = mean_value
        summary[f"std_{metric}"] = std_value

    train_values = [
        numeric_value(row, "train_time_s")
        for row in rows
    ]
    eval_values = [
        numeric_value(row, "eval_time_s")
        for row in rows
    ]

    mean_train, std_train = mean_std(train_values)
    mean_eval, std_eval = mean_std(eval_values)

    summary["mean_train_time_s"] = mean_train
    summary["std_train_time_s"] = std_train
    summary["mean_eval_time_s"] = mean_eval
    summary["std_eval_time_s"] = std_eval

    return summary


def ranking_key(summary):
    """
    Fixed deterministic selection rule:
      1) higher mean NDCG@20
      2) lower NDCG@20 variability
      3) lower mean training time
      4) lower latent dimension
      5) fewer iterations
      6) smaller batch size

    The secondary criteria matter only after the primary metric.
    """
    return (
        -float(summary[f"mean_{PRIMARY_METRIC}"]),
        float(summary[f"std_{PRIMARY_METRIC}"]),
        float(summary["mean_train_time_s"]),
        int(summary["k"]),
        int(summary["max_iter"]),
        int(summary["batch_size"]),
    )


def rank_summaries(summaries):
    ranked = sorted(
        summaries,
        key=ranking_key,
    )

    output = []

    for rank, summary in enumerate(ranked, start=1):
        row = dict(summary)
        row["rank"] = rank
        output.append(row)

    return output


def print_stage_ranking(title, ranked_rows, top_n=None):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    shown = ranked_rows if top_n is None else ranked_rows[:top_n]

    for row in shown:
        print(
            f"#{int(row['rank']):02d} "
            f"{row['config_id']} "
            f"k={row['k']} "
            f"lr={row['learning_rate']} "
            f"lamda={row['lamda']} "
            f"batch={row['batch_size']} "
            f"iter={row['max_iter']} | "
            f"{PRIMARY_METRIC}="
            f"{float(row[f'mean_{PRIMARY_METRIC}']):.6f} "
            f"± {float(row[f'std_{PRIMARY_METRIC}']):.6f} | "
            f"train={float(row['mean_train_time_s']):.2f}s"
        )


# ============================================================
# HPO stages
# ============================================================

def run_stage_a(
    configs,
    fold_data,
    trials_path,
    trial_rows,
    trial_lookup,
):
    fold_names = ["fold_1", "fold_2"]

    for config in configs:
        for fold_name in fold_names:
            fold = next(
                item
                for item in FOLDS
                if item["name"] == fold_name
            )

            ensure_trial(
                origin_stage="stage_a_screening",
                config=config,
                max_iter=SCREENING_MAX_ITER,
                seed=SCREENING_SEED,
                fold=fold,
                fold_rows=fold_data[fold_name],
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

    summaries = [
        aggregate_configuration(
            stage_name="stage_a_screening",
            config=config,
            max_iter=SCREENING_MAX_ITER,
            seeds=[SCREENING_SEED],
            fold_names=fold_names,
            trial_rows=trial_rows,
        )
        for config in configs
    ]

    return rank_summaries(summaries)


def run_stage_b(
    stage_a_ranked,
    config_map,
    fold_data,
    trials_path,
    trial_rows,
    trial_lookup,
):
    protected_ids = protected_config_ids(
        list(config_map.values())
    )

    selected_ids = unique_preserving_order(
        [
            row["config_id"]
            for row in stage_a_ranked[:TOP_STAGE_A]
        ]
        + protected_ids
    )

    fold_3 = next(
        fold
        for fold in FOLDS
        if fold["name"] == "fold_3"
    )

    for config_id in selected_ids:
        config = config_map[config_id]

        ensure_trial(
            origin_stage="stage_b_temporal_confirmation",
            config=config,
            max_iter=SCREENING_MAX_ITER,
            seed=SCREENING_SEED,
            fold=fold_3,
            fold_rows=fold_data["fold_3"],
            trials_path=trials_path,
            trial_rows=trial_rows,
            trial_lookup=trial_lookup,
        )

    summaries = [
        aggregate_configuration(
            stage_name="stage_b_temporal_confirmation",
            config=config_map[config_id],
            max_iter=SCREENING_MAX_ITER,
            seeds=[SCREENING_SEED],
            fold_names=["fold_1", "fold_2", "fold_3"],
            trial_rows=trial_rows,
        )
        for config_id in selected_ids
    ]

    return rank_summaries(summaries)


def run_stage_c(
    stage_b_ranked,
    config_map,
    fold_data,
    trials_path,
    trial_rows,
    trial_lookup,
):
    protected_ids = protected_config_ids(
        list(config_map.values())
    )

    selected_ids = unique_preserving_order(
        [
            row["config_id"]
            for row in stage_b_ranked[:TOP_STAGE_B]
        ]
        + protected_ids
    )

    for config_id in selected_ids:
        config = config_map[config_id]

        for max_iter in MAX_ITER_VALUES:
            for fold in FOLDS:
                ensure_trial(
                    origin_stage="stage_c_iteration_budget",
                    config=config,
                    max_iter=max_iter,
                    seed=SCREENING_SEED,
                    fold=fold,
                    fold_rows=fold_data[fold["name"]],
                    trials_path=trials_path,
                    trial_rows=trial_rows,
                    trial_lookup=trial_lookup,
                )

    summaries = []

    for config_id in selected_ids:
        config = config_map[config_id]

        for max_iter in MAX_ITER_VALUES:
            summaries.append(
                aggregate_configuration(
                    stage_name="stage_c_iteration_budget",
                    config=config,
                    max_iter=max_iter,
                    seeds=[SCREENING_SEED],
                    fold_names=[
                        "fold_1",
                        "fold_2",
                        "fold_3",
                    ],
                    trial_rows=trial_rows,
                )
            )

    return rank_summaries(summaries)


def run_stage_d(
    stage_c_ranked,
    config_map,
    fold_data,
    trials_path,
    trial_rows,
    trial_lookup,
):
    finalists = stage_c_ranked[:TOP_STAGE_C]

    for finalist in finalists:
        config = config_map[finalist["config_id"]]
        max_iter = int(finalist["max_iter"])

        for seed in CONFIRMATION_SEEDS:
            for fold in FOLDS:
                ensure_trial(
                    origin_stage="stage_d_multiseed_confirmation",
                    config=config,
                    max_iter=max_iter,
                    seed=seed,
                    fold=fold,
                    fold_rows=fold_data[fold["name"]],
                    trials_path=trials_path,
                    trial_rows=trial_rows,
                    trial_lookup=trial_lookup,
                )

    summaries = []

    for finalist in finalists:
        config = config_map[finalist["config_id"]]

        summaries.append(
            aggregate_configuration(
                stage_name="stage_d_multiseed_confirmation",
                config=config,
                max_iter=int(finalist["max_iter"]),
                seeds=CONFIRMATION_SEEDS,
                fold_names=[
                    "fold_1",
                    "fold_2",
                    "fold_3",
                ],
                trial_rows=trial_rows,
            )
        )

    return rank_summaries(summaries)


# ============================================================
# Reporting helpers
# ============================================================

def print_data_protocol(
    all_positive_rows,
    hpo_pool_rows,
    user_histories,
    fold_data,
):
    print("=" * 100)
    print(
        "IBPR BASE HYPERPARAMETER SEARCH - "
        "PER-USER TEMPORAL IMPLICIT-FEEDBACK PROTOCOL"
    )
    print("=" * 100)
    print(f"Dataset variant                 : MovieLens {VARIANT}")
    print(f"Positive threshold              : rating >= {RATING_THRESHOLD}")
    print("Feedback used by IBPR           : implicit positive (value = 1.0)")
    print(f"Primary selection metric        : {PRIMARY_METRIC}")
    print(f"Global HPO horizon              : first {100 * HPO_END_FRAC:.0f}%")
    print(
        "Later global data used here     : NO "
        "(60-100% remains outside HPO)"
    )
    print(f"Total implicit-positive rows    : {len(all_positive_rows):,}")
    print(f"Rows inside HPO horizon         : {len(hpo_pool_rows):,}")
    print(f"Users inside HPO horizon        : {len(user_histories):,}")

    eligible_user_count = sum(
        1
        for history in user_histories.values()
        if len(history) >= MIN_USER_INTERACTIONS_FOR_VALIDATION
    )

    print(
        "Users eligible for validation   : "
        f"{eligible_user_count:,} "
        f"(>= {MIN_USER_INTERACTIONS_FOR_VALIDATION} positives)"
    )
    print()
    print("Per-user temporal folds:")

    for fold in FOLDS:
        data = fold_data[fold["name"]]

        n_candidate = len(data["validation_original"])
        n_known = len(data["validation_known"])

        known_fraction = (
            n_known / n_candidate
            if n_candidate > 0
            else float("nan")
        )

        print(
            f"  {fold['name']}: "
            f"user-train first "
            f"{100 * fold['train_end_ratio']:.0f}% | "
            f"user-validation until "
            f"{100 * fold['val_end_ratio']:.0f}% | "
            f"train={len(data['train_rows']):,} "
            f"({len(data['train_users']):,} users, "
            f"{len(data['train_items']):,} items) | "
            f"val_candidate={n_candidate:,} | "
            f"val_known={n_known:,} "
            f"({100 * known_fraction:.2f}%) | "
            f"eval_users={len(data['validation_users']):,} | "
            f"eval_items={len(data['validation_items']):,}"
        )

    print()
    print(
        "Scope interpretation             : user warm-start is guaranteed "
        "by construction; candidate validation items not present in fold "
        "training are excluded."
    )
    print(
        "Temporal interpretation          : chronology is enforced within "
        "each user's history; this is not the global prequential protocol "
        "used later for H1-H4."
    )
    print()


def print_search_plan(configs):
    print("=" * 100)
    print("STAGE-A CORE CONFIGURATIONS")
    print("=" * 100)

    for config in configs:
        print(
            f"{config['config_id']} | "
            f"source={config['config_source']:<34s} | "
            f"k={config['k']:<3d} "
            f"lr={config['learning_rate']:<6g} "
            f"lamda={config['lamda']:<7g} "
            f"batch={config['batch_size']}"
        )

    print()
    protected_ids = protected_config_ids(configs)
    print(
        "Protected through Stage C: "
        + ", ".join(protected_ids)
    )
    print()
    print("Budget schedule:")
    print(
        f"  Stage A: {len(configs)} configs × folds 1-2 × "
        f"seed {SCREENING_SEED} × {SCREENING_MAX_ITER} iter"
    )
    print(
        f"  Stage B: top {TOP_STAGE_A} + protected anchors × fold 3 × "
        f"seed {SCREENING_SEED} × {SCREENING_MAX_ITER} iter"
    )
    print(
        f"  Stage C: top {TOP_STAGE_B} + protected anchors × folds 1-3 × "
        f"max_iter={MAX_ITER_VALUES}"
    )
    print(
        f"  Stage D: top {TOP_STAGE_C} full configs × folds 1-3 × "
        f"seeds={CONFIRMATION_SEEDS}"
    )
    print()


def final_best_row(stage_d_ranked):
    best = dict(stage_d_ranked[0])

    best["selection_metric"] = PRIMARY_METRIC
    best["feedback_type"] = "implicit_positive"
    best["rating_threshold"] = RATING_THRESHOLD
    best["dataset_variant"] = VARIANT
    best["hpo_end_fraction"] = HPO_END_FRAC

    return best


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    validate_hpo_protocol()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = (
        args.timestamp
        if args.timestamp
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    trials_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_ibpr_trials_{timestamp}.csv",
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_ibpr_summary_{timestamp}.csv",
    )

    best_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_ibpr_best_{timestamp}.csv",
    )

    log_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_ibpr_{timestamp}.txt",
    )

    log_mode = "a" if os.path.exists(log_path) else "w"

    with open(log_path, log_mode, encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)

        with redirect_stdout(tee):
            print()
            print("#" * 100)
            print(f"HPO TIMESTAMP: {timestamp}")
            print("#" * 100)
            print(f"Trials CSV : {trials_path}")
            print(f"Summary CSV: {summary_path}")
            print(f"Best CSV   : {best_path}")
            print()

            all_positive_rows = load_positive_chrono_movielens()

            hpo_pool_rows = build_hpo_pool(
                all_positive_rows
            )

            user_histories = build_user_histories(
                hpo_pool_rows
            )

            fold_data = {
                fold["name"]: prepare_fold_rows(
                    user_histories,
                    fold,
                )
                for fold in FOLDS
            }

            configs = build_stage_a_configs(
                args.screening_configs
            )

            config_map = {
                config["config_id"]: config
                for config in configs
            }

            print_data_protocol(
                all_positive_rows,
                hpo_pool_rows,
                user_histories,
                fold_data,
            )

            print_search_plan(configs)

            if args.plan_only:
                print("PLAN ONLY: no se entrenó ningún modelo.")
                return

            trial_rows = load_existing_trials(
                trials_path
            )

            trial_lookup = existing_trial_lookup(
                trial_rows
            )

            if trial_rows:
                print(
                    f"Resume: {len(trial_rows)} physical runs "
                    "already present and eligible for reuse."
                )
                print()

            # ------------------------------------------------
            # Stage A
            # ------------------------------------------------
            stage_a_ranked = run_stage_a(
                configs=configs,
                fold_data=fold_data,
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

            print_stage_ranking(
                "STAGE A - SCREENING RANKING",
                stage_a_ranked,
            )

            # ------------------------------------------------
            # Stage B
            # ------------------------------------------------
            stage_b_ranked = run_stage_b(
                stage_a_ranked=stage_a_ranked,
                config_map=config_map,
                fold_data=fold_data,
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

            print_stage_ranking(
                "STAGE B - THREE-FOLD TEMPORAL RANKING",
                stage_b_ranked,
            )

            # ------------------------------------------------
            # Stage C
            # ------------------------------------------------
            stage_c_ranked = run_stage_c(
                stage_b_ranked=stage_b_ranked,
                config_map=config_map,
                fold_data=fold_data,
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

            print_stage_ranking(
                "STAGE C - ITERATION-BUDGET RANKING",
                stage_c_ranked,
            )

            # ------------------------------------------------
            # Stage D
            # ------------------------------------------------
            stage_d_ranked = run_stage_d(
                stage_c_ranked=stage_c_ranked,
                config_map=config_map,
                fold_data=fold_data,
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

            print_stage_ranking(
                "STAGE D - MULTI-SEED FINAL CONFIRMATION",
                stage_d_ranked,
            )

            # Save one consolidated ranking file for all stages.
            all_summary_rows = (
                stage_a_ranked
                + stage_b_ranked
                + stage_c_ranked
                + stage_d_ranked
            )

            save_csv(
                summary_path,
                SUMMARY_FIELDS,
                all_summary_rows,
            )

            best = final_best_row(
                stage_d_ranked
            )

            best_fields = list(SUMMARY_FIELDS) + [
                "selection_metric",
                "feedback_type",
                "rating_threshold",
                "dataset_variant",
                "hpo_end_fraction",
            ]

            save_csv(
                best_path,
                best_fields,
                [best],
            )

            print()
            print("=" * 100)
            print("SELECTED IBPR CONFIGURATION")
            print("=" * 100)
            print(
                "IBPR_CONFIG = {\n"
                f'    "k": {int(best["k"])},\n'
                f'    "max_iter": {int(best["max_iter"])},\n'
                f'    "learning_rate": {float(best["learning_rate"])},\n'
                f'    "lamda": {float(best["lamda"])},\n'
                f'    "batch_size": {int(best["batch_size"])},\n'
                '    "verbose": True,\n'
                "}"
            )
            print()
            print(
                f"Final confirmation {PRIMARY_METRIC}: "
                f"{float(best[f'mean_{PRIMARY_METRIC}']):.6f} "
                f"± {float(best[f'std_{PRIMARY_METRIC}']):.6f}"
            )
            print()
            print(
                "Important: this script selected the IBPR base only. "
                "Do not tune OnlineIBPRMejorado parameters here."
            )
            print(
                "Validation scope: per-user temporal warm-start validation "
                "inside the globally earliest 60% development horizon."
            )


if __name__ == "__main__":
    main()
