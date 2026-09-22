import argparse
import csv
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np

try:
    import hyperparameter_search_ibpr as base
except ImportError:
    # Fallback useful if the validated per-user HPO script was not renamed.
    import hyperparameter_search_ibpr_per_user_temporal as base

from cornac.utils.Tee import Tee


# ============================================================
# Local refinement objective
# ============================================================

# Winner of the first HPO run (timestamp 20260901_213038).
INITIAL_HPO_WINNER = {
    "config_id": "R000",
    "config_source": "initial_hpo_winner_20260901_213038",
    "refinement_group": "baseline",
    "k": 20,
    "learning_rate": 0.005,
    "lamda": 0.0001,
    "batch_size": 512,
    "max_iter": 20,
}

# Fixed evaluation protocol inherited from the validated base-HPO script.
PRIMARY_METRIC = base.PRIMARY_METRIC
QUALITY_METRICS = list(base.QUALITY_METRICS)
FOLDS = list(base.FOLDS)

SCREENING_SEED = 42
CONFIRMATION_SEEDS = [42, 123, 2024]

TOP_STAGE_R1 = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# The local search deliberately crosses every boundary touched by the first HPO.
K_PROBES = [5, 10, 30]
LAMDA_PROBES = [0.00001, 0.00005, 0.0005]
BATCH_PROBES = [256, 1024, 2048]

# learning_rate and max_iter are evaluated jointly because a smaller learning
# rate may require a larger optimization budget to be assessed fairly.
LR_VALUES = [0.001, 0.0025, 0.005, 0.01]
ITER_VALUES = [10, 20, 50]

# Expanded local boundaries, used only to flag whether the final winner again
# lands on an outer edge and therefore deserves manual review.
EXPANDED_BOUNDS = {
    "k": (5, 30),
    "learning_rate": (0.001, 0.01),
    "lamda": (0.00001, 0.0005),
    "batch_size": (256, 2048),
    "max_iter": (10, 50),
}

TRIAL_FIELDS = [
    "origin_stage",
    "config_id",
    "config_source",
    "refinement_group",
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
    f"NDCG@{base.TOP_K}",
    f"Precision@{base.TOP_K}",
    f"Recall@{base.TOP_K}",
]

SUMMARY_FIELDS = [
    "stage",
    "rank",
    "config_id",
    "config_source",
    "refinement_group",
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
    f"mean_NDCG@{base.TOP_K}",
    f"std_NDCG@{base.TOP_K}",
    f"mean_Precision@{base.TOP_K}",
    f"std_Precision@{base.TOP_K}",
    f"mean_Recall@{base.TOP_K}",
    f"std_Recall@{base.TOP_K}",
    "mean_train_time_s",
    "std_train_time_s",
    "mean_eval_time_s",
    "std_eval_time_s",
    "delta_ndcg_vs_initial_baseline",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Targeted local refinement around the first IBPR HPO winner. "
            "Uses the same per-user temporal warm-start validation protocol."
        )
    )

    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Timestamp YYYYMMDD_HHMMSS. Reusing the same timestamp resumes "
            "completed physical trials."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the refinement plan without fitting models.",
    )

    return parser.parse_args()


# ============================================================
# Candidate construction
# ============================================================

def full_signature(config):
    return (
        int(config["k"]),
        float(config["learning_rate"]),
        float(config["lamda"]),
        int(config["batch_size"]),
        int(config["max_iter"]),
    )


def structural_signature(config):
    return (
        int(config["k"]),
        float(config["learning_rate"]),
        float(config["lamda"]),
        int(config["batch_size"]),
    )


def clone_candidate(base_config, **changes):
    config = dict(base_config)
    config.update(changes)
    return config


def build_stage_r1_candidates():
    """
    Targeted local design.

    Apart from the baseline:
      - k is probed below and locally above 20;
      - lambda is probed below and locally above 1e-4;
      - batch_size is probed below and above 512;
      - learning_rate x max_iter is crossed explicitly.

    This is not a second global HPO. It is a boundary/refinement experiment.
    """
    candidates = [dict(INITIAL_HPO_WINNER)]
    counter = 1

    for k in K_PROBES:
        candidates.append(
            clone_candidate(
                INITIAL_HPO_WINNER,
                config_id=f"R{counter:03d}",
                config_source="local_boundary_probe",
                refinement_group="k_probe",
                k=int(k),
            )
        )
        counter += 1

    for lamda in LAMDA_PROBES:
        candidates.append(
            clone_candidate(
                INITIAL_HPO_WINNER,
                config_id=f"R{counter:03d}",
                config_source="local_boundary_probe",
                refinement_group="lamda_probe",
                lamda=float(lamda),
            )
        )
        counter += 1

    for batch_size in BATCH_PROBES:
        candidates.append(
            clone_candidate(
                INITIAL_HPO_WINNER,
                config_id=f"R{counter:03d}",
                config_source="local_boundary_probe",
                refinement_group="batch_probe",
                batch_size=int(batch_size),
            )
        )
        counter += 1

    baseline_sig = full_signature(INITIAL_HPO_WINNER)

    for learning_rate in LR_VALUES:
        for max_iter in ITER_VALUES:
            candidate = clone_candidate(
                INITIAL_HPO_WINNER,
                config_id=f"R{counter:03d}",
                config_source="lr_iter_interaction_probe",
                refinement_group="lr_x_iter_probe",
                learning_rate=float(learning_rate),
                max_iter=int(max_iter),
            )

            if full_signature(candidate) == baseline_sig:
                continue

            candidates.append(candidate)
            counter += 1

    # Defensive uniqueness check.
    seen = set()

    for candidate in candidates:
        sig = full_signature(candidate)

        if sig in seen:
            raise ValueError(
                f"Configuración duplicada detectada en refinamiento: {sig}"
            )

        seen.add(sig)

    return candidates


def candidate_map(candidates):
    return {
        candidate["config_id"]: candidate
        for candidate in candidates
    }


# ============================================================
# CSV / resume helpers
# ============================================================

def append_csv(path, fieldnames, row):
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not exists:
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


def load_csv(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def trial_key_from_config(config, seed, fold_name):
    return (
        str(config["config_id"]),
        int(config["k"]),
        float(config["learning_rate"]),
        float(config["lamda"]),
        int(config["batch_size"]),
        int(config["max_iter"]),
        int(seed),
        str(fold_name),
    )


def trial_key_from_row(row):
    return (
        str(row["config_id"]),
        int(row["k"]),
        float(row["learning_rate"]),
        float(row["lamda"]),
        int(row["batch_size"]),
        int(row["max_iter"]),
        int(row["seed"]),
        str(row["fold"]),
    )


def build_trial_lookup(rows):
    lookup = {}

    for row in rows:
        key = trial_key_from_row(row)

        if key in lookup:
            raise ValueError(
                f"Trial duplicado en CSV de reanudación: {key}"
            )

        lookup[key] = row

    return lookup


# ============================================================
# Physical trials
# ============================================================

def ensure_trial(
    origin_stage,
    config,
    seed,
    fold,
    fold_rows,
    trials_path,
    trial_rows,
    trial_lookup,
):
    key = trial_key_from_config(
        config,
        seed,
        fold["name"],
    )

    if key in trial_lookup:
        print(
            "REUSE  "
            f"{config['config_id']} | "
            f"{fold['name']} | seed={seed} | "
            f"iter={config['max_iter']}"
        )
        return trial_lookup[key]

    print(
        "RUN    "
        f"{config['config_id']} | "
        f"group={config['refinement_group']} | "
        f"k={config['k']} "
        f"lr={config['learning_rate']} "
        f"lamda={config['lamda']} "
        f"batch={config['batch_size']} "
        f"iter={config['max_iter']} | "
        f"{fold['name']} | seed={seed}"
    )

    base_config = {
        "config_id": config["config_id"],
        "config_source": config["config_source"],
        "k": config["k"],
        "learning_rate": config["learning_rate"],
        "lamda": config["lamda"],
        "batch_size": config["batch_size"],
    }

    row = base.run_physical_trial(
        origin_stage=origin_stage,
        config=base_config,
        max_iter=config["max_iter"],
        seed=seed,
        fold=fold,
        fold_rows=fold_rows,
    )

    row["refinement_group"] = config["refinement_group"]

    append_csv(
        trials_path,
        TRIAL_FIELDS,
        row,
    )

    trial_rows.append(row)
    trial_lookup[key] = row

    print(
        f"       {PRIMARY_METRIC}="
        f"{float(row[PRIMARY_METRIC]):.6f} | "
        f"train={float(row['train_time_s']):.2f}s"
    )

    return row


# ============================================================
# Aggregation / ranking
# ============================================================

def mean_std(values):
    values = np.asarray(values, dtype=np.float64)

    mean_value = float(np.mean(values))

    if len(values) < 2:
        return mean_value, 0.0

    return mean_value, float(np.std(values, ddof=1))


def matching_rows(
    trial_rows,
    config,
    seeds,
    fold_names,
):
    expected = {
        trial_key_from_config(
            config,
            seed,
            fold_name,
        )
        for seed in seeds
        for fold_name in fold_names
    }

    selected = [
        row
        for row in trial_rows
        if trial_key_from_row(row) in expected
    ]

    if len(selected) != len(expected):
        raise ValueError(
            f"Faltan ejecuciones para {config['config_id']}: "
            f"esperadas={len(expected)}, encontradas={len(selected)}."
        )

    return selected


def aggregate_config(
    stage,
    config,
    seeds,
    fold_names,
    trial_rows,
):
    rows = matching_rows(
        trial_rows,
        config,
        seeds,
        fold_names,
    )

    summary = {
        "stage": stage,
        "config_id": config["config_id"],
        "config_source": config["config_source"],
        "refinement_group": config["refinement_group"],
        "k": config["k"],
        "learning_rate": config["learning_rate"],
        "lamda": config["lamda"],
        "batch_size": config["batch_size"],
        "max_iter": config["max_iter"],
        "seeds": ",".join(str(seed) for seed in seeds),
        "folds": ",".join(fold_names),
        "n_runs": len(rows),
    }

    for metric in QUALITY_METRICS:
        mean_value, std_value = mean_std(
            [float(row[metric]) for row in rows]
        )
        summary[f"mean_{metric}"] = mean_value
        summary[f"std_{metric}"] = std_value

    mean_train, std_train = mean_std(
        [float(row["train_time_s"]) for row in rows]
    )
    mean_eval, std_eval = mean_std(
        [float(row["eval_time_s"]) for row in rows]
    )

    summary["mean_train_time_s"] = mean_train
    summary["std_train_time_s"] = std_train
    summary["mean_eval_time_s"] = mean_eval
    summary["std_eval_time_s"] = std_eval
    summary["delta_ndcg_vs_initial_baseline"] = ""

    return summary


def ranking_key(row):
    return (
        -float(row[f"mean_{PRIMARY_METRIC}"]),
        float(row[f"std_{PRIMARY_METRIC}"]),
        float(row["mean_train_time_s"]),
        int(row["k"]),
        int(row["max_iter"]),
    )


def rank_rows(rows):
    ranked = sorted(rows, key=ranking_key)

    output = []

    for index, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = index
        output.append(item)

    return output


def add_baseline_deltas(ranked):
    baseline_rows = [
        row
        for row in ranked
        if row["config_id"] == INITIAL_HPO_WINNER["config_id"]
    ]

    if not baseline_rows:
        return ranked

    baseline_ndcg = float(
        baseline_rows[0][f"mean_{PRIMARY_METRIC}"]
    )

    output = []

    for row in ranked:
        item = dict(row)
        item["delta_ndcg_vs_initial_baseline"] = (
            float(item[f"mean_{PRIMARY_METRIC}"])
            - baseline_ndcg
        )
        output.append(item)

    return output


def print_ranking(title, ranked):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    for row in ranked:
        delta = row.get(
            "delta_ndcg_vs_initial_baseline",
            "",
        )

        delta_text = (
            f" | Δbaseline={float(delta):+.6f}"
            if delta != ""
            else ""
        )

        print(
            f"#{int(row['rank']):02d} "
            f"{row['config_id']} "
            f"[{row['refinement_group']}] | "
            f"k={row['k']} "
            f"lr={row['learning_rate']} "
            f"lamda={row['lamda']} "
            f"batch={row['batch_size']} "
            f"iter={row['max_iter']} | "
            f"{PRIMARY_METRIC}="
            f"{float(row[f'mean_{PRIMARY_METRIC}']):.6f} "
            f"± {float(row[f'std_{PRIMARY_METRIC}']):.6f}"
            f"{delta_text}"
        )


# ============================================================
# Stage R1
# ============================================================

def run_stage_r1(
    candidates,
    fold_data,
    trials_path,
    trial_rows,
    trial_lookup,
):
    fold_names = ["fold_1", "fold_2"]

    for config in candidates:
        for fold_name in fold_names:
            fold = next(
                item
                for item in FOLDS
                if item["name"] == fold_name
            )

            ensure_trial(
                origin_stage="refinement_r1_screening",
                config=config,
                seed=SCREENING_SEED,
                fold=fold,
                fold_rows=fold_data[fold_name],
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

    summaries = [
        aggregate_config(
            stage="refinement_r1_screening",
            config=config,
            seeds=[SCREENING_SEED],
            fold_names=fold_names,
            trial_rows=trial_rows,
        )
        for config in candidates
    ]

    ranked = rank_rows(summaries)
    return add_baseline_deltas(ranked)


# ============================================================
# Group winners + combined candidate
# ============================================================

def best_config_id_for_group(
    stage_r1_ranked,
    groups,
):
    valid = [
        row
        for row in stage_r1_ranked
        if row["refinement_group"] in groups
    ]

    if not valid:
        raise ValueError(
            f"No hay candidatos para grupos: {groups}"
        )

    return valid[0]["config_id"]


def build_combined_candidate(
    stage_r1_ranked,
    configs,
):
    cmap = candidate_map(configs)

    best_k = cmap[
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "k_probe"},
        )
    ]

    best_lamda = cmap[
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "lamda_probe"},
        )
    ]

    best_batch = cmap[
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "batch_probe"},
        )
    ]

    best_lr_iter = cmap[
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "lr_x_iter_probe"},
        )
    ]

    combined = {
        "config_id": "R900",
        "config_source": "combined_from_group_winners",
        "refinement_group": "combined_best",
        "k": int(best_k["k"]),
        "learning_rate": float(
            best_lr_iter["learning_rate"]
        ),
        "lamda": float(best_lamda["lamda"]),
        "batch_size": int(best_batch["batch_size"]),
        "max_iter": int(best_lr_iter["max_iter"]),
    }

    # If the combination is already an existing candidate, there is no need
    # to create a duplicate physical configuration.
    for config in configs:
        if full_signature(config) == full_signature(combined):
            return config, {
                "best_k_source": best_k["config_id"],
                "best_lamda_source": best_lamda["config_id"],
                "best_batch_source": best_batch["config_id"],
                "best_lr_iter_source": best_lr_iter["config_id"],
                "combined_was_existing": True,
            }

    return combined, {
        "best_k_source": best_k["config_id"],
        "best_lamda_source": best_lamda["config_id"],
        "best_batch_source": best_batch["config_id"],
        "best_lr_iter_source": best_lr_iter["config_id"],
        "combined_was_existing": False,
    }


# ============================================================
# Stage R2
# ============================================================

def unique_preserving_order(values):
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def run_stage_r2(
    stage_r1_ranked,
    initial_candidates,
    combined_candidate,
    fold_data,
    trials_path,
    trial_rows,
    trial_lookup,
):
    all_candidates = list(initial_candidates)

    if combined_candidate["config_id"] not in {
        item["config_id"]
        for item in all_candidates
    }:
        all_candidates.append(combined_candidate)

    cmap = candidate_map(all_candidates)

    group_winner_ids = [
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "k_probe"},
        ),
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "lamda_probe"},
        ),
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "batch_probe"},
        ),
        best_config_id_for_group(
            stage_r1_ranked,
            {"baseline", "lr_x_iter_probe"},
        ),
    ]

    selected_ids = unique_preserving_order(
        [
            row["config_id"]
            for row in stage_r1_ranked[:TOP_STAGE_R1]
        ]
        + group_winner_ids
        + [INITIAL_HPO_WINNER["config_id"]]
        + [combined_candidate["config_id"]]
    )

    # The combined configuration is new, so it needs folds 1 and 2 as well.
    if combined_candidate["config_id"] == "R900":
        for fold_name in ["fold_1", "fold_2"]:
            fold = next(
                item
                for item in FOLDS
                if item["name"] == fold_name
            )

            ensure_trial(
                origin_stage="refinement_r2_temporal_confirmation",
                config=combined_candidate,
                seed=SCREENING_SEED,
                fold=fold,
                fold_rows=fold_data[fold_name],
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

    fold_3 = next(
        item
        for item in FOLDS
        if item["name"] == "fold_3"
    )

    for config_id in selected_ids:
        config = cmap[config_id]

        ensure_trial(
            origin_stage="refinement_r2_temporal_confirmation",
            config=config,
            seed=SCREENING_SEED,
            fold=fold_3,
            fold_rows=fold_data["fold_3"],
            trials_path=trials_path,
            trial_rows=trial_rows,
            trial_lookup=trial_lookup,
        )

    summaries = [
        aggregate_config(
            stage="refinement_r2_temporal_confirmation",
            config=cmap[config_id],
            seeds=[SCREENING_SEED],
            fold_names=["fold_1", "fold_2", "fold_3"],
            trial_rows=trial_rows,
        )
        for config_id in selected_ids
    ]

    ranked = rank_rows(summaries)
    ranked = add_baseline_deltas(ranked)

    return ranked, all_candidates


# ============================================================
# Stage R3
# ============================================================

def choose_finalists(stage_r2_ranked, all_candidates):
    cmap = candidate_map(all_candidates)

    best_id = stage_r2_ranked[0]["config_id"]
    best_config = cmap[best_id]
    best_structure = structural_signature(best_config)

    alternative_id = None

    for row in stage_r2_ranked[1:]:
        candidate = cmap[row["config_id"]]

        if structural_signature(candidate) != best_structure:
            alternative_id = candidate["config_id"]
            break

    finalist_ids = [best_id]

    if alternative_id is not None:
        finalist_ids.append(alternative_id)

    # Baseline is always reconfirmed under exactly the same seeds/folds so the
    # refinement can be compared directly with the first-HPO winner.
    finalist_ids.append(
        INITIAL_HPO_WINNER["config_id"]
    )

    return unique_preserving_order(finalist_ids)


def run_stage_r3(
    stage_r2_ranked,
    all_candidates,
    fold_data,
    trials_path,
    trial_rows,
    trial_lookup,
):
    cmap = candidate_map(all_candidates)
    finalist_ids = choose_finalists(
        stage_r2_ranked,
        all_candidates,
    )

    print()
    print(
        "Stage R3 finalists: "
        + ", ".join(finalist_ids)
    )

    for config_id in finalist_ids:
        config = cmap[config_id]

        for seed in CONFIRMATION_SEEDS:
            for fold in FOLDS:
                ensure_trial(
                    origin_stage="refinement_r3_multiseed_confirmation",
                    config=config,
                    seed=seed,
                    fold=fold,
                    fold_rows=fold_data[fold["name"]],
                    trials_path=trials_path,
                    trial_rows=trial_rows,
                    trial_lookup=trial_lookup,
                )

    summaries = [
        aggregate_config(
            stage="refinement_r3_multiseed_confirmation",
            config=cmap[config_id],
            seeds=CONFIRMATION_SEEDS,
            fold_names=[
                "fold_1",
                "fold_2",
                "fold_3",
            ],
            trial_rows=trial_rows,
        )
        for config_id in finalist_ids
    ]

    ranked = rank_rows(summaries)
    ranked = add_baseline_deltas(ranked)

    return ranked


# ============================================================
# Boundary warning
# ============================================================

def boundary_hits(config):
    hits = []

    for parameter, (low, high) in EXPANDED_BOUNDS.items():
        value = config[parameter]

        if np.isclose(float(value), float(low)):
            hits.append(f"{parameter}=LOW({low})")

        if np.isclose(float(value), float(high)):
            hits.append(f"{parameter}=HIGH({high})")

    return hits


# ============================================================
# Plan / reporting
# ============================================================

def print_plan(candidates):
    print("=" * 110)
    print("IBPR LOCAL HYPERPARAMETER REFINEMENT")
    print("=" * 110)
    print(
        "Base protocol : same MovieLens 1M implicit-feedback, first 60% "
        "global HPO horizon, per-user temporal warm-start folds."
    )
    print(f"Primary metric: {PRIMARY_METRIC}")
    print()
    print("Initial HPO winner:")
    print(INITIAL_HPO_WINNER)
    print()
    print("Stage R1 candidates:")

    for config in candidates:
        print(
            f"  {config['config_id']} | "
            f"group={config['refinement_group']:<16s} | "
            f"k={config['k']:<2d} "
            f"lr={config['learning_rate']:<7g} "
            f"lamda={config['lamda']:<8g} "
            f"batch={config['batch_size']:<4d} "
            f"iter={config['max_iter']}"
        )

    print()
    print("Budget:")
    print(
        f"  R1: {len(candidates)} full configurations × folds 1-2 × "
        f"seed {SCREENING_SEED}"
    )
    print(
        "  R2: top 5 + best per probe group + baseline + combined-best "
        "candidate → fold 3"
    )
    print(
        "  R3: best + best structurally different + baseline × "
        f"3 folds × seeds={CONFIRMATION_SEEDS}"
    )
    print()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Reuse all protocol safeguards from the validated HPO script.
    base.validate_hpo_protocol()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = (
        args.timestamp
        if args.timestamp
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    trials_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_refinement_ibpr_trials_{timestamp}.csv",
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_refinement_ibpr_summary_{timestamp}.csv",
    )

    best_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_refinement_ibpr_best_{timestamp}.csv",
    )

    log_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_refinement_ibpr_{timestamp}.txt",
    )

    log_mode = "a" if os.path.exists(log_path) else "w"

    with open(log_path, log_mode, encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)

        with redirect_stdout(tee):
            print()
            print("#" * 110)
            print(f"IBPR REFINEMENT TIMESTAMP: {timestamp}")
            print("#" * 110)
            print(f"Trials CSV : {trials_path}")
            print(f"Summary CSV: {summary_path}")
            print(f"Best CSV   : {best_path}")
            print()

            all_positive_rows = (
                base.load_positive_chrono_movielens()
            )

            hpo_pool_rows = base.build_hpo_pool(
                all_positive_rows
            )

            user_histories = base.build_user_histories(
                hpo_pool_rows
            )

            fold_data = {
                fold["name"]: base.prepare_fold_rows(
                    user_histories,
                    fold,
                )
                for fold in FOLDS
            }

            candidates = build_stage_r1_candidates()

            base.print_data_protocol(
                all_positive_rows,
                hpo_pool_rows,
                user_histories,
                fold_data,
            )

            print_plan(candidates)

            if args.plan_only:
                print("PLAN ONLY: no se entrenó ningún modelo.")
                return

            trial_rows = load_csv(trials_path)
            trial_lookup = build_trial_lookup(
                trial_rows
            )

            if trial_rows:
                print(
                    f"Resume: {len(trial_rows)} trials físicos "
                    "ya disponibles."
                )
                print()

            # --------------------------------------------
            # R1
            # --------------------------------------------
            r1_ranked = run_stage_r1(
                candidates=candidates,
                fold_data=fold_data,
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

            print_ranking(
                "STAGE R1 - LOCAL SCREENING",
                r1_ranked,
            )

            combined_candidate, combined_info = (
                build_combined_candidate(
                    r1_ranked,
                    candidates,
                )
            )

            print()
            print("Combined-best construction:")
            print(combined_info)
            print("Combined candidate:")
            print(combined_candidate)

            # --------------------------------------------
            # R2
            # --------------------------------------------
            r2_ranked, all_candidates = run_stage_r2(
                stage_r1_ranked=r1_ranked,
                initial_candidates=candidates,
                combined_candidate=combined_candidate,
                fold_data=fold_data,
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

            print_ranking(
                "STAGE R2 - THREE-FOLD TEMPORAL CONFIRMATION",
                r2_ranked,
            )

            # --------------------------------------------
            # R3
            # --------------------------------------------
            r3_ranked = run_stage_r3(
                stage_r2_ranked=r2_ranked,
                all_candidates=all_candidates,
                fold_data=fold_data,
                trials_path=trials_path,
                trial_rows=trial_rows,
                trial_lookup=trial_lookup,
            )

            print_ranking(
                "STAGE R3 - MULTI-SEED FINAL CONFIRMATION",
                r3_ranked,
            )

            all_summary_rows = (
                r1_ranked
                + r2_ranked
                + r3_ranked
            )

            save_csv(
                summary_path,
                SUMMARY_FIELDS,
                all_summary_rows,
            )

            best = dict(r3_ranked[0])
            best_config = {
                "config_id": best["config_id"],
                "k": int(best["k"]),
                "learning_rate": float(
                    best["learning_rate"]
                ),
                "lamda": float(best["lamda"]),
                "batch_size": int(
                    best["batch_size"]
                ),
                "max_iter": int(
                    best["max_iter"]
                ),
            }

            hits = boundary_hits(best_config)

            BEST_FIELDS = list(SUMMARY_FIELDS) + [
                "selection_metric",
                "initial_hpo_timestamp",
                "boundary_hits",
                "feedback_type",
                "validation_protocol",
            ]

            best["selection_metric"] = PRIMARY_METRIC
            best["initial_hpo_timestamp"] = (
                "20260901_213038"
            )
            best["boundary_hits"] = ";".join(hits)
            best["feedback_type"] = "implicit_positive"
            best["validation_protocol"] = (
                "first_60pct_global_then_per_user_temporal_warm_start"
            )

            save_csv(
                best_path,
                BEST_FIELDS,
                [best],
            )

            print()
            print("=" * 110)
            print("REFINEMENT WINNER")
            print("=" * 110)
            print(
                "IBPR_CONFIG = {\n"
                f'    "k": {best_config["k"]},\n'
                f'    "max_iter": {best_config["max_iter"]},\n'
                f'    "learning_rate": {best_config["learning_rate"]},\n'
                f'    "lamda": {best_config["lamda"]},\n'
                f'    "batch_size": {best_config["batch_size"]},\n'
                '    "verbose": True,\n'
                "}"
            )
            print()
            print(
                f"Final {PRIMARY_METRIC}: "
                f"{float(best[f'mean_{PRIMARY_METRIC}']):.6f} "
                f"± {float(best[f'std_{PRIMARY_METRIC}']):.6f}"
            )
            print(
                "Delta vs initial HPO winner: "
                f"{float(best['delta_ndcg_vs_initial_baseline']):+.6f}"
            )

            if hits:
                print()
                print(
                    "BOUNDARY REVIEW WARNING: final winner still touches "
                    "expanded local boundary/boundaries:"
                )
                print("  " + ", ".join(hits))
                print(
                    "Do not automatically expand again; review the size and "
                    "consistency of the improvement before deciding."
                )
            else:
                print()
                print(
                    "No expanded local boundary was hit by the final winner."
                )


if __name__ == "__main__":
    main()
