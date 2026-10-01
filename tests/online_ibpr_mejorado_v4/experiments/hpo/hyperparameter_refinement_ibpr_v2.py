import argparse
import csv
import hashlib
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np

import hyperparameter_search_ibpr_v2 as base
from cornac.utils.Tee import Tee


# ============================================================
# Refinement protocol
# ============================================================

REFINEMENT_PROTOCOL_VERSION = "ibpr_local_refinement_v2_20260925"
REQUIRED_PARENT_PROTOCOL_VERSION = "ibpr_hpo_per_user_temporal_v2_20260924"
DEFAULT_SCREENING_CONFIGS = 20

PRIMARY_METRIC = base.PRIMARY_METRIC
QUALITY_METRICS = list(base.QUALITY_METRICS)
FOLDS = list(base.FOLDS)
SCREENING_SEED = 42
CONFIRMATION_SEEDS = [42, 123, 2024]
TOP_STAGE_R1 = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# These probes are inherited from the pre-existing targeted refinement design.
# They deliberately cross the boundaries touched by the base-HPO winner.
K_PROBES = [5, 10, 30]
LAMDA_PROBES = [0.00001, 0.00005, 0.0005]
BATCH_PROBES = [256, 1024, 2048]

# learning_rate and max_iter are crossed because a lower learning rate may need
# a larger optimization budget to be assessed fairly.
LR_VALUES = [0.001, 0.0025, 0.005, 0.01]
ITER_VALUES = [10, 20, 50]

EXPANDED_BOUNDS = {
    "k": (5, 30),
    "learning_rate": (0.001, 0.01),
    "lamda": (0.00001, 0.0005),
    "batch_size": (256, 2048),
    "max_iter": (10, 50),
}

PROVENANCE_FIELDS = [
    "protocol_hash",
    "dataset_sha256",
    "script_sha256",
    "ibpr_wrapper_sha256",
    "ibpr_core_sha256",
]

TRIAL_FIELDS = PROVENANCE_FIELDS + [
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
    "delta_ndcg_vs_parent_winner",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Targeted local refinement around the winner of the fingerprinted "
            "IBPR HPO V2. Uses exactly the same development data and folds."
        )
    )
    parser.add_argument(
        "--parent-hpo-timestamp",
        required=True,
        help=(
            "Timestamp of the completed base HPO V2 whose best CSV and manifest "
            "must exist in this script's results directory."
        ),
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Timestamp for this refinement. Reusing it resumes only when the "
            "refinement manifest matches exactly."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the validated refinement plan without fitting models.",
    )
    return parser.parse_args()


# ============================================================
# Parent HPO binding
# ============================================================

def load_one_csv_row(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Se esperaba exactamente una fila en {path}; hay {len(rows)}.")
    return rows[0]


def load_parent_hpo(parent_timestamp):
    best_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_ibpr_best_{parent_timestamp}.csv",
    )
    manifest_path = os.path.join(
        RESULTS_DIR,
        f"hyperparameter_search_ibpr_manifest_{parent_timestamp}.json",
    )

    best = load_one_csv_row(best_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("protocol_version") != REQUIRED_PARENT_PROTOCOL_VERSION:
        raise ValueError(
            "El HPO padre no usa el protocolo V2 requerido: "
            f"{manifest.get('protocol_version')!r}."
        )

    required_identity = [
        "protocol_hash",
        "dataset_sha256",
        "script_sha256",
        "ibpr_wrapper_sha256",
        "ibpr_core_sha256",
    ]
    for field in required_identity:
        if not manifest.get(field):
            raise ValueError(f"El manifest padre no contiene {field}.")
        if str(best.get(field, "")) != str(manifest[field]):
            raise ValueError(
                f"Best CSV y manifest padre no coinciden en {field}: "
                f"best={best.get(field)!r}, manifest={manifest[field]!r}."
            )

    if str(best.get("selection_metric")) != PRIMARY_METRIC:
        raise ValueError(
            f"El HPO padre seleccionó por {best.get('selection_metric')}, "
            f"no por {PRIMARY_METRIC}."
        )

    winner = {
        "config_id": "R000",
        "config_source": f"parent_hpo_winner_{parent_timestamp}",
        "refinement_group": "baseline",
        "k": int(best["k"]),
        "learning_rate": float(best["learning_rate"]),
        "lamda": float(best["lamda"]),
        "batch_size": int(best["batch_size"]),
        "max_iter": int(best["max_iter"]),
    }

    return {
        "timestamp": parent_timestamp,
        "best_path": os.path.abspath(best_path),
        "manifest_path": os.path.abspath(manifest_path),
        "best": best,
        "manifest": manifest,
        "winner": winner,
    }


def validate_parent_against_current_environment(parent, all_positive_rows):
    manifest = parent["manifest"]

    current_base_hash = base.sha256_file(os.path.abspath(base.__file__))
    if current_base_hash != manifest["script_sha256"]:
        raise ValueError(
            "hyperparameter_search_ibpr_v2.py cambió desde el HPO padre. "
            f"actual={current_base_hash}, padre={manifest['script_sha256']}."
        )

    current_protocol_hash = base.sha256_json(
        base.protocol_payload(DEFAULT_SCREENING_CONFIGS)
    )
    if current_protocol_hash != manifest["protocol_hash"]:
        raise ValueError(
            "La definición del protocolo base ya no coincide con el HPO padre."
        )

    current_dataset_hash = base.sha256_processed_rows(all_positive_rows)
    if current_dataset_hash != manifest["dataset_sha256"]:
        raise ValueError(
            "El dataset procesado actual no coincide con el HPO padre."
        )

    source_hashes = base.resolve_ibpr_source_hashes()
    for field in ["ibpr_wrapper_sha256", "ibpr_core_sha256"]:
        if source_hashes[field] != manifest[field]:
            raise ValueError(
                f"La fuente actual {field} no coincide con el HPO padre."
            )

    return source_hashes


# ============================================================
# Refinement provenance / manifest
# ============================================================

def refinement_protocol_payload(parent):
    return {
        "protocol_version": REFINEMENT_PROTOCOL_VERSION,
        "parent_hpo_timestamp": parent["timestamp"],
        "parent_protocol_version": parent["manifest"]["protocol_version"],
        "parent_protocol_hash": parent["manifest"]["protocol_hash"],
        "parent_dataset_sha256": parent["manifest"]["dataset_sha256"],
        "parent_script_sha256": parent["manifest"]["script_sha256"],
        "parent_winner": parent["winner"],
        "folds": FOLDS,
        "primary_metric": PRIMARY_METRIC,
        "screening_seed": SCREENING_SEED,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "top_stage_r1": TOP_STAGE_R1,
        "k_probes": K_PROBES,
        "lamda_probes": LAMDA_PROBES,
        "batch_probes": BATCH_PROBES,
        "lr_values": LR_VALUES,
        "iter_values": ITER_VALUES,
        "expanded_bounds": EXPANDED_BOUNDS,
        "selection_strategy": (
            "R1 folds1-2 seed42; R2 top5+group-winners+baseline+combined on fold3; "
            "R3 best+best-structurally-different+baseline on all folds and seeds"
        ),
    }


def build_refinement_provenance(parent, all_positive_rows, source_hashes):
    payload = refinement_protocol_payload(parent)
    return {
        "protocol_version": REFINEMENT_PROTOCOL_VERSION,
        "protocol_hash": base.sha256_json(payload),
        "dataset_sha256": base.sha256_processed_rows(all_positive_rows),
        "script_sha256": base.sha256_file(os.path.abspath(__file__)),
        "ibpr_wrapper_sha256": source_hashes["ibpr_wrapper_sha256"],
        "ibpr_core_sha256": source_hashes["ibpr_core_sha256"],
        "ibpr_wrapper_path": source_hashes["ibpr_wrapper_path"],
        "ibpr_core_path": source_hashes["ibpr_core_path"],
        "parent_hpo_timestamp": parent["timestamp"],
        "parent_hpo_manifest_path": parent["manifest_path"],
        "parent_hpo_best_path": parent["best_path"],
        "parent_protocol_hash": parent["manifest"]["protocol_hash"],
        "parent_script_sha256": parent["manifest"]["script_sha256"],
        "protocol_payload": payload,
    }


def save_or_validate_refinement_manifest(path, provenance):
    identity_fields = [
        "protocol_version",
        "protocol_hash",
        "dataset_sha256",
        "script_sha256",
        "ibpr_wrapper_sha256",
        "ibpr_core_sha256",
        "parent_hpo_timestamp",
        "parent_protocol_hash",
        "parent_script_sha256",
        "protocol_payload",
    ]
    current = {field: provenance[field] for field in identity_fields}
    current.update({
        "ibpr_wrapper_path": provenance["ibpr_wrapper_path"],
        "ibpr_core_path": provenance["ibpr_core_path"],
        "parent_hpo_manifest_path": provenance["parent_hpo_manifest_path"],
        "parent_hpo_best_path": provenance["parent_hpo_best_path"],
    })

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            existing = json.load(file)
        mismatch = [
            field for field in identity_fields
            if existing.get(field) != current.get(field)
        ]
        if mismatch:
            raise ValueError(
                "El manifest de refinamiento no coincide con esta ejecución. "
                "Campos distintos: " + ", ".join(mismatch) + ". Use un timestamp nuevo."
            )
        return

    with open(path, "w", encoding="utf-8") as file:
        json.dump(current, file, indent=2, sort_keys=True)
        file.write("\n")


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


def build_stage_r1_candidates(parent_winner):
    candidates = [dict(parent_winner)]
    counter = 1

    for k in K_PROBES:
        candidates.append(clone_candidate(
            parent_winner,
            config_id=f"R{counter:03d}",
            config_source="local_boundary_probe",
            refinement_group="k_probe",
            k=int(k),
        ))
        counter += 1

    for lamda in LAMDA_PROBES:
        candidates.append(clone_candidate(
            parent_winner,
            config_id=f"R{counter:03d}",
            config_source="local_boundary_probe",
            refinement_group="lamda_probe",
            lamda=float(lamda),
        ))
        counter += 1

    for batch_size in BATCH_PROBES:
        candidates.append(clone_candidate(
            parent_winner,
            config_id=f"R{counter:03d}",
            config_source="local_boundary_probe",
            refinement_group="batch_probe",
            batch_size=int(batch_size),
        ))
        counter += 1

    baseline_sig = full_signature(parent_winner)
    for learning_rate in LR_VALUES:
        for max_iter in ITER_VALUES:
            candidate = clone_candidate(
                parent_winner,
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

    seen = set()
    for candidate in candidates:
        sig = full_signature(candidate)
        if sig in seen:
            raise ValueError(f"Configuración duplicada en refinamiento: {sig}")
        seen.add(sig)
    return candidates


def candidate_map(candidates):
    return {candidate["config_id"]: candidate for candidate in candidates}


# ============================================================
# CSV / resume helpers
# ============================================================

def append_csv(path, fieldnames, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def build_trial_lookup(rows, provenance):
    lookup = {}
    for row in rows:
        for field in PROVENANCE_FIELDS:
            if str(row.get(field, "")) != str(provenance[field]):
                raise ValueError(
                    f"Trial existente no coincide con {field}; use un timestamp nuevo."
                )
        key = trial_key_from_row(row)
        if key in lookup:
            raise ValueError(f"Trial duplicado en CSV de reanudación: {key}")
        lookup[key] = row
    return lookup


# ============================================================
# Physical trials
# ============================================================

def ensure_trial(origin_stage, config, seed, fold, fold_rows,
                 trials_path, trial_rows, trial_lookup):
    key = trial_key_from_config(config, seed, fold["name"])
    if key in trial_lookup:
        print(
            "REUSE  "
            f"{config['config_id']} | {fold['name']} | seed={seed} | "
            f"iter={config['max_iter']}"
        )
        return trial_lookup[key]

    print(
        "RUN    "
        f"{config['config_id']} | group={config['refinement_group']} | "
        f"k={config['k']} lr={config['learning_rate']} "
        f"lamda={config['lamda']} batch={config['batch_size']} "
        f"iter={config['max_iter']} | {fold['name']} | seed={seed}"
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
    append_csv(trials_path, TRIAL_FIELDS, row)
    trial_rows.append(row)
    trial_lookup[key] = row
    print(
        f"       {PRIMARY_METRIC}={float(row[PRIMARY_METRIC]):.6f} | "
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


def matching_rows(trial_rows, config, seeds, fold_names):
    expected = {
        trial_key_from_config(config, seed, fold_name)
        for seed in seeds
        for fold_name in fold_names
    }
    selected = [row for row in trial_rows if trial_key_from_row(row) in expected]
    if len(selected) != len(expected):
        raise ValueError(
            f"Faltan ejecuciones para {config['config_id']}: "
            f"esperadas={len(expected)}, encontradas={len(selected)}."
        )
    return selected


def aggregate_config(stage, config, seeds, fold_names, trial_rows):
    rows = matching_rows(trial_rows, config, seeds, fold_names)
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
        mean_value, std_value = mean_std([float(row[metric]) for row in rows])
        summary[f"mean_{metric}"] = mean_value
        summary[f"std_{metric}"] = std_value
    mean_train, std_train = mean_std([float(row["train_time_s"]) for row in rows])
    mean_eval, std_eval = mean_std([float(row["eval_time_s"]) for row in rows])
    summary["mean_train_time_s"] = mean_train
    summary["std_train_time_s"] = std_train
    summary["mean_eval_time_s"] = mean_eval
    summary["std_eval_time_s"] = std_eval
    summary["delta_ndcg_vs_parent_winner"] = ""
    return summary


def ranking_key(row):
    return (
        -float(row[f"mean_{PRIMARY_METRIC}"]),
        float(row[f"std_{PRIMARY_METRIC}"]),
        float(row["mean_train_time_s"]),
        int(row["k"]),
        int(row["max_iter"]),
        int(row["batch_size"]),
    )


def rank_rows(rows):
    ranked = sorted(rows, key=ranking_key)
    output = []
    for index, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = index
        output.append(item)
    return output


def add_baseline_deltas(ranked, parent_winner):
    baseline_rows = [row for row in ranked if row["config_id"] == parent_winner["config_id"]]
    if not baseline_rows:
        return ranked
    baseline_ndcg = float(baseline_rows[0][f"mean_{PRIMARY_METRIC}"])
    output = []
    for row in ranked:
        item = dict(row)
        item["delta_ndcg_vs_parent_winner"] = (
            float(item[f"mean_{PRIMARY_METRIC}"]) - baseline_ndcg
        )
        output.append(item)
    return output


def print_ranking(title, ranked):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    for row in ranked:
        delta = row.get("delta_ndcg_vs_parent_winner", "")
        delta_text = f" | Δparent={float(delta):+.6f}" if delta != "" else ""
        print(
            f"#{int(row['rank']):02d} {row['config_id']} "
            f"[{row['refinement_group']}] | k={row['k']} "
            f"lr={row['learning_rate']} lamda={row['lamda']} "
            f"batch={row['batch_size']} iter={row['max_iter']} | "
            f"{PRIMARY_METRIC}={float(row[f'mean_{PRIMARY_METRIC}']):.6f} "
            f"± {float(row[f'std_{PRIMARY_METRIC}']):.6f}{delta_text}"
        )


# ============================================================
# R1 / combined / R2 / R3
# ============================================================

def run_stage_r1(candidates, parent_winner, fold_data, trials_path,
                 trial_rows, trial_lookup):
    fold_names = ["fold_1", "fold_2"]
    for config in candidates:
        for fold_name in fold_names:
            fold = next(item for item in FOLDS if item["name"] == fold_name)
            ensure_trial(
                "refinement_r1_screening", config, SCREENING_SEED, fold,
                fold_data[fold_name], trials_path, trial_rows, trial_lookup,
            )
    summaries = [
        aggregate_config(
            "refinement_r1_screening", config, [SCREENING_SEED],
            fold_names, trial_rows,
        )
        for config in candidates
    ]
    return add_baseline_deltas(rank_rows(summaries), parent_winner)


def best_config_id_for_group(stage_r1_ranked, groups):
    valid = [row for row in stage_r1_ranked if row["refinement_group"] in groups]
    if not valid:
        raise ValueError(f"No hay candidatos para grupos: {groups}")
    return valid[0]["config_id"]


def build_combined_candidate(stage_r1_ranked, configs, parent_winner):
    cmap = candidate_map(configs)
    best_k = cmap[best_config_id_for_group(stage_r1_ranked, {"baseline", "k_probe"})]
    best_lamda = cmap[best_config_id_for_group(stage_r1_ranked, {"baseline", "lamda_probe"})]
    best_batch = cmap[best_config_id_for_group(stage_r1_ranked, {"baseline", "batch_probe"})]
    best_lr_iter = cmap[best_config_id_for_group(stage_r1_ranked, {"baseline", "lr_x_iter_probe"})]
    combined = {
        "config_id": "R900",
        "config_source": "combined_from_group_winners",
        "refinement_group": "combined_best",
        "k": int(best_k["k"]),
        "learning_rate": float(best_lr_iter["learning_rate"]),
        "lamda": float(best_lamda["lamda"]),
        "batch_size": int(best_batch["batch_size"]),
        "max_iter": int(best_lr_iter["max_iter"]),
    }
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


def unique_preserving_order(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def run_stage_r2(stage_r1_ranked, initial_candidates, combined_candidate,
                 parent_winner, fold_data, trials_path, trial_rows, trial_lookup):
    all_candidates = list(initial_candidates)
    if combined_candidate["config_id"] not in {item["config_id"] for item in all_candidates}:
        all_candidates.append(combined_candidate)
    cmap = candidate_map(all_candidates)
    group_winner_ids = [
        best_config_id_for_group(stage_r1_ranked, {"baseline", "k_probe"}),
        best_config_id_for_group(stage_r1_ranked, {"baseline", "lamda_probe"}),
        best_config_id_for_group(stage_r1_ranked, {"baseline", "batch_probe"}),
        best_config_id_for_group(stage_r1_ranked, {"baseline", "lr_x_iter_probe"}),
    ]
    selected_ids = unique_preserving_order(
        [row["config_id"] for row in stage_r1_ranked[:TOP_STAGE_R1]]
        + group_winner_ids
        + [parent_winner["config_id"], combined_candidate["config_id"]]
    )

    if combined_candidate["config_id"] == "R900":
        for fold_name in ["fold_1", "fold_2"]:
            fold = next(item for item in FOLDS if item["name"] == fold_name)
            ensure_trial(
                "refinement_r2_temporal_confirmation", combined_candidate,
                SCREENING_SEED, fold, fold_data[fold_name], trials_path,
                trial_rows, trial_lookup,
            )

    fold_3 = next(item for item in FOLDS if item["name"] == "fold_3")
    for config_id in selected_ids:
        config = cmap[config_id]
        ensure_trial(
            "refinement_r2_temporal_confirmation", config, SCREENING_SEED,
            fold_3, fold_data["fold_3"], trials_path, trial_rows, trial_lookup,
        )

    summaries = [
        aggregate_config(
            "refinement_r2_temporal_confirmation", cmap[config_id],
            [SCREENING_SEED], ["fold_1", "fold_2", "fold_3"], trial_rows,
        )
        for config_id in selected_ids
    ]
    return add_baseline_deltas(rank_rows(summaries), parent_winner), all_candidates


def choose_finalists(stage_r2_ranked, all_candidates, parent_winner):
    cmap = candidate_map(all_candidates)
    best_id = stage_r2_ranked[0]["config_id"]
    best_structure = structural_signature(cmap[best_id])
    alternative_id = None
    for row in stage_r2_ranked[1:]:
        if structural_signature(cmap[row["config_id"]]) != best_structure:
            alternative_id = row["config_id"]
            break
    finalist_ids = [best_id]
    if alternative_id is not None:
        finalist_ids.append(alternative_id)
    finalist_ids.append(parent_winner["config_id"])
    return unique_preserving_order(finalist_ids)


def run_stage_r3(stage_r2_ranked, all_candidates, parent_winner, fold_data,
                 trials_path, trial_rows, trial_lookup):
    cmap = candidate_map(all_candidates)
    finalist_ids = choose_finalists(stage_r2_ranked, all_candidates, parent_winner)
    print("\nStage R3 finalists: " + ", ".join(finalist_ids))
    for config_id in finalist_ids:
        config = cmap[config_id]
        for seed in CONFIRMATION_SEEDS:
            for fold in FOLDS:
                ensure_trial(
                    "refinement_r3_multiseed_confirmation", config, seed, fold,
                    fold_data[fold["name"]], trials_path, trial_rows, trial_lookup,
                )
    summaries = [
        aggregate_config(
            "refinement_r3_multiseed_confirmation", cmap[config_id],
            CONFIRMATION_SEEDS, ["fold_1", "fold_2", "fold_3"], trial_rows,
        )
        for config_id in finalist_ids
    ]
    return add_baseline_deltas(rank_rows(summaries), parent_winner)


# ============================================================
# Reporting
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


def original_search_boundary_hits(config):
    hits = []
    base_bounds = {
        "k": (min(base.K_VALUES), max(base.K_VALUES)),
        "learning_rate": (min(base.LEARNING_RATE_VALUES), max(base.LEARNING_RATE_VALUES)),
        "lamda": (min(base.LAMDA_VALUES), max(base.LAMDA_VALUES)),
        "batch_size": (min(base.BATCH_SIZE_VALUES), max(base.BATCH_SIZE_VALUES)),
        "max_iter": (min(base.MAX_ITER_VALUES), max(base.MAX_ITER_VALUES)),
    }
    for parameter, (low, high) in base_bounds.items():
        value = float(config[parameter])
        if np.isclose(value, float(low)):
            hits.append(f"{parameter}=LOW({low})")
        if np.isclose(value, float(high)):
            hits.append(f"{parameter}=HIGH({high})")
    return hits


def print_data_protocol(all_positive_rows, hpo_pool_rows, boundary_info,
                        user_histories, fold_data, parent, provenance):
    print("=" * 110)
    print("IBPR LOCAL REFINEMENT V2 - SAME DEVELOPMENT PROTOCOL AS PARENT HPO")
    print("=" * 110)
    print(f"Parent HPO timestamp             : {parent['timestamp']}")
    print(f"Parent protocol hash             : {parent['manifest']['protocol_hash']}")
    print(f"Parent winner                    : {parent['winner']}")
    print(f"Dataset                          : MovieLens {base.VARIANT}")
    print(f"Positive threshold               : rating >= {base.RATING_THRESHOLD}")
    print(f"Primary metric                   : {PRIMARY_METRIC}")
    print(f"Total implicit-positive rows     : {len(all_positive_rows):,}")
    print(f"Target development rows          : {boundary_info['target_rows']:,}")
    print(f"Effective development rows       : {boundary_info['effective_rows']:,}")
    print(f"Tie rows added at boundary       : {boundary_info['tie_rows_added']:,}")
    print(f"Boundary timestamp               : {boundary_info['boundary_timestamp']}")
    print(f"Users inside development horizon : {len(user_histories):,}")
    print(f"Refinement protocol version      : {provenance['protocol_version']}")
    print(f"Refinement protocol hash         : {provenance['protocol_hash']}")
    print(f"Processed dataset SHA256         : {provenance['dataset_sha256']}")
    print(f"Refinement script SHA256         : {provenance['script_sha256']}")
    print(f"IBPR wrapper SHA256              : {provenance['ibpr_wrapper_sha256']}")
    print(f"IBPR core SHA256                 : {provenance['ibpr_core_sha256']}")
    print("\nPer-user temporal folds:")
    for fold in FOLDS:
        data = fold_data[fold["name"]]
        n_candidate = len(data["validation_original"])
        n_known = len(data["validation_known"])
        known_fraction = n_known / n_candidate if n_candidate else float("nan")
        print(
            f"  {fold['name']}: train={len(data['train_rows']):,} | "
            f"val_candidate={n_candidate:,} | val_known={n_known:,} "
            f"({100*known_fraction:.2f}%) | eval_users={len(data['validation_users']):,}"
        )
    print()


def print_plan(candidates, parent):
    print("=" * 110)
    print("IBPR TARGETED LOCAL REFINEMENT PLAN")
    print("=" * 110)
    print("Parent winner:")
    print(parent["winner"])
    print("Original-search boundary hits:")
    print("  " + ", ".join(original_search_boundary_hits(parent["winner"])))
    print("\nStage R1 candidates:")
    for config in candidates:
        print(
            f"  {config['config_id']} | group={config['refinement_group']:<16s} | "
            f"k={config['k']:<3d} lr={config['learning_rate']:<7g} "
            f"lamda={config['lamda']:<8g} batch={config['batch_size']:<4d} "
            f"iter={config['max_iter']}"
        )
    print("\nBudget:")
    print(f"  R1: {len(candidates)} configs × folds 1-2 × seed {SCREENING_SEED}")
    print("  R2: top 5 + group winners + parent + combined candidate → fold 3")
    print(
        "  R3: best + best structurally different + parent × 3 folds × "
        f"seeds={CONFIRMATION_SEEDS}"
    )
    print()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    base.validate_hpo_protocol()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    parent = load_parent_hpo(args.parent_hpo_timestamp)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    trials_path = os.path.join(RESULTS_DIR, f"hyperparameter_refinement_ibpr_v2_trials_{timestamp}.csv")
    summary_path = os.path.join(RESULTS_DIR, f"hyperparameter_refinement_ibpr_v2_summary_{timestamp}.csv")
    best_path = os.path.join(RESULTS_DIR, f"hyperparameter_refinement_ibpr_v2_best_{timestamp}.csv")
    manifest_path = os.path.join(RESULTS_DIR, f"hyperparameter_refinement_ibpr_v2_manifest_{timestamp}.json")
    log_path = os.path.join(RESULTS_DIR, f"hyperparameter_refinement_ibpr_v2_{timestamp}.txt")

    log_mode = "a" if os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            print("\n" + "#" * 110)
            print(f"IBPR REFINEMENT V2 TIMESTAMP: {timestamp}")
            print("#" * 110)
            print(f"Parent HPO : {parent['timestamp']}")
            print(f"Trials CSV : {trials_path}")
            print(f"Summary CSV: {summary_path}")
            print(f"Best CSV   : {best_path}")
            print(f"Manifest   : {manifest_path}\n")

            all_positive_rows = base.load_positive_chrono_movielens()
            source_hashes = validate_parent_against_current_environment(
                parent, all_positive_rows
            )
            hpo_pool_rows, boundary_info = base.build_hpo_pool(all_positive_rows)
            user_histories = base.build_user_histories(hpo_pool_rows)
            fold_data = {
                fold["name"]: base.prepare_fold_rows(user_histories, fold)
                for fold in FOLDS
            }

            provenance = build_refinement_provenance(
                parent, all_positive_rows, source_hashes
            )
            # base.run_physical_trial writes these fields into every row. We set
            # them to the refinement identity, not to the parent-HPO identity.
            base.CURRENT_PROVENANCE = provenance

            candidates = build_stage_r1_candidates(parent["winner"])
            print_data_protocol(
                all_positive_rows, hpo_pool_rows, boundary_info,
                user_histories, fold_data, parent, provenance,
            )
            print_plan(candidates, parent)

            if args.plan_only:
                print("PLAN ONLY: no se entrenó ningún modelo.")
                return

            if os.path.exists(trials_path) and not os.path.exists(manifest_path):
                raise ValueError(
                    "Existe un CSV de trials sin manifest de refinamiento V2 para "
                    "este timestamp. Use un timestamp nuevo."
                )
            save_or_validate_refinement_manifest(manifest_path, provenance)

            trial_rows = load_csv(trials_path)
            trial_lookup = build_trial_lookup(trial_rows, provenance)
            if trial_rows:
                print(f"Resume: {len(trial_rows)} trials físicos ya disponibles.\n")

            r1_ranked = run_stage_r1(
                candidates, parent["winner"], fold_data, trials_path,
                trial_rows, trial_lookup,
            )
            print_ranking("STAGE R1 - LOCAL SCREENING", r1_ranked)

            combined_candidate, combined_info = build_combined_candidate(
                r1_ranked, candidates, parent["winner"]
            )
            print("\nCombined-best construction:")
            print(combined_info)
            print("Combined candidate:")
            print(combined_candidate)

            r2_ranked, all_candidates = run_stage_r2(
                r1_ranked, candidates, combined_candidate, parent["winner"],
                fold_data, trials_path, trial_rows, trial_lookup,
            )
            print_ranking("STAGE R2 - THREE-FOLD TEMPORAL CONFIRMATION", r2_ranked)

            r3_ranked = run_stage_r3(
                r2_ranked, all_candidates, parent["winner"], fold_data,
                trials_path, trial_rows, trial_lookup,
            )
            print_ranking("STAGE R3 - MULTI-SEED FINAL CONFIRMATION", r3_ranked)

            save_csv(summary_path, SUMMARY_FIELDS, r1_ranked + r2_ranked + r3_ranked)

            best = dict(r3_ranked[0])
            best_config = {
                "config_id": best["config_id"],
                "k": int(best["k"]),
                "learning_rate": float(best["learning_rate"]),
                "lamda": float(best["lamda"]),
                "batch_size": int(best["batch_size"]),
                "max_iter": int(best["max_iter"]),
            }
            hits = boundary_hits(best_config)
            best_fields = list(SUMMARY_FIELDS) + [
                "selection_metric",
                "parent_hpo_timestamp",
                "parent_hpo_config_id",
                "parent_protocol_hash",
                "refinement_protocol_version",
                "refinement_protocol_hash",
                "dataset_sha256",
                "script_sha256",
                "ibpr_wrapper_sha256",
                "ibpr_core_sha256",
                "boundary_hits",
                "feedback_type",
                "validation_protocol",
            ]
            best.update({
                "selection_metric": PRIMARY_METRIC,
                "parent_hpo_timestamp": parent["timestamp"],
                "parent_hpo_config_id": parent["best"]["config_id"],
                "parent_protocol_hash": parent["manifest"]["protocol_hash"],
                "refinement_protocol_version": provenance["protocol_version"],
                "refinement_protocol_hash": provenance["protocol_hash"],
                "dataset_sha256": provenance["dataset_sha256"],
                "script_sha256": provenance["script_sha256"],
                "ibpr_wrapper_sha256": provenance["ibpr_wrapper_sha256"],
                "ibpr_core_sha256": provenance["ibpr_core_sha256"],
                "boundary_hits": ";".join(hits),
                "feedback_type": "implicit_positive",
                "validation_protocol": (
                    "tie_safe_first_60pct_global_then_per_user_temporal_warm_start"
                ),
            })
            save_csv(best_path, best_fields, [best])

            print("\n" + "=" * 110)
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
            print(
                f"Final {PRIMARY_METRIC}: {float(best[f'mean_{PRIMARY_METRIC}']):.6f} "
                f"± {float(best[f'std_{PRIMARY_METRIC}']):.6f}"
            )
            print(
                "Delta vs parent HPO winner: "
                f"{float(best['delta_ndcg_vs_parent_winner']):+.6f}"
            )
            if hits:
                print("\nBOUNDARY REVIEW WARNING: " + ", ".join(hits))
                print(
                    "No expanda automáticamente otra vez; revise magnitud y "
                    "consistencia antes de decidir."
                )
            else:
                print("\nNo expanded local boundary was hit by the refinement winner.")


if __name__ == "__main__":
    main()
