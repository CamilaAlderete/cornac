import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


# ============================================================
# Configuration
# ============================================================
SEEDS = [42, 123, 2024, 777, 999]
DEFAULT_TIMESTAMP = "20260830_232628"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TOP_K = 20

QUALITY_METRICS = [
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
]

ALL_METRICS = QUALITY_METRICS + [
    "cumulative_compute_time_s",
    "eval_time_s",
]

MODELS = [
    "IBPR_STALE",
    "ONLINE_IBPR",
    "IBPR_FULL_RETRAIN",
]

COMPARISONS = [
    ("ONLINE_MINUS_STALE", "ONLINE_IBPR", "IBPR_STALE"),
    (
        "ONLINE_MINUS_FULL_RETRAIN",
        "ONLINE_IBPR",
        "IBPR_FULL_RETRAIN",
    ),
    (
        "FULL_RETRAIN_MINUS_STALE",
        "IBPR_FULL_RETRAIN",
        "IBPR_STALE",
    ),
]

STAGE_ORDER = [
    "chunk_1_preupdate_eval",
    "chunk_2_preupdate_eval",
    "chunk_3_preupdate_eval",
    "chunk_4_preupdate_eval",
    "final_holdout",
]


# ============================================================
# Helpers
# ============================================================
def save_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in fieldnames}
            )


def load_experiment_rows(timestamp):
    rows = []

    for seed in SEEDS:
        path = os.path.join(
            RESULTS_DIR,
            (
                "master_stale_online_retrain_quality_"
                f"seed_{seed}_{timestamp}.csv"
            ),
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró el archivo requerido: {path}"
            )

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for raw_row in reader:
                row = dict(raw_row)
                row["seed"] = seed
                row["train_size"] = int(row["train_size"])

                for metric in ALL_METRICS:
                    row[metric] = float(row[metric])

                rows.append(row)

    return rows


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    mean_value = float(np.mean(values))

    if n < 2:
        return {
            "n": n,
            "mean": mean_value,
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    std_value = float(np.std(values, ddof=1))
    sem = std_value / math.sqrt(n)
    t_critical = float(stats.t.ppf(0.975, df=n - 1))
    half_width = t_critical * sem

    return {
        "n": n,
        "mean": mean_value,
        "std": std_value,
        "ci95_low": mean_value - half_width,
        "ci95_high": mean_value + half_width,
    }


def paired_test(values):
    values = np.asarray(values, dtype=np.float64)
    summary = summarize_values(values)

    std_value = summary["std"]

    if len(values) < 2:
        p_value = float("nan")
        cohens_dz = float("nan")
    elif np.isclose(std_value, 0.0):
        p_value = 1.0 if np.isclose(summary["mean"], 0.0) else 0.0
        cohens_dz = float("nan")
    else:
        test = stats.ttest_1samp(values, popmean=0.0)
        p_value = float(test.pvalue)
        cohens_dz = float(summary["mean"] / std_value)

    tolerance = 1e-12
    wins = int(np.sum(values > tolerance))
    ties = int(np.sum(np.abs(values) <= tolerance))
    losses = int(np.sum(values < -tolerance))

    return {
        **summary,
        "p_value_paired_t": p_value,
        "cohens_dz": cohens_dz,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def build_index(rows):
    return {
        (row["seed"], row["stage"], row["model"]): row
        for row in rows
    }


# ============================================================
# Analysis
# ============================================================
def build_stage_model_summary(rows):
    grouped = defaultdict(list)

    for row in rows:
        for metric in ALL_METRICS:
            grouped[
                (row["stage"], row["model"], metric)
            ].append(row[metric])

    output = []

    for stage in STAGE_ORDER:
        for model in MODELS:
            for metric in ALL_METRICS:
                values = grouped[(stage, model, metric)]

                if not values:
                    continue

                summary = summarize_values(values)

                output.append(
                    {
                        "stage": stage,
                        "model": model,
                        "metric": metric,
                        **summary,
                    }
                )

    return output


def build_paired_delta_summary(rows):
    index = build_index(rows)
    output = []

    for stage in STAGE_ORDER:
        for comparison_name, model_a, model_b in COMPARISONS:
            for metric in QUALITY_METRICS:
                deltas = []

                for seed in SEEDS:
                    row_a = index[(seed, stage, model_a)]
                    row_b = index[(seed, stage, model_b)]

                    deltas.append(
                        row_a[metric] - row_b[metric]
                    )

                result = paired_test(deltas)

                output.append(
                    {
                        "stage": stage,
                        "comparison": comparison_name,
                        "metric": metric,
                        "n": result["n"],
                        "mean_delta": result["mean"],
                        "std_delta": result["std"],
                        "ci95_low": result["ci95_low"],
                        "ci95_high": result["ci95_high"],
                        "wins": result["wins"],
                        "ties": result["ties"],
                        "losses": result["losses"],
                        "p_value_paired_t": result[
                            "p_value_paired_t"
                        ],
                        "cohens_dz": result["cohens_dz"],
                    }
                )

    return output


def build_efficiency_summary(rows):
    index = build_index(rows)
    output = []

    speedups = []
    online_fractions = []
    total_ratios = []

    for seed in SEEDS:
        stale = index[(seed, "final_holdout", "IBPR_STALE")]
        online = index[(seed, "final_holdout", "ONLINE_IBPR")]
        full = index[
            (seed, "final_holdout", "IBPR_FULL_RETRAIN")
        ]

        base_time = stale["cumulative_compute_time_s"]
        online_total = online["cumulative_compute_time_s"]
        full_total = full["cumulative_compute_time_s"]

        online_adaptation = online_total - base_time
        full_adaptation = full_total - base_time

        adaptation_speedup = (
            full_adaptation / online_adaptation
        )

        online_fraction_pct = (
            100.0 * online_adaptation / full_adaptation
        )

        total_cost_ratio = full_total / online_total

        speedups.append(adaptation_speedup)
        online_fractions.append(online_fraction_pct)
        total_ratios.append(total_cost_ratio)

        output.append(
            {
                "seed": seed,
                "base_train_time_s": base_time,
                "online_total_compute_s": online_total,
                "full_retrain_total_compute_s": full_total,
                "online_adaptation_time_s": online_adaptation,
                "full_retrain_adaptation_time_s": full_adaptation,
                "adaptation_speedup_full_over_online":
                    adaptation_speedup,
                "online_adaptation_fraction_of_full_pct":
                    online_fraction_pct,
                "total_cost_ratio_full_over_online":
                    total_cost_ratio,
            }
        )

    output.append(
        {
            "seed": "MEAN",
            "base_train_time_s": float(
                np.mean(
                    [
                        row["base_train_time_s"]
                        for row in output
                        if row["seed"] != "MEAN"
                    ]
                )
            ),
            "online_total_compute_s": float(
                np.mean(
                    [
                        row["online_total_compute_s"]
                        for row in output
                        if row["seed"] != "MEAN"
                    ]
                )
            ),
            "full_retrain_total_compute_s": float(
                np.mean(
                    [
                        row["full_retrain_total_compute_s"]
                        for row in output
                        if row["seed"] != "MEAN"
                    ]
                )
            ),
            "online_adaptation_time_s": float(
                np.mean(
                    [
                        row["online_adaptation_time_s"]
                        for row in output
                        if row["seed"] != "MEAN"
                    ]
                )
            ),
            "full_retrain_adaptation_time_s": float(
                np.mean(
                    [
                        row["full_retrain_adaptation_time_s"]
                        for row in output
                        if row["seed"] != "MEAN"
                    ]
                )
            ),
            "adaptation_speedup_full_over_online":
                float(np.mean(speedups)),
            "online_adaptation_fraction_of_full_pct":
                float(np.mean(online_fractions)),
            "total_cost_ratio_full_over_online":
                float(np.mean(total_ratios)),
        }
    )

    return output


def print_final_holdout_summary(stage_summary, paired_summary, efficiency):
    print()
    print("=" * 80)
    print("FINAL HOLDOUT - MULTI-SEED QUALITY")
    print("=" * 80)

    stage_lookup = {
        (row["stage"], row["model"], row["metric"]): row
        for row in stage_summary
    }

    for model in MODELS:
        print(model)

        for metric in QUALITY_METRICS:
            row = stage_lookup[
                ("final_holdout", model, metric)
            ]

            print(
                f"  {metric:12s}: "
                f"{row['mean']:.6f} ± {row['std']:.6f} "
                f"[95% CI {row['ci95_low']:.6f}, "
                f"{row['ci95_high']:.6f}]"
            )

        print()

    print("=" * 80)
    print("FINAL HOLDOUT - PAIRED DELTAS")
    print("=" * 80)

    for comparison_name, _, _ in COMPARISONS:
        print(comparison_name)

        comparison_rows = [
            row
            for row in paired_summary
            if row["stage"] == "final_holdout"
            and row["comparison"] == comparison_name
        ]

        for row in comparison_rows:
            print(
                f"  {row['metric']:12s}: "
                f"Δ={row['mean_delta']:+.6f}, "
                f"95% CI [{row['ci95_low']:+.6f}, "
                f"{row['ci95_high']:+.6f}], "
                f"W/T/L={row['wins']}/{row['ties']}/"
                f"{row['losses']}, "
                f"p={row['p_value_paired_t']:.4f}, "
                f"dz={row['cohens_dz']:.3f}"
            )

        print()

    efficiency_mean = next(
        row for row in efficiency if row["seed"] == "MEAN"
    )

    print("=" * 80)
    print("EFFICIENCY")
    print("=" * 80)
    print(
        "Mean Online adaptation time      : "
        f"{efficiency_mean['online_adaptation_time_s']:.6f} s"
    )
    print(
        "Mean Full Retrain adaptation time: "
        f"{efficiency_mean['full_retrain_adaptation_time_s']:.6f} s"
    )
    print(
        "Mean adaptation speedup          : "
        f"{efficiency_mean['adaptation_speedup_full_over_online']:.2f}x"
    )
    print(
        "Online fraction of retrain cost  : "
        f"{efficiency_mean['online_adaptation_fraction_of_full_pct']:.4f}%"
    )
    print(
        "Full/Online total-cost ratio     : "
        f"{efficiency_mean['total_cost_ratio_full_over_online']:.2f}x"
    )


# ============================================================
# Main
# ============================================================
def main():
    timestamp = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_TIMESTAMP
    )

    print(f"Analyzing experiment timestamp: {timestamp}")

    rows = load_experiment_rows(timestamp)

    expected_rows = len(SEEDS) * len(STAGE_ORDER) * len(MODELS)

    if len(rows) != expected_rows:
        raise ValueError(
            f"Se esperaban {expected_rows} filas, "
            f"pero se encontraron {len(rows)}."
        )

    stage_summary = build_stage_model_summary(rows)
    paired_summary = build_paired_delta_summary(rows)
    efficiency = build_efficiency_summary(rows)

    stage_summary_csv = os.path.join(
        RESULTS_DIR,
        f"multiseed_stage_model_summary_{timestamp}.csv",
    )

    paired_summary_csv = os.path.join(
        RESULTS_DIR,
        f"multiseed_stage_paired_deltas_{timestamp}.csv",
    )

    efficiency_csv = os.path.join(
        RESULTS_DIR,
        f"multiseed_efficiency_summary_{timestamp}.csv",
    )

    save_csv(
        stage_summary_csv,
        [
            "stage",
            "model",
            "metric",
            "n",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
        ],
        stage_summary,
    )

    save_csv(
        paired_summary_csv,
        [
            "stage",
            "comparison",
            "metric",
            "n",
            "mean_delta",
            "std_delta",
            "ci95_low",
            "ci95_high",
            "wins",
            "ties",
            "losses",
            "p_value_paired_t",
            "cohens_dz",
        ],
        paired_summary,
    )

    save_csv(
        efficiency_csv,
        [
            "seed",
            "base_train_time_s",
            "online_total_compute_s",
            "full_retrain_total_compute_s",
            "online_adaptation_time_s",
            "full_retrain_adaptation_time_s",
            "adaptation_speedup_full_over_online",
            "online_adaptation_fraction_of_full_pct",
            "total_cost_ratio_full_over_online",
        ],
        efficiency,
    )

    print_final_holdout_summary(
        stage_summary,
        paired_summary,
        efficiency,
    )

    print()
    print("=" * 80)
    print("ANALYSIS FILES")
    print("=" * 80)
    print(f"Stage summary : {stage_summary_csv}")
    print(f"Paired deltas : {paired_summary_csv}")
    print(f"Efficiency    : {efficiency_csv}")


if __name__ == "__main__":
    main()
