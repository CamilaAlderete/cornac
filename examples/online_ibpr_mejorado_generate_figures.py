import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOP_K = 20

MODELS = [
    "IBPR_STALE",
    "ONLINE_IBPR",
    "IBPR_FULL_RETRAIN",
]

MODEL_LABELS = {
    "IBPR_STALE": "IBPR Stale",
    "ONLINE_IBPR": "OnlineIBPR",
    "IBPR_FULL_RETRAIN": "IBPR Full Retrain",
}

STAGE_ORDER = [
    "chunk_1_preupdate_eval",
    "chunk_2_preupdate_eval",
    "chunk_3_preupdate_eval",
    "chunk_4_preupdate_eval",
    "final_holdout",
]

STAGE_LABELS = {
    "chunk_1_preupdate_eval": "Chunk 1",
    "chunk_2_preupdate_eval": "Chunk 2",
    "chunk_3_preupdate_eval": "Chunk 3",
    "chunk_4_preupdate_eval": "Chunk 4",
    "final_holdout": "Holdout final",
}

QUALITY_METRICS = [
    "AUC",
    "MAP",
    f"NDCG@{TOP_K}",
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera figuras reproducibles del experimento multi-seed "
            "IBPR Stale vs OnlineIBPR vs Full Retrain."
        )
    )
    parser.add_argument(
        "--timestamp",
        default="20260830_232628",
        help="Timestamp del experimento a analizar.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directorio que contiene los CSV de análisis.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directorio de salida. Por defecto: "
            "results/figures/<timestamp>/"
        ),
    )
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido: {path}"
        )


def save_figure(fig, output_dir, basename):
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Generated PNG: {png_path}")
    print(f"Generated PDF: {pdf_path}")


def load_analysis_files(results_dir, timestamp):
    stage_path = (
        results_dir
        / f"multiseed_stage_model_summary_{timestamp}.csv"
    )
    paired_path = (
        results_dir
        / f"multiseed_stage_paired_deltas_{timestamp}.csv"
    )
    efficiency_path = (
        results_dir
        / f"multiseed_efficiency_summary_{timestamp}.csv"
    )

    for path in (stage_path, paired_path, efficiency_path):
        require_file(path)

    return (
        read_csv(stage_path),
        read_csv(paired_path),
        read_csv(efficiency_path),
    )


def build_stage_lookup(stage_rows):
    lookup = {}

    for row in stage_rows:
        key = (
            row["stage"],
            row["model"],
            row["metric"],
        )
        lookup[key] = row

    return lookup


def generate_ndcg_temporal(stage_rows, output_dir):
    lookup = build_stage_lookup(stage_rows)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    x = np.arange(len(STAGE_ORDER))

    for model in MODELS:
        means = []
        lower_errors = []
        upper_errors = []

        for stage in STAGE_ORDER:
            row = lookup[(stage, model, f"NDCG@{TOP_K}")]

            mean_value = float(row["mean"])
            ci_low = float(row["ci95_low"])
            ci_high = float(row["ci95_high"])

            means.append(mean_value)
            lower_errors.append(mean_value - ci_low)
            upper_errors.append(ci_high - mean_value)

        ax.errorbar(
            x,
            means,
            yerr=np.asarray(
                [lower_errors, upper_errors]
            ),
            marker="o",
            capsize=3,
            label=MODEL_LABELS[model],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [STAGE_LABELS[stage] for stage in STAGE_ORDER]
    )
    ax.set_xlabel("Etapa de evaluación")
    ax.set_ylabel(f"NDCG@{TOP_K}")
    ax.set_title(
        f"Evolución temporal de NDCG@{TOP_K} "
        "durante el flujo incremental"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_01_ndcg20_evolucion_stream",
    )


def generate_online_vs_stale_deltas(paired_rows, output_dir):
    rows = [
        row
        for row in paired_rows
        if row["stage"] == "final_holdout"
        and row["comparison"] == "ONLINE_MINUS_STALE"
    ]

    by_metric = {
        row["metric"]: row
        for row in rows
    }

    means = []
    lower_errors = []
    upper_errors = []

    for metric in QUALITY_METRICS:
        row = by_metric[metric]

        mean_delta = float(row["mean_delta"])
        ci_low = float(row["ci95_low"])
        ci_high = float(row["ci95_high"])

        means.append(mean_delta)
        lower_errors.append(mean_delta - ci_low)
        upper_errors.append(ci_high - mean_delta)

    y = np.arange(len(QUALITY_METRICS))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.errorbar(
        means,
        y,
        xerr=np.asarray(
            [lower_errors, upper_errors]
        ),
        fmt="o",
        capsize=4,
    )

    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(QUALITY_METRICS)
    ax.set_xlabel(
        "Diferencia media (OnlineIBPR - IBPR Stale)"
    )
    ax.set_title(
        "Deltas emparejados en el holdout final "
        "con intervalo de confianza del 95%"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_02_deltas_online_vs_stale_ic95",
    )


def generate_adaptation_cost(efficiency_rows, output_dir):
    mean_row = next(
        row
        for row in efficiency_rows
        if row["seed"] == "MEAN"
    )

    online_time = float(
        mean_row["online_adaptation_time_s"]
    )
    retrain_time = float(
        mean_row["full_retrain_adaptation_time_s"]
    )
    speedup = float(
        mean_row[
            "adaptation_speedup_full_over_online"
        ]
    )

    labels = [
        "OnlineIBPR",
        "IBPR Full Retrain",
    ]
    values = [
        online_time,
        retrain_time,
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    bars = ax.bar(labels, values)

    ax.set_yscale("log")
    ax.set_ylabel(
        "Tiempo acumulado de adaptación (s, escala log)"
    )
    ax.set_title(
        "Coste computacional de adaptación incremental"
    )
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f} s",
            ha="center",
            va="bottom",
        )

    ax.text(
        0.5,
        0.93,
        f"Speedup medio: {speedup:.2f}×",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_03_coste_adaptacion_log",
    )


def write_manifest(output_dir, timestamp):
    manifest_path = output_dir / "figures_manifest.csv"

    rows = [
        {
            "figure_id": "fig_01",
            "file_base": "fig_01_ndcg20_evolucion_stream",
            "hypothesis": "H1 / H3",
            "purpose": (
                "Comparar la evolución temporal de NDCG@20 "
                "entre IBPR Stale, OnlineIBPR y Full Retrain."
            ),
        },
        {
            "figure_id": "fig_02",
            "file_base": "fig_02_deltas_online_vs_stale_ic95",
            "hypothesis": "H1",
            "purpose": (
                "Mostrar el delta medio OnlineIBPR - IBPR Stale "
                "y su IC95% para las métricas de calidad."
            ),
        },
        {
            "figure_id": "fig_03",
            "file_base": "fig_03_coste_adaptacion_log",
            "hypothesis": "H2",
            "purpose": (
                "Comparar el coste acumulado de adaptación "
                "OnlineIBPR frente a Full Retrain."
            ),
        },
    ]

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "figure_id",
                "file_base",
                "hypothesis",
                "purpose",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated manifest: {manifest_path}")
    print(f"Experiment timestamp: {timestamp}")


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            results_dir
            / "figures"
            / args.timestamp
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        stage_rows,
        paired_rows,
        efficiency_rows,
    ) = load_analysis_files(
        results_dir,
        args.timestamp,
    )

    generate_ndcg_temporal(
        stage_rows,
        output_dir,
    )

    generate_online_vs_stale_deltas(
        paired_rows,
        output_dir,
    )

    generate_adaptation_cost(
        efficiency_rows,
        output_dir,
    )

    write_manifest(
        output_dir,
        args.timestamp,
    )

    print()
    print("=" * 80)
    print("FIGURE GENERATION COMPLETED")
    print("=" * 80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
