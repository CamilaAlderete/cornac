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


# ============================================================
# CLI / IO
# ============================================================
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
    raw_final_path = (
        results_dir
        / f"master_multiseed_final_raw_{timestamp}.csv"
    )

    for path in (
        stage_path,
        paired_path,
        efficiency_path,
        raw_final_path,
    ):
        require_file(path)

    return (
        read_csv(stage_path),
        read_csv(paired_path),
        read_csv(efficiency_path),
        read_csv(raw_final_path),
    )


def build_stage_lookup(stage_rows):
    return {
        (
            row["stage"],
            row["model"],
            row["metric"],
        ): row
        for row in stage_rows
    }


# ============================================================
# Formal figure helpers
# ============================================================
def generate_temporal_metric(
    stage_rows,
    output_dir,
    metric,
    ylabel,
    title,
    basename,
):
    """
    Prequential interpretation:
    each chunk is evaluated before that same chunk is incorporated
    into OnlineIBPR / Full Retrain.
    """
    lookup = build_stage_lookup(stage_rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    x = np.arange(len(STAGE_ORDER))

    for model in MODELS:
        means = []
        lower_errors = []
        upper_errors = []

        for stage in STAGE_ORDER:
            row = lookup[(stage, model, metric)]

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
    ax.set_xlabel("Etapa de evaluación prequential")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        basename,
    )


def generate_online_vs_stale_deltas(
    paired_rows,
    output_dir,
):
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

    fig, ax = plt.subplots(figsize=(8.2, 4.9))

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
        "fig_04_deltas_online_vs_stale_ic95",
    )


def generate_adaptation_cost(
    efficiency_rows,
    output_dir,
):
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

    fig, ax = plt.subplots(figsize=(7.4, 4.9))

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
        "fig_05_coste_adaptacion_log",
    )


def generate_cumulative_compute(
    stage_rows,
    output_dir,
):
    lookup = build_stage_lookup(stage_rows)

    x = np.arange(len(STAGE_ORDER))

    model_values = {}

    for model in MODELS:
        values = []

        for stage in STAGE_ORDER:
            row = lookup[
                (
                    stage,
                    model,
                    "cumulative_compute_time_s",
                )
            ]
            values.append(float(row["mean"]))

        model_values[model] = values

    fig, ax = plt.subplots(figsize=(8.2, 4.9))

    # 1) Full Retrain: separado claramente del resto
    ax.plot(
        x,
        model_values["IBPR_FULL_RETRAIN"],
        marker="o",
        linewidth=2.2,
        label=MODEL_LABELS["IBPR_FULL_RETRAIN"],
        zorder=2,
    )

    # 2) OnlineIBPR: se dibuja primero entre los dos casi superpuestos
    ax.plot(
        x,
        model_values["ONLINE_IBPR"],
        marker="o",
        linewidth=2.2,
        label=MODEL_LABELS["ONLINE_IBPR"],
        zorder=3,
    )

    # 3) IBPR Stale: se dibuja encima, punteado, con más separación
    #    para que se siga viendo la línea naranja por debajo.
    ax.plot(
        x,
        model_values["IBPR_STALE"],
        marker="s",
        linewidth=2.0,
        linestyle=(0, (12, 8)),
        label=MODEL_LABELS["IBPR_STALE"],
        zorder=4,
    )

    ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [STAGE_LABELS[stage] for stage in STAGE_ORDER]
    )

    ax.set_xlabel("Etapa de evaluación prequential")
    ax.set_ylabel(
        "Coste computacional acumulado (s, escala log)"
    )

    ax.set_title(
        "Evolución del coste computacional acumulado"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="upper left",
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_06_coste_acumulado_prequential_log",
    )

def generate_seed_variability(
    raw_final_rows,
    output_dir,
    metric,
    ylabel,
    title,
    basename,
):
    grouped = {
        model: []
        for model in MODELS
    }

    for row in raw_final_rows:
        model = row["model"]

        if model in grouped:
            grouped[model].append(
                (
                    int(row["seed"]),
                    float(row[metric]),
                )
            )

    data = []
    labels = []

    for model in MODELS:
        values = [
            value
            for _, value in sorted(grouped[model])
        ]

        if not values:
            raise ValueError(
                f"No hay datos para {model} / {metric}."
            )

        data.append(values)
        labels.append(MODEL_LABELS[model])

    fig, ax = plt.subplots(figsize=(8.0, 4.9))

    ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )

    for model_index, model in enumerate(MODELS, start=1):
        seed_values = sorted(grouped[model])

        offsets = np.linspace(
            -0.08,
            0.08,
            num=len(seed_values),
        )

        for offset, (seed, value) in zip(
            offsets,
            seed_values,
        ):
            ax.scatter(
                model_index + offset,
                value,
                s=24,
            )
            ax.annotate(
                str(seed),
                (
                    model_index + offset,
                    value,
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        basename,
    )


# ============================================================
# Figure manifest
# ============================================================
def write_manifest(output_dir, timestamp):
    manifest_path = output_dir / "figures_manifest.csv"

    rows = [
        {
            "figure_id": "fig_01",
            "file_base": "fig_01_ndcg20_evolucion_prequential",
            "hypothesis": "H1 / H3",
            "purpose": (
                "Mostrar la evolución prequential de NDCG@20 "
                "entre IBPR Stale, OnlineIBPR y Full Retrain."
            ),
        },
        {
            "figure_id": "fig_02",
            "file_base": "fig_02_recall20_evolucion_prequential",
            "hypothesis": "H1 / H3",
            "purpose": (
                "Mostrar la evolución prequential de Recall@20 "
                "durante el flujo incremental."
            ),
        },
        {
            "figure_id": "fig_03",
            "file_base": "fig_03_precision20_evolucion_prequential",
            "hypothesis": "H1 / H3",
            "purpose": (
                "Mostrar la evolución prequential de Precision@20 "
                "durante el flujo incremental."
            ),
        },
        {
            "figure_id": "fig_04",
            "file_base": "fig_04_deltas_online_vs_stale_ic95",
            "hypothesis": "H1",
            "purpose": (
                "Mostrar el delta medio OnlineIBPR - IBPR Stale "
                "y su IC95% para las métricas de calidad."
            ),
        },
        {
            "figure_id": "fig_05",
            "file_base": "fig_05_coste_adaptacion_log",
            "hypothesis": "H2",
            "purpose": (
                "Comparar el coste acumulado de adaptación "
                "OnlineIBPR frente a Full Retrain."
            ),
        },
        {
            "figure_id": "fig_06",
            "file_base": "fig_06_coste_acumulado_prequential_log",
            "hypothesis": "H2",
            "purpose": (
                "Mostrar cómo evoluciona el coste computacional "
                "acumulado de los tres modelos a lo largo del stream."
            ),
        },
        {
            "figure_id": "fig_07",
            "file_base": "fig_07_ndcg20_variabilidad_seeds_holdout",
            "hypothesis": "H1 / H3",
            "purpose": (
                "Mostrar la variabilidad entre seeds de NDCG@20 "
                "en el holdout final."
            ),
        },
        {
            "figure_id": "fig_08",
            "file_base": "fig_08_recall20_variabilidad_seeds_holdout",
            "hypothesis": "H1 / H3",
            "purpose": (
                "Mostrar la variabilidad entre seeds de Recall@20 "
                "en el holdout final."
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


# ============================================================
# Main
# ============================================================
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
        raw_final_rows,
    ) = load_analysis_files(
        results_dir,
        args.timestamp,
    )

    # H1 / H3 - Prequential recommendation quality
    generate_temporal_metric(
        stage_rows=stage_rows,
        output_dir=output_dir,
        metric=f"NDCG@{TOP_K}",
        ylabel=f"NDCG@{TOP_K}",
        title=(
            f"Evolución prequential de NDCG@{TOP_K} "
            "durante el flujo incremental"
        ),
        basename="fig_01_ndcg20_evolucion_prequential",
    )

    generate_temporal_metric(
        stage_rows=stage_rows,
        output_dir=output_dir,
        metric=f"Recall@{TOP_K}",
        ylabel=f"Recall@{TOP_K}",
        title=(
            f"Evolución prequential de Recall@{TOP_K} "
            "durante el flujo incremental"
        ),
        basename="fig_02_recall20_evolucion_prequential",
    )

    generate_temporal_metric(
        stage_rows=stage_rows,
        output_dir=output_dir,
        metric=f"Precision@{TOP_K}",
        ylabel=f"Precision@{TOP_K}",
        title=(
            f"Evolución prequential de Precision@{TOP_K} "
            "durante el flujo incremental"
        ),
        basename="fig_03_precision20_evolucion_prequential",
    )

    # H1 - Paired statistical differences
    generate_online_vs_stale_deltas(
        paired_rows,
        output_dir,
    )

    # H2 - Adaptation and cumulative compute cost
    generate_adaptation_cost(
        efficiency_rows,
        output_dir,
    )

    generate_cumulative_compute(
        stage_rows,
        output_dir,
    )

    # H1 / H3 - Stability across random seeds
    generate_seed_variability(
        raw_final_rows=raw_final_rows,
        output_dir=output_dir,
        metric=f"NDCG@{TOP_K}",
        ylabel=f"NDCG@{TOP_K}",
        title=(
            f"Variabilidad entre seeds de NDCG@{TOP_K} "
            "en el holdout final"
        ),
        basename="fig_07_ndcg20_variabilidad_seeds_holdout",
    )

    generate_seed_variability(
        raw_final_rows=raw_final_rows,
        output_dir=output_dir,
        metric=f"Recall@{TOP_K}",
        ylabel=f"Recall@{TOP_K}",
        title=(
            f"Variabilidad entre seeds de Recall@{TOP_K} "
            "en el holdout final"
        ),
        basename="fig_08_recall20_variabilidad_seeds_holdout",
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
