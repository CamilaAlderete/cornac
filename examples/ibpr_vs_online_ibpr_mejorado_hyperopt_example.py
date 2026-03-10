"""Experiment: IBPR vs OnlineIBPRMejorado with hyper-parameter tuning on Cornac.

This script compares:
1) Untuned IBPR
2) Untuned OnlineIBPRMejorado
3) Tuned IBPR via Cornac GridSearch
4) Tuned OnlineIBPRMejorado via Cornac GridSearch

Why GridSearch instead of RandomSearch?
- Cornac officially supports both GridSearch and RandomSearch.
- GridSearch only needs discrete domains.
- RandomSearch internally samples with `self.model.seed`; since these custom wrappers
  do not expose a `seed` parameter, GridSearch is the safer built-in choice here.

Protocol:
- Dataset: MovieLens 100K
- Feedback: implicit via rating_threshold=3.0
- Split: train / validation / test = 0.8 / 0.1 / 0.1
- Tuning metric: NDCG@20
- Final comparison metrics: Recall@20, Precision@20, NDCG@20, AUC, MAP
"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.hyperopt import Discrete, GridSearch
import os
import sys
from datetime import datetime
from contextlib import redirect_stdout
from cornac.utils.Tee import Tee


# Custom/local models already integrated in the Cornac package tree
from cornac.models.ibpr import IBPR
from cornac.models.online_ibpr_mejorado import OnlineIBPRMejorado

SEED = 42
RATING_THRESHOLD = 3.0
TOP_K = 20


def build_eval_method():
    """Create a split with validation, required by Cornac hyperopt."""
    data = movielens.load_feedback(variant="100K")
    return RatioSplit(
        data=data,
        test_size=0.1,
        val_size=0.1,
        rating_threshold=RATING_THRESHOLD,
        exclude_unknowns=True,
        verbose=True,
        seed=SEED,
    )


def build_metrics():
    return [
        cornac.metrics.Recall(k=TOP_K),
        cornac.metrics.Precision(k=TOP_K),
        cornac.metrics.NDCG(k=TOP_K),
        cornac.metrics.AUC(),
        cornac.metrics.MAP(),
    ]


def build_default_models():
    ibpr_default = IBPR(
        k=20,
        max_iter=50,
        learning_rate=0.01,
        lamda=0.001,
        batch_size=256,
        verbose=True,
        name="IBPR (default)",
    )

    online_default = OnlineIBPRMejorado(
        k=20,
        max_iter=50,
        learning_rate=0.01,
        lamda=0.001,
        batch_size=256,
        update_V=False,
        neg_sampling="uniform",
        normalize=True,
        loss_mode="cosine_bpr",
        verbose=True,
        name="OnlineIBPRMejorado (default)",
    )

    return ibpr_default, online_default


def build_tuned_models(eval_method):
    tune_metric = cornac.metrics.NDCG(k=TOP_K)

    # Baselines for search
    ibpr_base = IBPR(
        verbose=True,
        name="IBPR (tuned)",
    )

    online_base = OnlineIBPRMejorado(
        verbose=True,
        name="OnlineIBPRMejorado (tuned)",
    )

    # Keep spaces moderate so the experiment remains practical.
    # You can expand them later if needed.
    ibpr_space = [
        Discrete(name="k", values=[20, 50]),
        Discrete(name="max_iter", values=[30, 50]),
        Discrete(name="learning_rate", values=[0.001, 0.005, 0.01]),
        Discrete(name="lamda", values=[0.0005, 0.001, 0.005]),
        Discrete(name="batch_size", values=[256, 512]),
    ]

    online_space = [
        Discrete(name="k", values=[20, 50]),
        Discrete(name="max_iter", values=[10, 30, 50]),
        Discrete(name="learning_rate", values=[0.001, 0.005, 0.01]),
        Discrete(name="lamda", values=[0.0005, 0.001, 0.005]),
        Discrete(name="batch_size", values=[256, 512]),
        Discrete(name="update_V", values=[False]),
        Discrete(name="neg_sampling", values=["uniform"]),
        Discrete(name="normalize", values=[True]),
        Discrete(name="loss_mode", values=["angular", "cosine_bpr"]),
    ]

    ibpr_search = GridSearch(
        model=ibpr_base,
        space=ibpr_space,
        metric=tune_metric,
        eval_method=eval_method,
    )

    online_search = GridSearch(
        model=online_base,
        space=online_space,
        metric=tune_metric,
        eval_method=eval_method,
    )

    return ibpr_search, online_search


def print_best_search_results(search_models):
    print("\n" + "=" * 80)
    print("BEST HYPER-PARAMETERS FOUND")
    print("=" * 80)
    for search_model in search_models:
        print(f"{search_model.name}")
        print(f"  best_params = {getattr(search_model, 'best_params', None)}")
        print(f"  best_score  = {getattr(search_model, 'best_score', None)}")
        print("-" * 80)


def main():
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/ibpr_vs_online_ibpr_{timestamp}.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)

        with redirect_stdout(tee):
            eval_method = build_eval_method()
            metrics = build_metrics()

            ibpr_default, online_default = build_default_models()
            ibpr_search, online_search = build_tuned_models(eval_method)

            models = [
                ibpr_default,
                online_default,
                # ibpr_search,
                # online_search,
            ]

            experiment = cornac.Experiment(
                eval_method=eval_method,
                models=models,
                metrics=metrics,
                user_based=True,
            )
            experiment.run()

            print_best_search_results([ibpr_default, online_default])  # , ibpr_search, online_search])


if __name__ == "__main__":
    main()
