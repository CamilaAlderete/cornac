"""FASE B1 — auditoría ampliada de invariantes de OnlineIBPRMejorado.

Este archivo NO modifica la implementación. Separa:
1) invariantes obligatorios para el camino científico warm-start incremental;
2) caracterización explícita de comportamientos actuales que todavía no se
   consideran requisitos científicos.

Ejecutar dentro del entorno Cornac del proyecto:
    python -m unittest -v test_online_ibpr_mejorado_invariants_v2.py
"""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from cornac.models.online_ibpr_mejorado.online_ibpr_mejorado import (
    online_ibpr_mejorado,
    _build_user_positive_sets,
    _sample_negatives_uniform,
)
from cornac.models.online_ibpr_mejorado.recom_online_ibpr_mejorado import (
    OnlineIBPRMejorado,
)


class FakeRNG:
    """RNG controlado para pruebas deterministas del negative sampling."""

    def __init__(self, values):
        self.values = iter(values)

    def integers(self, *args, **kwargs):
        return next(self.values)


class OnlineIBPRAuditFixture(unittest.TestCase):
    def setUp(self):
        self.k = 3
        self.history = csr_matrix(
            np.array(
                [
                    [1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [1, 0, 0, 1, 0],
                ],
                dtype=np.float32,
            )
        )
        # Intencionalmente no unitarias para caracterizar normalize=True.
        self.U = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=np.float32,
        )
        self.V = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.recent_pairs = np.array([[0, 2], [1, 3]], dtype=np.int64)

    def run_partial(
        self,
        *,
        update_V=False,
        normalize=False,
        seed=42,
        loss_mode="cosine_bpr",
        recent_pairs=None,
        history_csr=None,
        max_steps=None,
        n_epochs=1,
        batch_size=2,
        neg_sampling="uniform",
    ):
        return online_ibpr_mejorado(
            train_set=None,
            k=self.k,
            lamda=0.001,
            n_epochs=n_epochs,
            learning_rate=0.05,
            batch_size=batch_size,
            init_params={"U": self.U.copy(), "V": self.V.copy()},
            update_V=update_V,
            neg_sampling=neg_sampling,
            normalize=normalize,
            recent_pairs=self.recent_pairs if recent_pairs is None else recent_pairs,
            history_csr=self.history if history_csr is None else history_csr,
            max_steps=max_steps,
            random_seed=seed,
            loss_mode=loss_mode,
        )


class TestCriticalInvariants(OnlineIBPRAuditFixture):
    """Invariantes que sí condicionan el camino científico propuesto."""

    def test_partial_requires_warm_start(self):
        with self.assertRaises(ValueError):
            online_ibpr_mejorado(
                train_set=None,
                k=self.k,
                recent_pairs=self.recent_pairs,
                history_csr=self.history,
                update_V=False,
                init_params=None,
            )

    def test_update_v_false_updates_u_and_keeps_v_bitwise_exact(self):
        result = self.run_partial(update_V=False, normalize=False)
        self.assertFalse(np.array_equal(self.U, result["U"]))
        np.testing.assert_array_equal(self.V, result["V"])

    def test_update_v_false_normalize_true_still_keeps_v_bitwise_exact(self):
        result = self.run_partial(update_V=False, normalize=True)
        np.testing.assert_array_equal(self.V, result["V"])

    def test_update_v_true_updates_v(self):
        result = self.run_partial(update_V=True, normalize=False)
        self.assertFalse(np.array_equal(self.V, result["V"]))

    def test_negative_sampling_rejects_history_and_current_positive(self):
        user_pos_sets = _build_user_positive_sets(self.history, [0])
        user_pos_sets[0].add(2)  # positivo reciente
        sampled = _sample_negatives_uniform(
            batch_u=np.array([0]),
            user_pos_sets=user_pos_sets,
            n_items=5,
            rng=FakeRNG([2, 1, 4]),
        )
        self.assertEqual(int(sampled[0]), 4)

    def test_user_without_available_negative_is_rejected(self):
        full_history = csr_matrix(np.ones((1, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            online_ibpr_mejorado(
                train_set=None,
                k=3,
                lamda=0.001,
                n_epochs=1,
                learning_rate=0.01,
                batch_size=1,
                init_params={
                    "U": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                    "V": np.eye(3, dtype=np.float32),
                },
                update_V=False,
                neg_sampling="uniform",
                normalize=False,
                recent_pairs=np.array([[0, 1]], dtype=np.int64),
                history_csr=full_history,
                random_seed=42,
                loss_mode="cosine_bpr",
            )

    def test_partial_rejects_non_uniform_negative_sampling(self):
        with self.assertRaises(NotImplementedError):
            self.run_partial(neg_sampling="popularity")

    def test_invalid_loss_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_partial(loss_mode="not_a_loss")

    def test_invalid_recent_pair_indices_are_rejected(self):
        bad_cases = [
            np.array([[-1, 2]], dtype=np.int64),
            np.array([[3, 2]], dtype=np.int64),
            np.array([[0, -1]], dtype=np.int64),
            np.array([[0, 5]], dtype=np.int64),
        ]
        for bad in bad_cases:
            with self.subTest(recent_pairs=bad.tolist()):
                with self.assertRaises(ValueError):
                    self.run_partial(recent_pairs=bad)

    def test_invalid_factor_shapes_are_rejected(self):
        bad_U = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            online_ibpr_mejorado(
                k=self.k,
                init_params={"U": bad_U, "V": self.V.copy()},
                update_V=False,
                recent_pairs=self.recent_pairs,
                history_csr=self.history,
            )

        with self.assertRaises(ValueError):
            online_ibpr_mejorado(
                k=self.k,
                init_params={"U": self.U[:2].copy(), "V": self.V.copy()},
                update_V=False,
                recent_pairs=self.recent_pairs,
                history_csr=self.history,
            )

    def test_empty_update_is_bitwise_identity_even_with_normalize_true(self):
        result = self.run_partial(
            recent_pairs=np.empty((0, 2), dtype=np.int64),
            normalize=True,
        )
        np.testing.assert_array_equal(self.U, result["U"])
        np.testing.assert_array_equal(self.V, result["V"])

    def test_same_seed_is_deterministic_for_both_losses(self):
        for loss_mode in ("angular", "cosine_bpr"):
            with self.subTest(loss_mode=loss_mode):
                result_1 = self.run_partial(seed=17, loss_mode=loss_mode)
                result_2 = self.run_partial(seed=17, loss_mode=loss_mode)
                np.testing.assert_array_equal(result_1["U"], result_2["U"])
                np.testing.assert_array_equal(result_1["V"], result_2["V"])

    def test_max_steps_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_partial(max_steps=0)

    def test_max_steps_one_is_global_across_epochs(self):
        one_epoch = self.run_partial(
            batch_size=1,
            n_epochs=1,
            max_steps=1,
            seed=21,
        )
        many_epochs = self.run_partial(
            batch_size=1,
            n_epochs=5,
            max_steps=1,
            seed=21,
        )
        np.testing.assert_array_equal(one_epoch["U"], many_epochs["U"])
        np.testing.assert_array_equal(one_epoch["V"], many_epochs["V"])

    def test_nonaffected_user_is_exact_when_normalize_false(self):
        result = self.run_partial(normalize=False)
        # user 2 no aparece en recent_pairs
        np.testing.assert_array_equal(self.U[2], result["U"][2])

    def test_wrapper_empty_update_does_not_consume_seed(self):
        model = OnlineIBPRMejorado(
            k=self.k,
            init_params={"U": self.U.copy(), "V": self.V.copy()},
            update_V=False,
            normalize=False,
            loss_mode="cosine_bpr",
            seed=42,
        )
        self.assertEqual(model._partial_update_count, 0)
        model.partial_fit_recent(recent_pairs=[], history_csr=self.history)
        self.assertEqual(model._partial_update_count, 0)
        model.partial_fit_recent(
            recent_pairs=self.recent_pairs,
            history_csr=self.history,
        )
        self.assertEqual(model._partial_update_count, 1)

    def test_wrapper_nonempty_calls_consume_consecutive_seeds(self):
        model = OnlineIBPRMejorado(
            k=self.k,
            init_params={"U": self.U.copy(), "V": self.V.copy()},
            update_V=False,
            normalize=False,
            loss_mode="cosine_bpr",
            seed=42,
        )

        captured = []

        def fake_core(**kwargs):
            captured.append(int(kwargs["random_seed"]))
            return {
                "U": np.asarray(kwargs["init_params"]["U"]).copy(),
                "V": np.asarray(kwargs["init_params"]["V"]).copy(),
            }

        target = (
            "cornac.models.online_ibpr_mejorado.online_ibpr_mejorado."
            "online_ibpr_mejorado"
        )
        with patch(target, side_effect=fake_core):
            model.partial_fit_recent([], self.history)
            model.partial_fit_recent(self.recent_pairs, self.history)
            model.partial_fit_recent([], self.history)
            model.partial_fit_recent(self.recent_pairs, self.history)
            model.partial_fit_recent(self.recent_pairs, self.history)

        # Las llamadas vacías no incrementan el contador: 42, 42, 43, 43, 44.
        self.assertEqual(captured, [42, 42, 43, 43, 44])
        self.assertEqual(model._partial_update_count, 3)

    def test_repeated_update_sequence_is_reproducible_and_keeps_v_exact(self):
        def make_model():
            return OnlineIBPRMejorado(
                k=self.k,
                learning_rate=0.01,
                lamda=0.001,
                batch_size=1,
                init_params={"U": self.U.copy(), "V": self.V.copy()},
                update_V=False,
                normalize=False,
                loss_mode="angular",
                seed=73,
            )

        # Dos llamadas causales: history_2 incorpora el positivo de la primera.
        recent_1 = np.array([[0, 2]], dtype=np.int64)
        history_2_dense = self.history.toarray().copy()
        history_2_dense[0, 2] = 1.0
        history_2 = csr_matrix(history_2_dense)
        recent_2 = np.array([[1, 3]], dtype=np.int64)

        model_a = make_model()
        model_b = make_model()
        for model in (model_a, model_b):
            model.partial_fit_recent(recent_1, self.history, n_epochs=2)
            model.partial_fit_recent(recent_2, history_2, n_epochs=2)

        np.testing.assert_array_equal(model_a.U, model_b.U)
        np.testing.assert_array_equal(model_a.V, model_b.V)
        np.testing.assert_array_equal(self.V, model_a.V)
        self.assertEqual(model_a._partial_update_count, 2)
        self.assertEqual(model_b._partial_update_count, 2)


class TestCharacterizationOnly(OnlineIBPRAuditFixture):
    """Comportamientos observados; NO son requisitos científicos congelados."""

    def test_characterize_normalize_true_normalizes_nonaffected_user_rows(self):
        result = self.run_partial(normalize=True)
        # user 2 no participa en recent_pairs, pero U completa se normaliza al final.
        self.assertFalse(np.array_equal(self.U[2], result["U"][2]))
        np.testing.assert_allclose(
            np.linalg.norm(result["U"][2]),
            1.0,
            rtol=0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
