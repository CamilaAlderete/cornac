import unittest

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
    """RNG controlado para probar negative sampling."""

    def __init__(self, values):
        self.values = iter(values)

    def integers(self, *args, **kwargs):
        return next(self.values)


class TestOnlineIBPRMejoradoInvariants(unittest.TestCase):

    def setUp(self):
        self.k = 3

        # Historial:
        # user 0 -> items 0, 1
        # user 1 -> items 1, 2
        # user 2 -> items 0, 3
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

        self.U = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.V = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Estos positivos NO están todavía en history.
        self.recent_pairs = np.array(
            [
                [0, 2],
                [1, 3],
            ],
            dtype=np.int64,
        )

    def _run_partial(self, update_V=False, seed=42):
        return online_ibpr_mejorado(
            train_set=None,
            k=self.k,
            lamda=0.001,
            n_epochs=1,
            learning_rate=0.05,
            batch_size=2,
            init_params={
                "U": self.U.copy(),
                "V": self.V.copy(),
            },
            update_V=update_V,
            neg_sampling="uniform",
            normalize=False,
            recent_pairs=self.recent_pairs,
            history_csr=self.history,
            max_steps=None,
            random_seed=seed,
            loss_mode="cosine_bpr",
        )

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

    def test_update_v_false_updates_u_and_keeps_v_exact(self):
        result = self._run_partial(update_V=False)

        self.assertFalse(
            np.array_equal(self.U, result["U"]),
            "U debería cambiar después del update.",
        )

        np.testing.assert_array_equal(
            self.V,
            result["V"],
            err_msg="V debe permanecer exactamente igual cuando update_V=False.",
        )

    def test_update_v_true_updates_v(self):
        result = self._run_partial(update_V=True)

        self.assertFalse(
            np.array_equal(self.V, result["V"]),
            "V debería cambiar cuando update_V=True.",
        )

    def test_negative_sampling_rejects_history_and_current_positive(self):
        user_pos_sets = _build_user_positive_sets(
            self.history,
            [0],
        )

        # El item 2 es el nuevo positivo del usuario 0.
        user_pos_sets[0].add(2)

        # Forzamos:
        # 2 -> positivo reciente, debe rechazarse
        # 1 -> positivo histórico, debe rechazarse
        # 4 -> negativo válido
        rng = FakeRNG([2, 1, 4])

        sampled = _sample_negatives_uniform(
            batch_u=np.array([0]),
            user_pos_sets=user_pos_sets,
            n_items=5,
            rng=rng,
        )

        self.assertEqual(sampled[0], 4)

    def test_invalid_shapes_are_rejected(self):
        bad_U = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            online_ibpr_mejorado(
                k=self.k,
                init_params={
                    "U": bad_U,
                    "V": self.V.copy(),
                },
                update_V=False,
                recent_pairs=self.recent_pairs,
                history_csr=self.history,
            )

        wrong_user_count = self.U[:2].copy()

        with self.assertRaises(ValueError):
            online_ibpr_mejorado(
                k=self.k,
                init_params={
                    "U": wrong_user_count,
                    "V": self.V.copy(),
                },
                update_V=False,
                recent_pairs=self.recent_pairs,
                history_csr=self.history,
            )

    def test_empty_update_is_identity(self):
        result = online_ibpr_mejorado(
            k=self.k,
            init_params={
                "U": self.U.copy(),
                "V": self.V.copy(),
            },
            update_V=False,
            recent_pairs=[],
            history_csr=self.history,
            random_seed=42,
        )

        np.testing.assert_array_equal(self.U, result["U"])
        np.testing.assert_array_equal(self.V, result["V"])

    def test_same_seed_produces_same_result(self):
        result_1 = self._run_partial(
            update_V=False,
            seed=42,
        )

        result_2 = self._run_partial(
            update_V=False,
            seed=42,
        )

        np.testing.assert_allclose(
            result_1["U"],
            result_2["U"],
            rtol=0,
            atol=1e-7,
        )

        np.testing.assert_array_equal(
            result_1["V"],
            result_2["V"],
        )

    def test_max_steps_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            online_ibpr_mejorado(
                k=self.k,
                init_params={
                    "U": self.U.copy(),
                    "V": self.V.copy(),
                },
                update_V=False,
                recent_pairs=self.recent_pairs,
                history_csr=self.history,
                max_steps=0,
            )

    def test_empty_wrapper_update_does_not_consume_seed(self):
        model = OnlineIBPRMejorado(
            k=self.k,
            init_params={
                "U": self.U.copy(),
                "V": self.V.copy(),
            },
            update_V=False,
            normalize=False,
            loss_mode="cosine_bpr",
            seed=42,
        )

        self.assertEqual(model._partial_update_count, 0)

        model.partial_fit_recent(
            recent_pairs=[],
            history_csr=self.history,
        )

        self.assertEqual(
            model._partial_update_count,
            0,
            "Un update vacío no debe consumir un seed.",
        )

        model.partial_fit_recent(
            recent_pairs=self.recent_pairs,
            history_csr=self.history,
        )

        self.assertEqual(
            model._partial_update_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()