# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np

from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException


class OnlineIBPRMejorado(Recommender, ANNMixin):
    """Online Indexable Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 20
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.05
        The learning rate for SGD.

    lamda: float, optional, default: 0.001
        The regularization parameter.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'IBRP'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} \
        please see below the definition of U and V.

        U: csc_matrix, shape (n_users,k)
            The user latent factors, optional initialization via init_params.

        V: csc_matrix, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    update_V: bool, optional, default: False
        Whether to update item factors online. When False, only user factors are
        updated for faster online adaptation.

    neg_sampling: str, optional, default: 'uniform'
        Negative sampling strategy passed to `train_set.uij_iter`, options are
        {'uniform', 'popularity'}.

    normalize: bool, optional, default: False
        If True, L2-normalize user and item factors after training to keep
        scoring consistent with the angular objective.

    References
    ----------
    * Le, D. D., & Lauw, H. W. (2017, November). Indexable Bayesian personalized ranking for efficient top-k recommendation.\
      In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (pp. 1389-1398). ACM.
    """

    def __init__(
        self,
        k=20,
        max_iter=100,
        learning_rate=0.05,
        lamda=0.001,
        batch_size=100,
        name="online_ibpr_mejorado",
        trainable=True,
        verbose=False,
        init_params=None,
        update_V=False,
        neg_sampling="uniform",
        normalize=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.name = name
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.batch_size = batch_size

        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)  # matrix of user factors
        self.V = self.init_params.get("V", None)  # matrix of item factors
        # Extended configuration
        self.update_V = update_V
        self.neg_sampling = neg_sampling
        self.normalize = normalize

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """

        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            from .online_ibpr_mejorado import online_ibpr_mejorado

            res = online_ibpr_mejorado(
                train_set,
                k=self.k,
                n_epochs=self.max_iter,
                lamda=self.lamda,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                init_params={"U": self.U, "V": self.V},
                update_V=self.update_V,
                neg_sampling=self.neg_sampling,
                normalize=self.normalize,
                verbose=self.verbose,
            )
            self.U = np.asarray(res["U"])
            self.V = np.asarray(res["V"])

            if self.verbose:
                print("Learning completed")

        return self

    def partial_fit_recent(self, recent_pairs, history_csr, max_steps=None, n_epochs=1):
        """Actualiza el modelo usando solo interacciones recientes.

        Parameters
        ----------
        recent_pairs : np.ndarray
            Array shape (n, 2) con pares [u, i] recientes.
        history_csr : scipy.sparse.csr_matrix
            Matriz global usuario-item para negative sampling.
        max_steps : int, optional
            Máximo número de mini-batches a procesar.
        n_epochs : int, optional
            Cantidad de pasadas sobre recent_pairs. En online suele ser 1.

        Returns
        -------
        self
        """
        if self.trainable:
            from .online_ibpr_mejorado import online_ibpr_mejorado

            res = online_ibpr_mejorado(
                train_set=None,
                k=self.k,
                n_epochs=n_epochs,
                lamda=self.lamda,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                init_params={"U": self.U, "V": self.V},
                update_V=self.update_V,
                neg_sampling=self.neg_sampling,
                normalize=self.normalize,
                verbose=self.verbose,
                recent_pairs=recent_pairs,
                history_csr=history_csr,
                max_steps=max_steps,
            )

            self.U = np.asarray(res["U"])
            self.V = np.asarray(res["V"])

            if self.verbose:
                print("Partial learning completed")

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if item_idx is None:
            return self.V.dot(self.U[user_idx, :])

        return self.V[item_idx, :].dot(self.U[user_idx, :])

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        measure: MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of user vectors for all users available in the model.
        """
        return self.U

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        return self.V
