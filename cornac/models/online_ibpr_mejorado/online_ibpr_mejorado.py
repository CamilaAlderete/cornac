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
import torch
import torch.nn.functional as F

# Helpers para lograr el entrenamiento parcial
def _build_user_positive_sets(history_csr):
    """Construye, para cada usuario, el conjunto de ítems positivos históricos."""
    user_pos_sets = []
    for u in range(history_csr.shape[0]):
        start = history_csr.indptr[u]
        end = history_csr.indptr[u + 1]
        user_pos_sets.append(set(history_csr.indices[start:end]))
    return user_pos_sets


def _sample_negatives_uniform(batch_u, user_pos_sets, n_items, rng):
    """Muestrea un ítem negativo j para cada usuario del batch."""
    batch_j = np.empty(len(batch_u), dtype=np.int64)

    for idx, u in enumerate(batch_u):
        positives = user_pos_sets[u]

        # Caso extremo: si el usuario ya interactuó con todo, no hay negativo válido
        if len(positives) >= n_items:
            raise ValueError(
                f"El usuario {u} no tiene ítems negativos disponibles para muestreo."
            )

        j = rng.integers(0, n_items)
        while j in positives:
            j = rng.integers(0, n_items)

        batch_j[idx] = j

    return batch_j


def _iter_recent_batches(recent_pairs, batch_size, shuffle, rng):
    """Itera batches de pares recientes (u, i)."""
    recent_pairs = np.asarray(recent_pairs, dtype=np.int64)

    if recent_pairs.ndim != 2 or recent_pairs.shape[1] != 2:
        raise ValueError("recent_pairs debe tener shape (n, 2) con columnas [u, i].")

    indices = np.arange(len(recent_pairs))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = recent_pairs[batch_idx]
        yield batch[:, 0], batch[:, 1]

def online_ibpr_mejorado(
    train_set=None,
    k=20,
    lamda=0.005,
    n_epochs=150,
    learning_rate=0.001,
    batch_size=100,
    init_params=None,
    update_V=False,
    neg_sampling="uniform",
    normalize=False,
    verbose=False,
    recent_pairs=None,
    history_csr=None,
    max_steps=None,
    random_seed=42,
    loss_mode="angular",
):
    """Online training loop for Indexable BPR.

    Supports two execution modes:

    1) Classic mode:
       Uses `train_set.uij_iter(...)` to draw triplets (u, i, j) from a Cornac dataset.

    2) Partial-update mode:
       Uses only `recent_pairs` as new positive events (u, i), while `history_csr`
       is used to sample valid negatives j for each user.

    Parameters
    ----------
    train_set: cornac.data.Dataset, optional
        Dataset providing user-item interactions and utilities for negative sampling.
        Required only in classic mode.

    k: int
        Latent dimension.

    lamda: float
        L2 regularization strength.

    n_epochs: int
        Number of passes over the mini-batch stream.

    learning_rate: float
        Learning rate for Adam optimizer.

    batch_size: int
        Mini-batch size for SGD/Adam.

    init_params: dict, optional
        Optional initial parameters: {"U": np.ndarray, "V": np.ndarray}.

    update_V: bool, default=False
        When True, both U and V are updated online. When False, only U is updated.

    neg_sampling: {"uniform", "popularity"}, default="uniform"
        Negative sampling strategy. In partial-update mode, only "uniform" is supported
        in this first version.

    normalize: bool, default=False
        If True, L2-normalizes U and V at the end.

    verbose: bool, default=False
        If True, prints training progress.

    recent_pairs: np.ndarray, optional
        Array of shape (n, 2) with recent interactions [u, i]. If provided, the function
        enters partial-update mode.

    history_csr: scipy.sparse.csr_matrix, optional
        Global user-item history used for valid negative sampling in partial-update mode.

    max_steps: int, optional
        Maximum number of mini-batches to process per call in partial-update mode.

    random_seed: int, optional
        Random seed for shuffling recent_pairs and sampling negatives.

    Returns
    -------
    dict
        {"U": np.ndarray, "V": np.ndarray}

    Notes
    -----
    - The function supports both classic full-dataset training and partial online updates.
    - In partial-update mode, recent_pairs define positives, while history_csr defines
      which items cannot be sampled as negatives.
    """

    # Use CSR for shapes; we don't materialize COO triplets here on purpose.
    use_recent_mode = recent_pairs is not None

    if use_recent_mode:
        if history_csr is None:
            raise ValueError(
                "En modo recent_pairs debes proveer history_csr para negative sampling."
            )

        X = history_csr
        recent_pairs = np.asarray(recent_pairs, dtype=np.int64)

        if recent_pairs.ndim != 2 or recent_pairs.shape[1] != 2:
            raise ValueError("recent_pairs debe tener shape (n, 2) con columnas [u, i].")

        if len(recent_pairs) > 0:
            max_user_idx = X.shape[0] - 1
            max_item_idx = X.shape[1] - 1

            if np.any(recent_pairs[:, 0] < 0) or np.any(recent_pairs[:, 0] > max_user_idx):
                raise ValueError(
                    f"recent_pairs contiene user_idx fuera de rango. "
                    f"Rango válido: [0, {max_user_idx}]"
                )

            if np.any(recent_pairs[:, 1] < 0) or np.any(recent_pairs[:, 1] > max_item_idx):
                raise ValueError(
                    f"recent_pairs contiene item_idx fuera de rango. "
                    f"Rango válido: [0, {max_item_idx}]"
                )

        if len(recent_pairs) == 0:
            # No hay nada nuevo que aprender; devolvemos los parámetros actuales
            if init_params is None or init_params.get("U") is None or init_params.get("V") is None:
                raise ValueError(
                    "recent_pairs está vacío y no hay init_params válidos para devolver."
                )

    else:
        if train_set is None:
            raise ValueError(
                "Debes proveer train_set o bien recent_pairs + history_csr."
            )
        X = train_set.csr_matrix

    # Initialize user/item factors; warm-start if provided.
    if init_params is None:
        init_params = {"U": None, "V": None}

    if init_params.get("U") is None:
        U = torch.randn(X.shape[0], k, requires_grad=True)
    else:
        U_np = init_params["U"]
        if U_np.shape[1] != k:
            raise ValueError(
                f"init_params['U'] has k={U_np.shape[1]} but k={k} was requested."
            )
        U = torch.from_numpy(U_np).clone().detach().requires_grad_(True)
        # Expand for new users if needed (online-friendly). Keeping extra rows (if any).
        if U.shape[0] < X.shape[0]:
            pad = torch.randn(X.shape[0] - U.shape[0], k)
            U = torch.cat([U, pad], dim=0).requires_grad_(True)

    if init_params.get("V") is None:
        # If items are kept fixed (update_V=False), avoid tracking gradients for V
        V = torch.randn(X.shape[1], k, requires_grad=update_V)
    else:
        V_np = init_params["V"]
        if V_np.shape[1] != k:
            raise ValueError(
                f"init_params['V'] has k={V_np.shape[1]} but k={k} was requested."
            )
        V = (
            torch.from_numpy(V_np).clone().detach().requires_grad_(update_V)
        )
        # Expand for new items if needed (online-friendly). Keeping extra rows (if any).
        if V.shape[0] < X.shape[1]:
            pad = torch.randn(X.shape[1] - V.shape[0], k)
            V = torch.cat([V, pad], dim=0).requires_grad_(update_V)

    # Optimizer: by default update only U for fast online behavior.
    params = [U] + ([V] if update_V else [])
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Small epsilon to guard against zero norms in normalization.
    eps = 1e-12

    V_norm_fixed = None
    if not update_V:
        with torch.no_grad():
            V_norm_fixed = V / (V.norm(dim=1, keepdim=True) + eps)

    rng = np.random.default_rng(random_seed)

    if loss_mode not in {"angular", "cosine_bpr"}:
        raise ValueError("loss_mode debe ser 'angular' o 'cosine_bpr'.")

    user_pos_sets = None
    if use_recent_mode:
        if neg_sampling != "uniform":
            raise NotImplementedError(
                "En esta primera versión del modo recent_pairs, usa neg_sampling='uniform'."
            )
        user_pos_sets = _build_user_positive_sets(X)

    stop_early = False
    total_steps = 0

    for epoch in range(1, n_epochs + 1):
        sum_loss = 0.0
        count = 0

        if use_recent_mode:
            batch_iterator = _iter_recent_batches(
                recent_pairs=recent_pairs,
                batch_size=batch_size,
                shuffle=True,
                rng=rng,
            )
        else:
            batch_iterator = train_set.uij_iter(
                batch_size=batch_size,
                shuffle=True,
                neg_sampling=neg_sampling,
            )

        for batch in batch_iterator:
            if use_recent_mode:
                batch_u, batch_i = batch
                batch_j = _sample_negatives_uniform(
                    batch_u=batch_u,
                    user_pos_sets=user_pos_sets,
                    n_items=X.shape[1],
                    rng=rng,
                )
            else:
                batch_u, batch_i, batch_j = batch

            # Ensure integer Long indices for PyTorch indexing (robust to dtype issues)
            bu = torch.as_tensor(batch_u, dtype=torch.long, device=U.device)
            bi = torch.as_tensor(batch_i, dtype=torch.long, device=U.device)
            bj = torch.as_tensor(batch_j, dtype=torch.long, device=U.device)

            # Gather batch factors
            regU = U[bu, :]

            if update_V:
                regI = V[bi, :]
                regJ = V[bj, :]
            else:
                regI = None
                regJ = None

            # Batch factors are used directly for regularization to avoid NumPy unique/union overhead.

            # Normalize to compute angular distances (as in IBPR)
            regU_norm = regU / (regU.norm(dim=1, keepdim=True) + eps)

            if update_V:
                regI_norm = regI / (regI.norm(dim=1, keepdim=True) + eps)
                regJ_norm = regJ / (regJ.norm(dim=1, keepdim=True) + eps)
            else:
                regI_norm = V_norm_fixed[bi, :]
                regJ_norm = V_norm_fixed[bj, :]

            # Pairwise similarity scores
            dot_ui = torch.sum(regU_norm * regI_norm, dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
            dot_uj = torch.sum(regU_norm * regJ_norm, dim=1).clamp(-1 + 1e-7, 1 - 1e-7)

            if loss_mode == "angular":
                score_diff = torch.acos(dot_uj) - torch.acos(dot_ui)
                rank_loss = F.softplus(-score_diff).sum()
            else:  # loss_mode == "cosine_bpr"
                score_diff = dot_ui - dot_uj
                rank_loss = F.softplus(-score_diff).sum()

            if update_V:
                reg_term = regU.norm().pow(2) + regI.norm().pow(2) + regJ.norm().pow(2)
            else:
                reg_term = regU.norm().pow(2)

            loss = lamda * reg_term + rank_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Book-keeping for reporting
            sum_loss += loss.detach().item()
            count += len(batch_u)
            total_steps += 1

            if use_recent_mode and max_steps is not None and total_steps >= max_steps:
                stop_early = True
                break

        if stop_early:
            break

        if verbose:
            avg_loss = sum_loss / max(count, 1)
            print(f"Epoch {epoch}/{n_epochs} - avg loss per sample: {avg_loss:.6f}")

    # Optional normalization for deployment consistency.
    if normalize:
        # Normalize both U and V for scoring consistency regardless of update_V
        with torch.no_grad():
            U = torch.nn.functional.normalize(U, p=2, dim=1)
            V = torch.nn.functional.normalize(V, p=2, dim=1)

    U = U.data.cpu().numpy()
    V = V.data.cpu().numpy()

    return {"U": U, "V": V}
