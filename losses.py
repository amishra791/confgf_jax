from flax import nnx
import jax.numpy as jnp
import jax
import jax.random as jr

from data import MoleculeData
from model import compute_dist
import optax

from typing import List
from tqdm import tqdm

@nnx.jit
def dist_score_matching_loss(model, mol_data: MoleculeData, sigmas, sigma_idx, z_d):
    sigma = sigmas[sigma_idx]
    edge_mask = mol_data.edge_mask
    m_all = edge_mask.astype(z_d.dtype)

    noise = z_d * sigma
    true_dist = compute_dist(mol_data.pos, mol_data.edge_index)
    perturbed_dist = true_dist + noise
    target = -1. / (sigma ** 2) * noise
    dist_scores = model(mol_data, perturbed_dist, sigma)

    per_edge = (dist_scores - target) ** 2

    def masked_mean(x, mask_f):
        denom = jnp.sum(mask_f) + 1e-8
        return jnp.sum(x * mask_f) / denom, denom

    mse_all, n_all = masked_mean(per_edge, m_all)
    rmse_all = jnp.sqrt(mse_all + 1e-12)

    target_mse_all, _ = masked_mean(target ** 2, m_all)
    rel_rmse_all = jnp.sqrt(mse_all / (target_mse_all + 1e-12))
    sigma_rmse_all = sigma * rmse_all

    # ----------------------------
    # Group masks by edge_type
    # ----------------------------
    et = mol_data.edge_type.astype(jnp.int32)

    # bond edges: exclude UNSPECIFIED=0 (padding sentinel)
    m_bond = m_all * (et > 0) * (et < 5)
    # hop2/hop3 (given num_bond_types = len(BONDS) = 5)
    m_hop2 = m_all * (et == 6)
    m_hop3 = m_all * (et == 7)
    # UNSPECIFIED edges (should be ~0 if edge_mask is correct)
    m_unspec = m_all * (et == 0)

    mse_bond, n_bond = masked_mean(per_edge, m_bond)
    rmse_bond = jnp.sqrt(mse_bond + 1e-12)
    target_mse_bond, _ = masked_mean(target ** 2, m_bond)
    rel_rmse_bond = jnp.sqrt(mse_bond / (target_mse_bond + 1e-12))

    mse_hop2, n_hop2 = masked_mean(per_edge, m_hop2)
    rmse_hop2 = jnp.sqrt(mse_hop2 + 1e-12)
    target_mse_hop2, _ = masked_mean(target ** 2, m_hop2)
    rel_rmse_hop2 = jnp.sqrt(mse_hop2 / (target_mse_hop2 + 1e-12))

    mse_hop3, n_hop3 = masked_mean(per_edge, m_hop3)
    rmse_hop3 = jnp.sqrt(mse_hop3 + 1e-12)
    target_mse_hop3, _ = masked_mean(target ** 2, m_hop3)
    rel_rmse_hop3 = jnp.sqrt(mse_hop3 / (target_mse_hop3 + 1e-12))

    # loss = 0.5 * (sigma ** 2) * mse_all
    loss_bond = 0.5 * sigma**2 * mse_bond
    loss_hop2 = 0.5 * sigma**2 * mse_hop2
    loss_hop3 = 0.5 * sigma**2 * mse_hop3

    loss = loss_bond + loss_hop2 + loss_hop3

    # sanity metric: how many UNSPECIFIED edges are accidentally "real". should be 0.
    n_unspec = jnp.sum(m_unspec)

    return loss, (
        sigma_idx,

        # overall
        rmse_all, rel_rmse_all, sigma_rmse_all,
        n_all,

        # bonds
        rmse_bond, rel_rmse_bond, n_bond,

        # hop2
        rmse_hop2, rel_rmse_hop2, n_hop2,

        # hop3
        rmse_hop3, rel_rmse_hop3, n_hop3,

        # debug
        n_unspec,
    )

@nnx.jit
def train_step(model, optimizer, mol_data, sigmas, sigma_idx, z_d):
    (
        loss, (
        sigma_idx,

        # overall
        rmse_all, rel_rmse_all, sigma_rmse_all,
        n_all,

        # bonds
        rmse_bond, rel_rmse_bond, n_bond,

        # hop2
        rmse_hop2, rel_rmse_hop2, n_hop2,

        # hop3
        rmse_hop3, rel_rmse_hop3, n_hop3,

        # debug
        n_unspec,
    )
    ), grads = nnx.value_and_grad(dist_score_matching_loss, has_aux=True)(
        model, mol_data, sigmas, sigma_idx, z_d
    )
    optimizer.update(model, grads)

    grad_norm = optax.global_norm(grads)
    return (
        loss, grad_norm, sigma_idx,

        # overall
        rmse_all, rel_rmse_all, sigma_rmse_all,
        n_all,

        # bonds
        rmse_bond, rel_rmse_bond, n_bond,

        # hop2
        rmse_hop2, rel_rmse_hop2, n_hop2,

        # hop3
        rmse_hop3, rel_rmse_hop3, n_hop3,

        # debug
        n_unspec,
    )


def eval_model(model, data: List[MoleculeData], sigmas, rngs, E_max):
    losses = []
    for mol in tqdm(data, desc="Evaluating model"): 
        z_d = jr.normal(rngs(), shape=(E_max,))
        sigma_idx = jr.randint(rngs(), shape=(), minval=0, maxval=len(sigmas))
        loss, _ = dist_score_matching_loss(model, mol, sigmas, sigma_idx, z_d)
        losses.append(loss)

    return jnp.mean(jnp.array(losses))
