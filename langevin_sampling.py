import jax.random as jr
from torchvision.transforms.functional import to_pil_image
import numpy as np
import jax.numpy as jnp
import jax
from model import compute_dist
from tqdm import tqdm


def convert_dist_score_to_pos_score(dist_scores, distances, cur_conformer, mol):
    """
    dist_scores: (E_max)
    distances: (E_max)
    cur_conformer: (V_max, 3)
    mol: MoleculeData object with padding info
    """
    edge_index = mol.edge_index
    src = edge_index[0] # j
    dst = edge_index[1] # i

    # (E_max, 3)
    r_i = cur_conformer[dst]
    r_j = cur_conformer[src]

    edge_mask = mol.edge_mask.astype(bool)
    eps = 1e-8
    dist_masked = jnp.where(edge_mask, distances, 1.0)
    dist_masked_inv = 1. / (dist_masked + eps)

    # (E_max, 3): per edge contribution
    # we multiply by 2 because our model was trained on dist^2
    contrib_e = (2.0 * dist_scores)[:, None] * (r_i - r_j)
    contrib_e = jnp.where(edge_mask[:, None], contrib_e, 0.0)

    Vmax = cur_conformer.shape[0]
    return jax.ops.segment_sum(contrib_e, dst, num_segments=Vmax)


def langevin_sampling(model, mol, sigmas, rngs, N_max):
    model.eval()

    sigma_max = sigmas[0]
    sigma_min = sigmas[-1]
    assert sigma_max == jnp.max(sigmas)
    assert sigma_min == jnp.min(sigmas)
    E_max = mol.edge_index.shape[1]

    T = 100
    eps = 2.4e-6

    # sample a random conformer
    cur_pos_v3 = jr.normal(rngs.sampling(), (N_max, 3)) * sigma_max

    for i in tqdm(range(sigmas.shape[0]), desc="Sigma levels"):
        print(f"sigma {i}: {sigmas[i]}")
        cur_sigma = sigmas[i]
        cur_alpha = eps * (cur_sigma / sigma_min) ** 2
        for t in tqdm(range(T), desc=f"Langevin steps (cur_sigma={cur_sigma:.3f})", leave=False):
            distances = compute_dist(cur_pos_v3, mol.edge_index)
            # we can safely pass in mol without modifying pos since the score model is based off of distances only
            dist_scores = model(mol, distances, sigmas[i])

            conformer_scores = convert_dist_score_to_pos_score(dist_scores, distances, cur_pos_v3, mol)
            z = jr.normal(rngs.sampling(), shape=cur_pos_v3.shape)
            cur_pos_v3 = cur_pos_v3 + cur_alpha * conformer_scores + jnp.sqrt(2 * cur_alpha) * z
            

    return cur_pos_v3


