from flax import nnx
import jax
import jax.numpy as jnp

from data import MoleculeData, BONDS
from layers import MLP, GIN

# number of elements in the periodic table is 118, but atomic numbers are 1-indexed.
# furthermore, we only use a subset of these elements, a bit overkill for our use case
NUM_ELEMENTS = 119
NUM_HOPS = 3


def compute_dist(pos: jax.Array, edge_index: jax.Array) -> jax.Array:
    """
    Given a matrix of positions and an edge index, computes the distance vector. 

    pos: |V| x 3 matrix of xyz coordinates
    edge_index: 2 x |E| matrix where rows is edge_index[0] and cols is at index 1
    """
    src, dst = edge_index[0], edge_index[1]
    delta = pos[src] - pos[dst]          # (E,3)
    return jnp.linalg.norm(delta, axis=-1)  # (E,)


class DistanceScoreModel(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        self.hidden_dim = 256

        self.node_embedding = nnx.Embed(num_embeddings=NUM_ELEMENTS, features=self.hidden_dim, rngs=rngs)
        self.edge_embedding = nnx.Embed(num_embeddings=len(BONDS) + NUM_HOPS, features=self.hidden_dim, rngs=rngs)
        self.dist_mlp = MLP(in_features=1, hidden_feature_list=[self.hidden_dim, self.hidden_dim], rngs=rngs)
        self.sigma_mlp = MLP(in_features=1, hidden_feature_list=[self.hidden_dim, self.hidden_dim], rngs=rngs)
        

        self.gin = GIN(num_layers=10, hidden_dim=self.hidden_dim, rngs=rngs)

        self.output_mlp = MLP(
            in_features=3 * self.hidden_dim, 
            hidden_feature_list=[self.hidden_dim, self.hidden_dim // 2, 1],
            rngs=rngs
        )

    def __call__(self, mol_data: MoleculeData, distances: jax.Array | None, sigma):
        
        node_embed = self.node_embedding(mol_data.atom_type)
        edge_embed = self.edge_embedding(mol_data.edge_type)

        dist_mat = distances if distances is not None else compute_dist(mol_data.pos, mol_data.edge_index) 
        edge_mask = mol_data.edge_mask.astype(dist_mat.dtype)  # (E,)
        edge_mask_e1 = edge_mask[:, None]                      # (E,1)
        dist_mat = dist_mat * edge_mask
        dist_embeds = self.dist_mlp(dist_mat[:, None]) * edge_mask_e1
        edge_embed = edge_embed * dist_embeds

        node_feats = self.gin(mol_data.edge_index, node_embed, edge_embed, mol_data.edge_mask)

        h_row, h_col = node_feats[mol_data.edge_index[0]], node_feats[mol_data.edge_index[1]]
        h_node_combined = h_row * h_col * edge_mask_e1

        sigma_in = jnp.log(jnp.maximum(sigma, 1e-12)).reshape(1, 1)
        sigma_embed = self.sigma_mlp(sigma_in)[0]                       # (H,)
        sigma_edge = jnp.broadcast_to(sigma_embed[None, :], (dist_embeds.shape[0], self.hidden_dim))  # (E, H)

        dist_features = jnp.concat([h_node_combined, dist_embeds, sigma_edge], axis=-1)

        raw_scores = self.output_mlp(dist_features) * edge_mask_e1

        out = jnp.squeeze(raw_scores / sigma)

        return out
