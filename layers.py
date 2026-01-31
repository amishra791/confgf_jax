from flax import nnx
import jax
import jax.numpy as jnp
from jraphx.nn.conv.message_passing import MessagePassing

class MLP(nnx.Module):
    def __init__(self, in_features, hidden_feature_list, rngs: nnx.Rngs, *, activate_final: bool = False):
        dims = [in_features] + list(hidden_feature_list)
        self.layers = nnx.List([
            nnx.Linear(in_features=dims[i], out_features=dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ])
        self.activate_final = activate_final

    def __call__(self, x):
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != last or self.activate_final:
                x = nnx.silu(x)
        return x

class GINEConv(MessagePassing):

    def __init__(self, nn: nnx.Module):
        super().__init__(aggr='mean', flow='source_to_target')
        self.nn = nn

    def message(
        self,
        x_j: jnp.ndarray,
        x_i: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # by construction of score model, x and edge_attr share the same feature dimension
        return nnx.silu(x_j + edge_attr)
        

    def update(
        self,
        aggr_out: jnp.ndarray,
        x: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # by construction of score model, x and aggr_out share the same feature dimension
        out = self.nn(x + aggr_out)
        return out

class GINBlock(nnx.Module):
    def __init__(self, hidden_dim, rngs: nnx.Rngs):
        self.ln = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.mlp = MLP(hidden_dim, [hidden_dim, hidden_dim], rngs)
        self.conv = GINEConv(self.mlp)

    def __call__(self, x, edge_index, edge_attr):
        h = self.ln(x)
        h = self.conv(x, edge_index, edge_attr)
        return x + h

class GIN(nnx.Module):

    def __init__(self, num_layers, hidden_dim, rngs: nnx.Rngs):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nnx.List([GINBlock(hidden_dim, rngs) for _ in range(num_layers)])


    def __call__(self, edge_index, node_attr, edge_attr, edge_mask):
        """
        edge_index: (2, E_max)
        node_attr:  (N_max, hidden_dim)
        edge_attr:  (E_max, hidden_dim)
        edge_mask:  (E_max,) bool, True for real edges
        """
        # Add PAD node with zero features at index = N_max
        pad_node_feat = jnp.zeros((1, node_attr.shape[-1]), dtype=node_attr.dtype)
        node_attr = jnp.concatenate([node_attr, pad_node_feat], axis=0)
        pad_idx = node_attr.shape[0] - 1  # == N_max

        # Remap padded edges to (pad_idx, pad_idx)
        # edge_mask True = real edge; False = padded edge
        edge_index = jnp.where(edge_mask[None, :], edge_index, pad_idx)

        # Optional redundancy: ensure padded edge_attr is zero
        edge_attr = edge_attr * edge_mask[:, None]

        out = node_attr
        for _, block in enumerate(self.layers):
            out = block(out, edge_index, edge_attr)
        return out
