from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import orbax.checkpoint as ocp

from jraphx import Data
import jax.numpy as jnp
from flax.struct import dataclass, field
from rdkit.Chem.rdchem import BondType
from typing import Optional
import numpy as np
from tqdm import tqdm


# we use 0 since atomic numbers are 1-indexed
ATOMIC_NUMBER_PAD_VAL = 0
# unspecified is used for padding
BONDS = {BondType.UNSPECIFIED: 0, BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}


@dataclass
class MoleculeData(Data):
    """Data class for RDKit molecules."""
    atom_type: Optional[jnp.ndarray] = None           # Atom type [num_atoms,]
    edge_type: Optional[jnp.ndarray] = None           # Edge type [num_edges,]
    totalenergy: Optional[jnp.ndarray] = None         # the absolute energy of this conformer, in Hartree. scalar
    boltzmannweight: Optional[jnp.ndarray] = None     # statistical weight of this conformer. scalar

    node_mask: Optional[jnp.ndarray] = None   # [N_max,] bool
    edge_mask: Optional[jnp.ndarray] = None   # [E_max,] bool


def _to_np(a):
    """Convert JAX/NumPy arrays to NumPy arrays; pass through python scalars."""
    if a is None:
        return None
    if isinstance(a, (np.ndarray,)):
        return a
    # JAX arrays / DeviceArrays
    try:
        return np.asarray(a)
    except Exception:
        return a

def _npz_payload_from_mol(m: "MoleculeData") -> dict:
    """
    Only include non-None fields. This keeps files compact and avoids object arrays.
    Assumes MoleculeData fields match those referenced below.
    """
    payload = {}

    # Base Data fields
    if m.x is not None:         payload["x"] = _to_np(m.x)
    if m.edge_index is not None:payload["edge_index"] = _to_np(m.edge_index)  # (2, E)
    if m.edge_attr is not None: payload["edge_attr"] = _to_np(m.edge_attr)
    if m.y is not None:         payload["y"] = _to_np(m.y)
    if m.pos is not None:       payload["pos"] = _to_np(m.pos)
    if m.batch is not None:     payload["batch"] = _to_np(m.batch)
    if m.ptr is not None:       payload["ptr"] = _to_np(m.ptr)

    # MoleculeData fields
    if m.atom_type is not None:       payload["atom_type"] = _to_np(m.atom_type)
    if m.edge_type is not None:       payload["edge_type"] = _to_np(m.edge_type)
    if m.totalenergy is not None:     payload["totalenergy"] = _to_np(m.totalenergy)
    if m.boltzmannweight is not None: payload["boltzmannweight"] = _to_np(m.boltzmannweight)

    return payload

def _get_arr(arrs: np.lib.npyio.NpzFile, key: str):
    """Return jnp.array(...) if key exists, else None."""
    if key not in arrs:
        return None
    return jnp.array(arrs[key])


def augment_hop_edges(
    mol,
    *,
    num_bond_types: int,
    order: int = 3,
) :
    def binarize(x):
        return (x > 0).astype(np.int32)
    
    N = int(mol.atom_type.shape[0])
    edge_index = mol.edge_index
    edge_type = mol.edge_type.astype(np.int32)

    # --- build dense adjacency and dense type matrix ---
    src = edge_index[0].astype(np.int32)
    dst = edge_index[1].astype(np.int32)

    # drop any invalid edges and self-loops
    valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N) & (src != dst)
    src_v = src[valid]
    dst_v = dst[valid]
    e_v = edge_type[valid]

    adj_mat = np.zeros((N, N), dtype=np.int32)
    adj_mat[src_v, dst_v] = 1
    type_mat = np.zeros((N, N), dtype=np.int32)
    np.maximum.at(type_mat, (src_v, dst_v), e_v)

    # Runtime sanity check
    if int(np.max(e_v, initial=0)) >= num_bond_types:
        raise ValueError(
            f"Found bond edge_type >= num_bond_types. "
            f"max(edge_type)={int(jnp.max(e_v))}, num_bond_types={num_bond_types}"
        )

    I = np.eye(N, dtype=np.int32)
    # adj_mats[0] is self
    # adj_mats[1] is <= 1 hop reachability
    adj_mats = [I, binarize(adj_mat + I)]

    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    
    # order_mat stores the min distance from i -> j, up to order
    order_mat = np.zeros((N, N), dtype=np.int32)
    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

    # Synthetic type ids for hop > 1:
    # hop=2 -> num_bond_types + 1
    # hop=3 -> num_bond_types + 2
    # hop=1 isn't included since it's already an actual bond type
    type_highorder = np.where(
        order_mat > 1,
        num_bond_types + order_mat - 1,
        0,
    ).astype(jnp.int32)

    # Ensure we never overwrite real bonds
    if np.any((type_mat > 0) & (type_highorder > 0)):
        raise ValueError("Overlap detected: higher-order edge collides with an existing bond edge.")
    
    type_new = type_mat + type_highorder  # (N, N) int32, 0 means no edge

    # --- dense -> sparse ---
    rows, cols = np.nonzero(type_new)  # 1D arrays of same length
    new_edge_index = np.stack([rows.astype(np.int32), cols.astype(np.int32)], axis=0)
    new_edge_type  = type_new[rows, cols].astype(np.int32)

    return mol.replace(
        edge_index=jnp.asarray(new_edge_index),
        edge_type=jnp.asarray(new_edge_type),
    )


def pad_molecule_data_np(mol: MoleculeData, N_max: int, E_max: int) -> MoleculeData:
    assert mol.pos is not None
    assert mol.edge_index is not None
    assert mol.atom_type is not None
    assert mol.edge_type is not None

    # Convert once to NumPy (handles jnp arrays too)
    atom_type = np.asarray(mol.atom_type)
    pos       = np.asarray(mol.pos)
    edge_index= np.asarray(mol.edge_index)
    edge_type = np.asarray(mol.edge_type)

    N = pos.shape[0]
    E = edge_index.shape[1]
    if N > N_max or E > E_max:
        raise ValueError(f"Got N={N}, E={E} but N_max={N_max}, E_max={E_max}")

    # Masks (NumPy)
    node_mask = np.zeros((N_max,), dtype=bool)
    node_mask[:N] = True
    edge_mask = np.zeros((E_max,), dtype=bool)
    edge_mask[:E] = True

    # Padded arrays
    atom_type_p = np.full((N_max,), ATOMIC_NUMBER_PAD_VAL, dtype=atom_type.dtype)
    atom_type_p[:N] = atom_type

    pos_p = np.zeros((N_max, 3), dtype=pos.dtype)
    pos_p[:N, :] = pos

    edge_index_p = np.zeros((2, E_max), dtype=edge_index.dtype)
    edge_index_p[:, :E] = edge_index

    edge_type_p = np.full((E_max,), BONDS[BondType.UNSPECIFIED], dtype=edge_type.dtype)
    edge_type_p[:E] = edge_type

    # Convert to jnp once
    return mol.replace(
        atom_type=jnp.asarray(atom_type_p),
        pos=jnp.asarray(pos_p),
        edge_index=jnp.asarray(edge_index_p),
        edge_type=jnp.asarray(edge_type_p),
        node_mask=jnp.asarray(node_mask),
        edge_mask=jnp.asarray(edge_mask),
    )

def pad_molecule_data(mol: MoleculeData, N_max: int, E_max: int) -> MoleculeData:
    # --- Infer sizes ---
    assert mol.pos is not None, "mol.pos required"
    assert mol.edge_index is not None, "mol.edge_index required"
    assert mol.atom_type is not None, "mol.atom_type required"
    assert mol.edge_type is not None, "mol.edge_type required"

    N = mol.pos.shape[0]
    E = mol.edge_index.shape[1]

    # --- Masks ---
    node_mask = jnp.arange(N_max) < N          # (N_max,) bool
    edge_mask = jnp.arange(E_max) < E          # (E_max,) bool

    # --- Pad node tensors ---
    # atom_type: (N,) -> (N_max,)
    atom_type = jnp.pad(mol.atom_type, (0, N_max - N), constant_values=ATOMIC_NUMBER_PAD_VAL)

    # pos: (N, 3) -> (N_max, 3)
    pos = jnp.pad(mol.pos, ((0, N_max - N), (0, 0)), constant_values=0.0)

    # --- Pad edge tensors ---
    # edge_index: (2, E) -> (2, E_max), pad with (0, 0) edges
    edge_index = jnp.pad(mol.edge_index, ((0, 0), (0, E_max - E)), constant_values=0)

    # edge_type: (E,) -> (E_max,)
    edge_type = jnp.pad(mol.edge_type, (0, E_max - E), constant_values=BONDS[BondType.UNSPECIFIED])


    return mol.replace(
        atom_type=atom_type,
        pos=pos,
        edge_index=edge_index,
        edge_type=edge_type,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

def save_molecules_split(
    split_dir: str | Path,
    samples: list[tuple["MoleculeData", str, str]],
    *,
    split_name: str,
    compress: bool = True,
) -> None:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    saver = np.savez_compressed if compress else np.savez

    records = []
    for i, (m, smiles, molblock) in enumerate(tqdm(samples, desc=f"Saving split={split_name}")):
        fname = f"mol_{i:06d}.npz"
        fpath = split_dir / fname

        payload = _npz_payload_from_mol(m)
        if not payload:
            raise ValueError(f"Refusing to save empty MoleculeData at index {i}.")
        saver(fpath, **payload)

        records.append({"file": fname, "smiles": smiles, "molblock": molblock})

    (split_dir / "index.json").write_text(json.dumps({
        "split": split_name,
        "num_samples": len(records),
        "records": records,
    }))



def load_molecules_split(
    split_dir: str | Path,
    *,
    num_mols: int | None = None,
) -> List[tuple["MoleculeData", str]]:
    """
    Load one split directory.
    - Reads index.json records for filenames (and smiles, optionally).
    - Returns a list of (MoleculeData, smiles) tuples.
    """
    split_dir = Path(split_dir)

    index_path = split_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.json: {index_path}")

    index_obj = json.loads(index_path.read_text())
    records = index_obj["records"]

    if num_mols is not None:
        records = records[:num_mols]

    out: List[tuple["MoleculeData", str]] = []
    for rec in tqdm(records, desc=f"Loading split={split_dir.name}"):
        fpath = split_dir / rec["file"]
        smiles = rec["smiles"]
        molblock = rec["molblock"]

        arrs = np.load(fpath, allow_pickle=False)

        m = MoleculeData(
            # Base Data
            x=_get_arr(arrs, "x"),
            edge_index=_get_arr(arrs, "edge_index"),
            edge_attr=_get_arr(arrs, "edge_attr"),
            y=_get_arr(arrs, "y"),
            pos=_get_arr(arrs, "pos"),
            batch=_get_arr(arrs, "batch"),
            ptr=_get_arr(arrs, "ptr"),

            # MoleculeData
            atom_type=_get_arr(arrs, "atom_type"),
            edge_type=_get_arr(arrs, "edge_type"),
            totalenergy=_get_arr(arrs, "totalenergy"),
            boltzmannweight=_get_arr(arrs, "boltzmannweight"),
        )
        out.append((m, smiles, molblock))

    return out

