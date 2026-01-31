import os
import json
import random
import pickle

import argparse
from tqdm import tqdm

import numpy as np
import jax.numpy as jnp
from jraphx import Data
from jraphx.utils import scatter
from flax import nnx
import orbax.checkpoint as ocp

from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType

from data import MoleculeData, BONDS, save_molecules_split


# Scripts adapted from https://github.com/DeepGraphLearning/ConfGF/blob/main/confgf/dataset/dataset.py
# For a primer on how to interpret GEOM data, see this example notebook: https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb

def rdmol_to_data(mol:Mol, global_features, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = jnp.array(mol.GetConformer(0).GetPositions(), dtype=jnp.float32)


    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = jnp.array(atomic_number, dtype=jnp.int32)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BONDS[bond.GetBondType()]]

    edge_index = jnp.array([row, col], dtype=jnp.int32)
    edge_type = jnp.array(edge_type)

    # sort edges first by row, then by column
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    # calculate number of hydrogens connected to said atom. currently unused
    row, col = edge_index
    hs = (z == 1).astype(jnp.float32)
    num_hs = scatter(hs[row], col, dim_size=N, reduce='add').tolist()

    data = MoleculeData(
        atom_type=z, 
        pos=pos, 
        edge_index=edge_index, 
        edge_type=edge_type, 
        totalenergy=jnp.array(global_features['totalenergy']).astype(jnp.float32),
        boltzmannweight=jnp.array(global_features['boltzmannweight']).astype(jnp.float32))

    return data

def split_list(items, train_frac=0.8, seed=0):
    """
    Splits items into train/val/test where val=test=(1-train)/2.
    Returns: train_items, val_items, test_items
    """
    assert 0.0 < train_frac < 1.0
    rest = 1.0 - train_frac
    val_frac = rest / 2.0
    test_frac = rest / 2.0

    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    # ensure exact partition sizes sum to n
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    assert len(test_items) == n_test

    return train_items, val_items, test_items



def preprocess_GEOM_dataset(base_path, pickle_paths, conf_per_mol=5):
    samples = []  # list of (MoleculeData, smiles)
    bad_case = 0

    for rel_path in tqdm(pickle_paths):
        with open(os.path.join(base_path, rel_path), "rb") as fin:
            mol = pickle.load(fin)

        uniqueconfs = mol.get("uniqueconfs")
        conformers = mol.get("conformers")
        if uniqueconfs is None or conformers is None:
            bad_case += 1
            continue
        if uniqueconfs > len(conformers) or uniqueconfs <= 0:
            bad_case += 1
            continue
        if uniqueconfs < conf_per_mol:
            bad_case += 1
            continue

        smiles = mol.get("smiles")

        if uniqueconfs == conf_per_mol:
            conf_ids = np.arange(uniqueconfs)
        else:
            all_weights = np.array([c.get("boltzmannweight", -1.0) for c in conformers])
            conf_ids = (-all_weights).argsort()[:conf_per_mol]

        for conf_id in conf_ids:
            conf_meta = conformers[conf_id]
            global_features = {
                "totalenergy": conf_meta["totalenergy"],
                "boltzmannweight": conf_meta["boltzmannweight"],
            }
            rdmol = conf_meta["rd_mol"]
            data = rdmol_to_data(rdmol, global_features)
            molblock = Chem.MolToMolBlock(rdmol, confId=0)
            samples.append((data, smiles, molblock))

    print(f"bad case: {bad_case}")
    print(f"done! produced {len(samples)} conformer-samples")
    return samples




if __name__ == "__main__":
    # default base_dir: /home/aditya_mishra791/rdkit_folder
    # default save_dir: /home/aditya_mishra791/conformation_diffusion/preprocessed_data
    parser = argparse.ArgumentParser(description="Program to preprocess GEOM-QM9")
    parser.add_argument("base_dir", type=str, help="path to rdkit_folder")
    parser.add_argument("save_dir", type=str, help="where to save preprocessed data")

    parser.add_argument("--conf_per_mol", type=int, default=5)
    parser.add_argument("--tot_mol_size", type=int, default=50000)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    qm9_summary_file = os.path.join(args.base_dir, "summary_qm9.json")
    with open(qm9_summary_file, "r") as f:
        qm9_summ = json.load(f)

    # filter valid pickle paths (molecule-level list)
    pickle_path_list = []
    for smiles, meta_mol in tqdm(qm9_summ.items()):
        u_conf = meta_mol.get("uniqueconfs")
        pickle_path = meta_mol.get("pickle_path")
        if u_conf is None or pickle_path is None:
            continue
        if u_conf < args.conf_per_mol:
            continue
        pickle_path_list.append(pickle_path)

    # deterministic shuffle + cap to tot_mol_size
    rng = random.Random(args.seed)
    rng.shuffle(pickle_path_list)

    assert len(pickle_path_list) >= args.tot_mol_size, (
        f"available mols {len(pickle_path_list)} < tot_mol_size {args.tot_mol_size}"
    )
    pickle_path_list = pickle_path_list[:args.tot_mol_size]

    train_paths, val_paths, test_paths = split_list(
        pickle_path_list, train_frac=args.train_frac, seed=args.seed
    )

    # create split directories
    train_dir = os.path.join(args.save_dir, "train")
    val_dir = os.path.join(args.save_dir, "val")
    test_dir = os.path.join(args.save_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # preprocess per split
    train_data = preprocess_GEOM_dataset(args.base_dir, train_paths, conf_per_mol=args.conf_per_mol)
    val_data   = preprocess_GEOM_dataset(args.base_dir, val_paths,   conf_per_mol=args.conf_per_mol)
    test_data  = preprocess_GEOM_dataset(args.base_dir, test_paths,  conf_per_mol=args.conf_per_mol)

    # save per split
    save_molecules_split(train_dir, train_data, split_name='train')
    save_molecules_split(val_dir, val_data, split_name='val')
    save_molecules_split(test_dir, test_data, split_name='test')
