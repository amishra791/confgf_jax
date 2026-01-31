import os
# os.environ["JAX_LOG_COMPILES"] = "1"

import argparse
from pathlib import Path

from data import load_molecules_split, augment_hop_edges, pad_molecule_data_np
from model import DistanceScoreModel
from losses import train_step, eval_model

from flax import nnx
import jax.numpy as jnp
import jax.random as jr
import jax
import optax
from tqdm import tqdm
import numpy as np
import math
import optax
import orbax.checkpoint as ocp
from data import MoleculeData, BONDS

import wandb

N_MAX = 29
# since we are computing 2/3 hop neighbors, use an upper-bound of a complete graph
E_MAX = N_MAX * (N_MAX - 1)



parser = argparse.ArgumentParser(description="Program to train model")
parser.add_argument("load_dir", type=str, help="where to load preprocessed data from")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--microbatch_size", type=int, default=128)

args = parser.parse_args()

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="amishra791",
    # Set the wandb project where this run will be logged.
    project="conformation-diffusion",
    # Track hyperparameters and run metadata.
    config={
        "dataset": "QM9",
        "epochs": args.num_epochs,
    },
)

# setup checkpoint dir
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/qm9_diffusion_checkpoint_prod')
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

train_data = load_molecules_split(Path(args.load_dir) / 'train')
train_data = [
    pad_molecule_data_np(augment_hop_edges(m, num_bond_types=len(BONDS)), N_MAX, E_MAX)
    for (m, *_) in tqdm(train_data, desc="Padding molecules")
]
train_data = jax.device_put(train_data)
print(f"Loaded and padded {len(train_data)} training molecules")

val_data = load_molecules_split(Path(args.load_dir) / 'val')
val_data = [
    pad_molecule_data_np(augment_hop_edges(m, num_bond_types=len(BONDS)), N_MAX, E_MAX)
    for (m, *_) in tqdm(val_data, desc="Padding molecules")
]
val_data = jax.device_put(val_data)
print(f"Loaded and padded {len(val_data)} val molecules")

rngs = nnx.Rngs(0, params=1)
model = DistanceScoreModel(rngs=rngs)

sigmas = jnp.geomspace(0.5, 0.02, 40)
num_sigmas = len(sigmas)

sigma_loss_sum = np.zeros(num_sigmas, dtype=np.float64)
sigma_count = np.zeros(num_sigmas, dtype=np.int64)
sigma_rmse_sum = np.zeros((num_sigmas,), dtype=np.float64)
sigma_rel_rmse_sum = np.zeros((num_sigmas,), dtype=np.float64)
sigma_sigma_rmse_sum = np.zeros((num_sigmas,), dtype=np.float64)
# Per-sigma accumulators for edge-type groups (averaged over sampling events)
sigma_rmse_bond_sum = np.zeros(num_sigmas, dtype=np.float64)
sigma_rel_bond_sum  = np.zeros(num_sigmas, dtype=np.float64)
sigma_n_bond_sum    = np.zeros(num_sigmas, dtype=np.float64)

sigma_rmse_hop2_sum = np.zeros(num_sigmas, dtype=np.float64)
sigma_rel_hop2_sum  = np.zeros(num_sigmas, dtype=np.float64)
sigma_n_hop2_sum    = np.zeros(num_sigmas, dtype=np.float64)

sigma_rmse_hop3_sum = np.zeros(num_sigmas, dtype=np.float64)
sigma_rel_hop3_sum  = np.zeros(num_sigmas, dtype=np.float64)
sigma_n_hop3_sum    = np.zeros(num_sigmas, dtype=np.float64)

sigma_n_unspec_sum  = np.zeros(num_sigmas, dtype=np.float64)


# ----------------------------
# Training length / batching
# ----------------------------
num_epochs = args.num_epochs
microbatch_size = args.microbatch_size  # default 128
accum_k = microbatch_size  # "effective batch size" in micro-steps (one graph per micro-step)
microsteps_per_epoch = len(train_data)

num_iters = num_epochs * microsteps_per_epoch  # total microsteps

# optax schedule expects "step" = count of optimizer updates *as seen by the transform*.
# With MultiSteps, the base optimizer only applies once per accum_k, so the schedule
# is naturally advanced per update (not per microstep), which is what we want.
clip_value = 5.0

peak_lr = 5e-4
warmup_steps = int(0.04 * (num_iters // accum_k))
total_steps = num_iters // accum_k
end_lr = 1e-4

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=peak_lr,
    warmup_steps=warmup_steps,
    decay_steps=int(0.8 * total_steps),
    end_value=end_lr,
)

base_tx = optax.chain(
    optax.clip_by_global_norm(clip_value),
    optax.adam(learning_rate=lr_schedule),
)

tx = optax.MultiSteps(
    base_tx,
    every_k_schedule=accum_k,
    use_grad_mean=True,  # averages grads over the k micro-steps
)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)


best_val_loss = jnp.inf
model.train()
for i in range(num_iters):
    update_step = i // accum_k  # integer optimizer-update index

    # Sample molecule
    idx = jr.randint(rngs(), (), 0, len(train_data))
    cur_mol = train_data[idx]

    # Sample z_d
    z_d = jr.normal(rngs(), shape=(E_MAX,))

    # Sample sigma index using the new sampler
    sigma_idx = jr.randint(rngs(), (), 0, len(sigmas))

    (loss, grad_norm,
        sigma_idx,
        rmse, rel_rmse, sigma_rmse, n_all,
        rmse_bond, rel_rmse_bond, n_bond,
        rmse_hop2, rel_rmse_hop2, n_hop2,
        rmse_hop3, rel_rmse_hop3, n_hop3,
        n_unspec) = train_step(
            model, optimizer, cur_mol, sigmas, sigma_idx, z_d
        )

    loss_val = float(loss)
    rmse_val = float(rmse)
    rel_rmse_val = float(rel_rmse)
    sigma_rmse_val = float(sigma_rmse)
    idx_val = int(sigma_idx)

    sigma_loss_sum[idx_val] += loss_val
    sigma_rmse_sum[idx_val] += rmse_val
    sigma_rel_rmse_sum[idx_val] += rel_rmse_val
    sigma_sigma_rmse_sum[idx_val] += sigma_rmse_val
    sigma_count[idx_val] += 1

    # Group-wise accumulators (per-sample means)
    sigma_rmse_bond_sum[idx_val] += float(rmse_bond)
    sigma_rel_bond_sum[idx_val]  += float(rel_rmse_bond)
    sigma_n_bond_sum[idx_val]    += float(n_bond)

    sigma_rmse_hop2_sum[idx_val] += float(rmse_hop2)
    sigma_rel_hop2_sum[idx_val]  += float(rel_rmse_hop2)
    sigma_n_hop2_sum[idx_val]    += float(n_hop2)

    sigma_rmse_hop3_sum[idx_val] += float(rmse_hop3)
    sigma_rel_hop3_sum[idx_val]  += float(rel_rmse_hop3)
    sigma_n_hop3_sum[idx_val]    += float(n_hop3)

    sigma_n_unspec_sum[idx_val]  += float(n_unspec)

    if i % accum_k == (accum_k - 1):
        cur_lr = float(lr_schedule(update_step))
        # lam = sigma_mixture_lambda(update_step)
        # cur_lr = 1e-3
        lam = 0.0
        metrics_dict = {
            "sigma/mix_lambda": float(lam),
            "opt/lr": float(cur_lr),
        }
        run.log(metrics_dict)

    if i % (accum_k * 50) == (accum_k * 50 - 1):
        freq = sigma_count / max(int(sigma_count.sum()), 1)
        count = np.maximum(sigma_count, 1)
        mean_loss = sigma_loss_sum / count
        mean_rmse = sigma_rmse_sum / count
        mean_rel_rmse = sigma_rel_rmse_sum / count
        mean_sigma_rmse = sigma_sigma_rmse_sum / count
        mean_rmse_bond = sigma_rmse_bond_sum / count

        mean_rel_bond  = sigma_rel_bond_sum  / count
        mean_n_bond    = sigma_n_bond_sum    / count

        mean_rmse_hop2 = sigma_rmse_hop2_sum / count
        mean_rel_hop2  = sigma_rel_hop2_sum  / count
        mean_n_hop2    = sigma_n_hop2_sum    / count

        mean_rmse_hop3 = sigma_rmse_hop3_sum / count
        mean_rel_hop3  = sigma_rel_hop3_sum  / count
        mean_n_hop3    = sigma_n_hop3_sum    / count

        mean_n_unspec  = sigma_n_unspec_sum  / count
        print(f"Sigma Metrics at Microbatch {i:8d} (update {update_step})")
        print("High sigma losses:   ", mean_loss[:3])
        print("Mid sigma losses: ", mean_loss[19:21])
        print("Low sigma losses: ", mean_loss[-3:])
        print("High sigma relRMSE:   ", mean_rel_rmse[:3])
        print("Mid sigma relRMSE:  ", mean_rel_rmse[19:21])
        print("Low sigma relRMSE:  ", mean_rel_rmse[-3:])

        print("High sigma relRMSE bonds: ", mean_rel_bond[:3])
        print("Mid  sigma relRMSE bonds:", mean_rel_bond[19:21])
        print("Low  sigma relRMSE bonds:", mean_rel_bond[-3:])

        print("High sigma relRMSE hop2:  ", mean_rel_hop2[:3])
        print("Mid  sigma relRMSE hop2:", mean_rel_hop2[19:21])
        print("Low  sigma relRMSE hop2:", mean_rel_hop2[-3:])

        print("High sigma relRMSE hop3:  ", mean_rel_hop3[:3])
        print("Mid  sigma relRMSE hop3:", mean_rel_hop3[19:21])
        print("Low  sigma relRMSE hop3:", mean_rel_hop3[-3:])

        print("Mean UNSPEC edges counted as real (should be ~0):", mean_n_unspec[:3], mean_n_unspec[19:21], mean_n_unspec[-3:])


        sigma_loss_sum[:] = 0
        sigma_rmse_sum[:] = 0
        sigma_rel_rmse_sum[:] = 0
        sigma_sigma_rmse_sum[:] = 0
        sigma_count[:] = 0
        sigma_rmse_bond_sum[:] = 0
        sigma_rel_bond_sum[:]  = 0
        sigma_n_bond_sum[:]    = 0

        sigma_rmse_hop2_sum[:] = 0
        sigma_rel_hop2_sum[:]  = 0
        sigma_n_hop2_sum[:]    = 0

        sigma_rmse_hop3_sum[:] = 0
        sigma_rel_hop3_sum[:]  = 0
        sigma_n_hop3_sum[:]    = 0

        sigma_n_unspec_sum[:]  = 0

        # Validate
        model.eval()
        val_loss = eval_model(model, val_data, sigmas, rngs, E_MAX)
        model.train()
        print(f"Microbatch {i} | Validation Loss {val_loss:.4f}")
        metrics_dict = {'val/loss': float(val_loss)}
        for base_idx in [0, len(sigmas) // 2, len(sigmas) - 2]:
            for inc in range(2):
                j = base_idx + inc
                metrics_dict[f"sigma/value_idx{j:02d}"] = float(sigmas[j])
                metrics_dict[f"sigma/freq_idx{j:02d}"]  = float(freq[j])

                metrics_dict[f"overall/loss_idx{j:02d}"]      = float(mean_loss[j])
                metrics_dict[f"overall/rmse_idx{j:02d}"]      = float(mean_rmse[j])
                metrics_dict[f"overall/relrmse_idx{j:02d}"]   = float(mean_rel_rmse[j])
                metrics_dict[f"overall/sigmarmse_idx{j:02d}"] = float(mean_sigma_rmse[j])

                metrics_dict[f"bond/rmse_idx{j:02d}"]      = float(mean_rmse_bond[j])
                metrics_dict[f"bond/relrmse_idx{j:02d}"]   = float(mean_rel_bond[j])
                metrics_dict[f"bond/n_edges_idx{j:02d}"]   = float(mean_n_bond[j])

                metrics_dict[f"hop2/rmse_idx{j:02d}"]      = float(mean_rmse_hop2[j])
                metrics_dict[f"hop2/relrmse_idx{j:02d}"]   = float(mean_rel_hop2[j])
                metrics_dict[f"hop2/n_edges_idx{j:02d}"]   = float(mean_n_hop2[j])

                metrics_dict[f"hop3/rmse_idx{j:02d}"]      = float(mean_rmse_hop3[j])
                metrics_dict[f"hop3/relrmse_idx{j:02d}"]   = float(mean_rel_hop3[j])
                metrics_dict[f"hop3/n_edges_idx{j:02d}"]   = float(mean_n_hop3[j])

                metrics_dict[f"debug/n_unspec_idx{j:02d}"] = float(mean_n_unspec[j])
        run.log(metrics_dict)
        # if val_loss < best_val_loss:
        # print("Saving model.")
        # best_val_loss = val_loss
        # _, state = nnx.split(model)
        # ckptr.save(ckpt_dir / 'state', force=True, args=ocp.args.StandardSave(state))
        # ckptr.wait_until_finished()

    # Save latest model
    if i % (accum_k * 5000) == (accum_k * 5000 - 1):
        print("Saving model.")
        _, state = nnx.split(model)
        ckptr.save(ckpt_dir / 'state', force=True, args=ocp.args.StandardSave(state))
        ckptr.wait_until_finished()

