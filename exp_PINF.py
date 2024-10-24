""" Plot the retrieval results from the energy dynamics """
#%% Testing inference code
import numpy as np
import jax.numpy as jnp
import jax
import jax.random as jr
from einops import rearrange
from data_utils import get_data, DataOpts
from typing import *
from tqdm import trange
import equinox as eqx
from kernel_sims import SIM_REGISTRY as KernelOpts
from PIL import Image
import jax_utils as ju
import tyro
from pathlib import Path

class Args:
    beta: float = 60. # Inverse temperature
    Y: int = 180_000 # Number of random features
    kernel: str = "SinCosL2DAM" # Which kernel function to use
    clamp: bool = True # Whether to clamp open pixels or not
    rerandomize_each_step: bool = False # Whether to rerandomize the weights each step
    device: int = 0 # Which GPU to use
    mask_after_pct: float = 0.33 # Fraction of pixels after which to mask in the query
    alpha: float = 0.1 # Energy descent step size
    depth: int = 300 # Number of gradient descent steps
    n_queries: int = 4 # Number of queries
    n_memories: int = 4 # Number of memories
    seed: int = 42 # Random seed for selecting memories and random features
    figout_dir: str = "figs/FIG1" # Where to save the output figures

if ju.is_interactive():
    args = tyro.cli(Args)
else:
    args = Args()

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

def run_inference(
    qs0, 
    memories, 
    beta, 
    Y, 
    weight_seed:int=42, 
    kernel:str="SinCosL2DAM", 
    clamp=False,
    alpha=0.1,
    depth=1000,
    mask_after=None,
    rerandomize_each_step=False,
    ):

    only_rerandomize_phi_each_step=False # Do not rerandomize only phi each step, this is not gradient descent if you do

    mask_after = int(d // 2) if mask_after is None else mask_after
    clamp_until = mask_after
    rng = jr.PRNGKey(weight_seed)
    key, rng = jr.split(rng)
    if beta == "opt":
        kdam = KernelOpts[kernel](key, d, Y, beta=1.).condition_beta_on_memories(memories)
    else:
        kdam = KernelOpts[kernel](key, d, Y, beta=beta)
    energyf = jax.jit(jax.vmap(jax.value_and_grad(kdam.energy), in_axes=(0, None)))

    # Descend normal energy.
    energies = []
    qs = qs0
    for i in trange(depth):
        Eog, dEogdx = energyf(qs, memories)
        if clamp:
            qs = qs - alpha * dEogdx.at[:, :clamp_until].set(0)
        else:
            qs = qs - alpha * dEogdx
        energies.append(Eog)
    
    qsout_show = jnp.clip(qs, 0, 1/jnp.sqrt(d)) * jnp.sqrt(d)

    standard_trajectory = {
        "qsout": qsout_show,
        "energies": energies
    }

    T = kdam.kernelize_memories(memories) # Varies depending on beta
    @eqx.filter_jit
    def kernel_energyf(kdam, qs, T):
        return jax.vmap(jax.value_and_grad(kdam.kernel_energy), in_axes=(0, None))(qs, T)

    T = kdam.kernelize_memories(memories)
    energies = []
    qs = qs0

    for i in trange(depth):
        if rerandomize_each_step:
            k1, k2, rng = jr.split(rng, 3)
            kdam = eqx.tree_at(lambda model: model.S, kdam, jr.normal(k1, kdam.S.shape))
            kdam = eqx.tree_at(lambda model: model.b, kdam, jr.normal(k2, kdam.b.shape))
            if not only_rerandomize_phi_each_step:
                T = kdam.kernelize_memories(memories)
        Ekernel, dEkdx = kernel_energyf(kdam, qs, T)
        if clamp:
            qs = qs - alpha * dEkdx.at[:, :clamp_until].set(0)
        else:
            qs = qs - alpha * dEkdx
        energies.append(Ekernel)

    qs0_show = qs0 * jnp.sqrt(d)
    qsout_show = jnp.clip(qs, 0, 1/jnp.sqrt(d)) * jnp.sqrt(d)
    
    kernelized_trajectory = {
        "qsout": qsout_show,
        "energies": energies
    }

    return standard_trajectory, kernelized_trajectory

if __name__ == "__main__":
    beta = args.beta
    Y = args.Y

    figout_dir = Path(args.figout_dir)
    figout_dir.mkdir(parents=True, exist_ok=True)

    ## Load randomly from seed
    M = get_data(DataOpts.tiny_imagenet)
    M = rearrange(M, "n ... -> n (...)")
    N, d = M.shape
    M = M / (M.max() * np.sqrt(d))

    n_queries = 2**2
    n_memories = 2**2
    mask_after = int(d * args.mask_after_pct)
    assert n_queries <= n_memories

    rng = jr.PRNGKey(args.seed)
    idxs = jr.choice(rng, jnp.arange(N,), (n_memories,), replace=False)
    qs_og = jnp.array(M[idxs[:n_queries]])
    qs_og_show = qs_og * jnp.sqrt(d)
    qs0 = qs_og.at[:, mask_after:].set(0)
    qs0_show = qs0 * jnp.sqrt(d)
    memories = M[idxs]

    # Get both trajectories
    standard_trajectory, kernelized_trajectory = run_inference(qs0, memories, beta, Y, kernel="SinCosL2DAM", clamp=args.clamp, rerandomize_each_step=args.rerandomize_each_step, mask_after=mask_after, alpha=args.alpha, depth=args.depth)

    # Col wise
    cols = ['query', 'kernelized', 'standard', 'original']
    img_arrays = np.array(rearrange([qs0_show, kernelized_trajectory['qsout'], standard_trajectory['qsout'], qs_og_show], "s n (h w c) -> s (n h) w c", h=64, w=64, c=3))

    for i, img in enumerate(img_arrays):
        im = Image.fromarray((img * 255).astype(np.uint8))
        if ju.is_interactive():
            im.show()
        im.save(figout_dir / f"img_{cols[i]}.png")