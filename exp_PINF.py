""" Plot the retrieval results from the energy dynamics """
#%% Testing inference code
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
from dataclasses import dataclass

@dataclass
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
    figout_dir: str = "figs/PINF" # Where to save the output figures
    pick_imgs_randomly: bool = True # Whether to pick images randomly or from diverse_example_idxs
    plot_energies: bool = True # Whether to plot the energies
    Eto: int = 250 # Plot up to this many energy steps.
    show_ims_by: str = "col" # "row" or "col"

default_args = {
    "fig1": (
        "Make Figure 1",
        Args(),
    ),
    "fig2": (
        "Make Figure 2",
        Args(figout_dir="figs/PINF2", beta=60., Y=int(2e5), n_queries=20, n_memories=20, mask_after_pct=0.6, depth=1000, plot_energies=False, show_ims_by="row", pick_imgs_randomly=False),
    ),
}

if ju.is_interactive():
    args = default_args["fig1"][1] # Default for interactive
else:
    args = tyro.extras.overridable_config_cli(default_args)

assert args.show_ims_by in ["col", "row"]

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

    d = qs0.shape[-1]
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
    figout_dir = Path(args.figout_dir)
    figout_dir.mkdir(parents=True, exist_ok=True)

    ## Load randomly from seed
    M = get_data(DataOpts.tiny_imagenet)
    M = rearrange(M, "n ... -> n (...)")
    N, d = M.shape
    M = M / (M.max() * np.sqrt(d))

    mask_after = int(d * args.mask_after_pct)
    assert args.n_queries <= args.n_memories

    rng = jr.PRNGKey(args.seed)
    diverse_example_idxs = [
        # 68146, # (red sock, white background) # BAD
        73606, # (teddy bear, black background)
        83444, # (goose, dark ocean background)
        69932, # (red car, gray background)
        40404, # (open box, brown background)
        94931, # (green acorn, green background)
        92025, # (white chihuahua, small brown background)
        131, # (goldfish, green background)
        # 79596, # (white pillar, blue gradient background) # BAD
        19459, # (monarch butterfly, light brown background)
        20035, # (yellow butterfly, gray-blue background)
        # 57149, # (black tux w/ saxophone, white-gray background)
        8309, # (pink lobster, light blue  seabed)
        6085, # (blue jellyfish, black background)
        # 76695, # (black and white mesh) # BAD
        87197, # (lemon, pink background)
        62867, # (boxing sillhouette, orange background)
        5964, # (gray koala)
        # 40471, # (pink blue sock) # BAD
        52936, # (pink car)
        13076, # (Animal jumping on grassy background)
        # 4183, # (spider on purple background) # BAD
        75649, # Archway
        # 41631, # Gray keyboard
        27488, # (Black red robe)
        9546, # (Pair of penguins)
        82380, # (Pie on table)
    ]
    if args.pick_imgs_randomly:
        idxs = jr.choice(rng, jnp.arange(N), (args.n_memories,), replace=False)
    else:
        assert args.n_memories <= len(diverse_example_idxs)
        idxs = diverse_example_idxs[:args.n_memories]

    assert args.n_queries <= args.n_memories
    qs_og = jnp.array(M[idxs[:args.n_queries]])
    qs_og_show = qs_og * jnp.sqrt(d)
    qs0 = qs_og.at[:, mask_after:].set(0)
    qs0_show = qs0 * jnp.sqrt(d)
    memories = M[idxs]

    # Get both trajectories
    standard_trajectory, kernelized_trajectory = run_inference(qs0, memories, args.beta, args.Y, kernel="SinCosL2DAM", clamp=args.clamp, rerandomize_each_step=args.rerandomize_each_step, mask_after=mask_after, alpha=args.alpha, depth=args.depth)

    # Col wise
    if args.show_ims_by == "col":
        cols = ['query', 'kernelized', 'standard', 'original']
        img_arrays = np.array(rearrange([qs0_show, kernelized_trajectory['qsout'], standard_trajectory['qsout'], qs_og_show], "s n (h w c) -> s (n h) w c", h=64, w=64, c=3))

        for i, img in enumerate(img_arrays):
            im = Image.fromarray((img * 255).astype(np.uint8))
            if ju.is_interactive():
                im.show()
            im.save(figout_dir / f"img_{cols[i]}.png")

    elif args.show_ims_by == "row":
        img_arrays = np.array(rearrange([qs0_show, kernelized_trajectory['qsout'], standard_trajectory['qsout'], qs_og_show], "s n (h w c) -> s h (n w) c", h=64, w=64, c=3))
        big_img = rearrange(img_arrays[:-1], "s h w c -> (s h) w c")

        if ju.is_interactive():
            fig, ax = plt.subplots(1,1, figsize=(8, 20))
            ax.imshow(big_img)
            ax.axis('off')
        im = Image.fromarray((big_img * 255).astype(np.uint8))
        im.save(figout_dir / f"PINF2.png")
    else:
        raise ValueError(f"show_ims_by must be 'col' or 'row', got {args.show_ims_by}")

    print(f"Figures saved to {figout_dir}")

    #%% Save energies
    if args.plot_energies:
        dam_color = "#0AA551"
        kdam_color = "#0E86F6"
        colors = ["#DC330A", "#A7008F", "#7699CE", "#8F5C25"]
        fig, ax = plt.subplots(figsize=(4.5,4))
        standard_energies = np.stack(standard_trajectory['energies']).T
        kernelized_energies = np.stack(kernelized_trajectory['energies']).T
        Eto = args.Eto
        for i in range(args.n_queries):
            ax.plot(standard_energies[i][:Eto], c=dam_color, linestyle="--", label=f"Standard {i}", linewidth=2., alpha=0.4)
            ax.plot(kernelized_energies[i][:Eto], c=kdam_color, linestyle="-", label=f"Kernelized {i}", linewidth=2, alpha=0.8)

        standard_line = mlines.Line2D([0], [0], color=dam_color, linestyle="--", label='DAM Energies')
        kernelized_line = mlines.Line2D([0], [0], color=kdam_color, linestyle="-", label='kDAM Energies', linewidth=2.)
        ax.legend(handles=[kernelized_line, standard_line], fontsize=14)
        ax.set_xticks([])

        ax.tick_params(axis=u'x', which=u'both',length=0)

        fig.show()
        fig.savefig(figout_dir / "energies.png")