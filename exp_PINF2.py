""" Plot the retrieval results from the energy dynamics """
#%% Testing inference code
import numpy as np
import jax.numpy as jnp
import jax
import jax.random as jr
import matplotlib.pyplot as plt
from einops import rearrange
from data_utils import get_letter_data, get_eyestate_data, get_phoneme_data, get_mnist_traindata, get_cifar_traindata, get_tiny_imagenet_traindata
from enum import Enum
from plotting import show_img
from typing import *
from tqdm import trange
import equinox as eqx
from kernel_sims import SIM_REGISTRY as KernelOpts
import optax
from pathlib import Path
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import path_fixes as pf

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class DataOpts(Enum):
    letter = "letter"
    phoneme = "phoneme"
    eyestate = "eyestate"
    mnist = "mnist"
    cifar10 = "cifar10"
    tiny_imagenet = "tiny_imagenet"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

def get_data(datatype: DataOpts = DataOpts.mnist):
    if datatype == DataOpts.letter:
        return get_letter_data()[0]
    if datatype == DataOpts.phoneme:
        return get_phoneme_data()[0]
    if datatype == DataOpts.eyestate:
        return get_eyestate_data()[0]
    if datatype == DataOpts.mnist:
        return get_mnist_traindata()[0]
    if datatype == DataOpts.cifar10:
        return get_cifar_traindata()[0]
    if datatype == DataOpts.tiny_imagenet:
        return get_tiny_imagenet_traindata()[0]
    raise ValueError(f"Unknown datatype: {datatype}")

def show_before_after(xs0, xs_end, fig=None, show_colorbar=False, xs_og=None):
    n = xs0.shape[0]
    nh = nw = int(np.sqrt(n))
    if xs0.shape[-1] == 784:
        h, w, c = 28, 28, 1
    elif xs0.shape[-1] == 3072:
        h, w, c = 32, 32, 3
    elif xs0.shape[-1] == 12288:
        h, w, c = 64, 64, 3
    else:
        raise ValueError(f"Unknown image shape: {xs0.shape}")
        
    if xs_og is None:
        xs = rearrange([xs0, xs_end], "s n (h w c) -> n h (s w) c", h=h, w=h, c=c)
    else:
        xs = rearrange([xs0, xs_end, xs_og], "s n (h w c) -> n h (s w) c", h=h, w=h, c=c)
    xs = np.pad(xs, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant", constant_values=-xs.max())
    xs = rearrange(xs[:nh*nw], "(nh nw) h w c -> (nh h) (nw w) c", nh=nh, nw=nw)

    fig, ax = show_img(xs, fig=fig, show_colorbar=show_colorbar)
    return fig, ax

def show_trajectory_reconstructions(energies, xs0, xs_end, show_colorbar=False, xs_og=None):
    fig = plt.figure(layout="constrained", figsize=(20,12))
    subfigs = fig.subfigures(2, 1, hspace=0.07)
    show_before_after(xs0, xs_end, fig=subfigs[0], show_colorbar=show_colorbar, xs_og=xs_og);
    ax_energy = subfigs[1].subplots(1, 1)
    ax_energy.plot(energies)
    return fig, subfigs, ax_energy

def show_mnist_ims(xs):
    """Turn batch of mnist vectors into a grid of images."""
    assert len(xs.shape) > 1, "Needs batch dim and "
    n = xs.shape[0]
    xs = rearrange(xs, "... (h w c) -> ... h w c", h=28, w=28, c=1)

    kh = kw = int(np.sqrt(n))
    xshow = rearrange(xs[:kh*kw], "(kh kw) h w c -> (kh h) (kw w) c", kh=kh, kw=kw)
    fig, ax = show_img(xshow)
    return fig

def run_inference(
    qs0, 
    memories, 
    beta, 
    m, 
    weight_seed:int=42, 
    kernel:str="SinCosL2DAM", 
    clamp=False,
    alpha=0.1,
    depth=1000,
    mask_after=None,
    rerandomize_each_step=False,
    only_rerandomize_phi_each_step=False, # This does NOT work... If `rerandomize_each_step` is True, do NOT recompute `T`, only rerandomize phi
    show_colorbar=False,
    qs_og=None,
    ):
    n_queries, d = qs0.shape
    n_memories, _ = memories.shape

    mask_after = d // 2 if mask_after is None else mask_after
    clamp_until = mask_after
    rng = jr.PRNGKey(weight_seed)
    key, rng = jr.split(rng)
    if beta == "opt":
        kdam = KernelOpts[kernel](key, d, m, beta=1.).condition_beta_on_memories(memories)
    else:
        kdam = KernelOpts[kernel](key, d, m, beta=beta)
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

#%%
M = get_data(DataOpts.tiny_imagenet)
#%% Choose images from TIMNET
# h = w = 12
# N = h * w
# # M = rearrange(M, "(kh kw) h w c -> (kh h) (kw w) c")
# idxs = jr.permutation(jr.PRNGKey(42), jnp.arange(M.shape[0]), axis=0, independent=True)

# k = 0
# while True:
#     with plt.ion():

#         fig, axs = plt.subplots(h, w, figsize=(12,12))

#         for i in range(h):
#             for j in range(w):
#                 idx = idxs[k + i*h + j]
#                 ax = axs[i,j]
#                 ax.imshow(M[idx])
#                 ax.axis("off")
#                 ax.set_title(f"Idx: {idx}")

#         fig.tight_layout()
#         fig.show()

#         # Check if user says yes
#         response = input("Next page? [Y/n]: ")
#         if response.lower() == "n":
#             break
#         k += h * w

#         plt.close(fig)


#%%

if __name__ == "__main__":
    beta = 57.
    m = 173_000

    figout_dir = pf.FIGS / "FIG1"
    figout_dir.mkdir(parents=True, exist_ok=True)

    ## Load randomly from seed
    seed = 42 
    M = get_data(DataOpts.tiny_imagenet)
    M = rearrange(M, "n ... -> n (...)")
    N, d = M.shape
    M = M / (M.max() * np.sqrt(d))

    idxs = [
        68146, # (red sock, white background)
        73606, # (teddy bear, black background)
        83444, # (goose, dark ocean background)
        69932, # (red car, gray background)
        40404, # (open box, brown background)
        94931, # (green acorn, green background)
        92025, # (white chihuahua, small brown background)
        131, # (goldfish, green background)
        79596, # (white pillar, blue gradient background)
        19459, # (monarch butterfly, light brown background)
        20035, # (yellow butterfly, gray-blue background)
        57149, # (black tux w/ saxophone, white-gray background)
        8309, # (pink lobster, light blue  seabed)
        6085, # (blue jellyfish, black background)
        76695, # (black and white mesh)
        87197, # (lemon, pink background)
        62867, # (boxing sillhouette, orange background)
        5964, # (gray koala)
    ]
    n_queries = 4
    # n_memories = 2**2
    mask_after = d // 2
    # assert n_queries <= n_memories


    qs_og = jnp.array(M[idxs[:n_queries]])
    qs_og_show = qs_og * jnp.sqrt(d)
    qs0 = qs_og.at[:, mask_after:].set(0)
    qs0_show = qs0 * jnp.sqrt(d)
    memories = M[idxs]

    # Estandard
    standard_trajectory, kernelized_trajectory = run_inference(qs0, memories, beta, m, kernel="SinCosL2DAM", clamp=True, rerandomize_each_step=False, only_rerandomize_phi_each_step=False, mask_after=mask_after, alpha=0.1, depth=300, qs_og=qs_og_show)

    show_ims_by = "col" # or "row"
    # Row wise
    # img_arrays = np.array(rearrange([qs0_show, kernelized_trajectory['qsout'], standard_trajectory['qsout']], "s n (h w c) -> n h (s w) c", h=64, w=64, c=3))

    # Col wise
    cols = ['query', 'kernelized', 'standard', 'original']
    img_arrays = np.array(rearrange([qs0_show, kernelized_trajectory['qsout'], standard_trajectory['qsout'], qs_og_show], "s n (h w c) -> s (n h) w c", h=64, w=64, c=3))

    for i, img in enumerate(img_arrays):
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.show()
        im.save(figout_dir / f"{show_ims_by}_img_{cols[i]}.png")
        # Save each of the trajectories

    #%% Save Original images
    # og_img_outdir = figout_dir / "og_imgs"
    # og_img_outdir.mkdir(parents=True, exist_ok=True)

    # og_imgs = [np.array(rearrange(qs_og_show, "n (h w c) -> n h w c", h=64, w=64, c=3))[-1]]

    # og_img_names = ["sunflower-mural"]

    # for name, img in zip(og_img_names, og_imgs):
    #     im = Image.fromarray((img * 255).astype(np.uint8))
    #     im.show()
    #     im.save(og_img_outdir / f"{name}.png")

    
    # colors = ["red", "blue", "green", "purple"]
    dam_color = "#0AA551"
    kdam_color = "#0E86F6"
    colors = ["#DC330A", "#A7008F", "#7699CE", "#8F5C25"]
    fig, ax = plt.subplots(figsize=(4.5,4))
    standard_energies = np.stack(standard_trajectory['energies']).T
    kernelized_energies = np.stack(kernelized_trajectory['energies']).T
    Eto = 250
    for i in range(n_queries):
        # ax.plot(standard_energies[i][:Eto], c=colors[i], label=f"Standard {i}", linewidth=2.)
        # ax.plot(kernelized_energies[i][:Eto], c=colors[i], linestyle=":", label=f"Kernelized {i}", linewidth=3.)
        ax.plot(standard_energies[i][:Eto], c=dam_color, linestyle="--", label=f"Standard {i}", linewidth=2., alpha=0.4)
        ax.plot(kernelized_energies[i][:Eto], c=kdam_color, linestyle="-", label=f"Kernelized {i}", linewidth=2, alpha=0.8)

    # red_patch = mlines.AxLine(color='red', label='The red data')
    # blue_patch = mlines.AxLine(color='blue', label='The blue data')
    # mlines.

    standard_line = mlines.Line2D([0], [0], color=dam_color, linestyle="--", label='DAM Energies')
    kernelized_line = mlines.Line2D([0], [0], color=kdam_color, linestyle="-", label='kDAM Energies', linewidth=2.)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Energy")
    ax.legend(handles=[kernelized_line, standard_line], fontsize=14)
    # ax.legend()
    # ax.xaxis.axis('off')
    # ax.axis('off')
    ax.set_xticks([])

    ax.tick_params(axis=u'x', which=u'both',length=0)

    fig.show()
    fig.savefig(figout_dir / "energies.png")

# %%
