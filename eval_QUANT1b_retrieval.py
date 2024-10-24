#%%
import pandas as pd
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Array, PyTree, jaxtyped
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tyro
from typing import *
from dataclasses import dataclass
import path_fixes as pf
from enum import Enum
import matplotlib as mpl
from matplotlib.colors import Normalize


@dataclass
class Args:
    ckpt_dir: str # Where to load the checkpoints
    outfname: str  # Where to save output figure
    device: str = "0" # Which device to run on. One of {"cpu", "0", "1", ...}, where numbers represent the GPU

    def get_device(self): return "" if self.device.lower() in ["", "cpu"] else self.device

args = Args(ckpt_dir="results/QUANT1b_near_0.1_retrieval/ckpts", outfname="QUANT1b_retrieval.png")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]=args.get_device()

resultsdir = Path(args.ckpt_dir)
fnames = list(resultsdir.glob("*.pkl"))

df = pd.concat([pd.read_pickle(f) for f in fnames])
df = df.drop_duplicates(subset=["dataset", "d", "beta", "m", "kernel"], inplace=False)

dims = sorted(df["d"].unique())
ms = [5e3, 1e4, 4e4, 8e4, 1.2e5, 1.6e5, 2e5, 3e5, 4e5, 5e5]
ms = [int(m) for m in ms]
mticks = [m for m in ms if m not in [500, 10_000]]
mshow = [f"{m/1000:0.0f}k" for m in mticks]

betas = df["beta"].unique()

# Choose betas we want to see
betas = [10., 30., 40., 50.]
new_betas = list(sorted([b for b in betas if not (isinstance(b, str) or b == 1)], reverse=True))
betas = new_betas

norm = Normalize(vmin=0., vmax=len(betas)-1+0.25)
cmap = mpl.colormaps['plasma']

cols = ["at", "near", "far"]
fig, axs = plt.subplots(len(dims), len(cols), figsize=(len(cols)*3, len(dims)*3), squeeze=False)

axcol = 0
nQ = df.iloc[0].q0.shape[0]
balpha = 1.0

df["q0diff_hamming_mean"] = df["q0diff_hamming"].apply(lambda x: jnp.mean(x))
df["qneardiff_hamming_mean"] = df["qneardiff_hamming"].apply(lambda x: jnp.mean(x))
df["memdiff_hamming_mean"] = df["memdiff_hamming"].apply(lambda x: jnp.mean(x))

df["q0diff_hamming_std"] = df["q0diff_hamming"].apply(lambda x: jnp.std(x) / x.shape[0])
df["qneardiff_hamming_std"] = df["qneardiff_hamming"].apply(lambda x: jnp.std(x) / x.shape[0])
df["memdiff_hamming_std"] = df["memdiff_hamming"].apply(lambda x: jnp.std(x) / x.shape[0])

ham_rand_guess = 0.5
hamming_range = [-.01, ham_rand_guess + 0.05] # 0.5 is the expected bax
def plot_ref_lines(ax):
    ax.axhline(ham_rand_guess, color='r', linestyle='--', label="Rand noise")

plot_fmt = 'o-'

def get_color(_b):
    colors_big_to_small = [
        "#28348E",
        "#458FBC",
        "#91CBBC",
        "#CEE8B9",
    ]
    return colors_big_to_small[_b]
    # return cmap(norm(_b))

def get_marker_size(_b):
    return 5

def get_dot_color(_b):
    colors_big_to_small = [
        "#060B4F",
        "#13508B",
        "#52A28B",
        "#A6D386",
    ]
    return colors_big_to_small[_b]

def get_linewidth(_b):
    widths_big_to_small = [
        3.25,
        2.75,
        2, 
        1.25
    ]
    return widths_big_to_small[_b]

def plot_beta_line(ax, xs, ys, errs, beta, _b):
    ax.errorbar(xs, ys, yerr=errs, label=f"beta={beta}", fmt=plot_fmt, color=get_color(_b), alpha=balpha, linewidth=get_linewidth(_b), mec=get_dot_color(_b), mfc=get_dot_color(_b), elinewidth=1.75)

for _d, d in enumerate(dims):
    dfd = df[df["d"] == d]

    dfdb_all = dfd[dfd['beta'].isin(set(betas))]

    for axcol, qdist in enumerate(["mem", "qnear", "q0"]):
        ax = axs[_d, axcol]

        for _b, beta in enumerate(betas):
            dfdb = dfd[dfd["beta"] == beta].sort_values(by="m")
            dfdb = dfdb[dfdb["m"].isin(ms)]
            mnow = dfdb['m']

            ykey = f"{qdist}diff_hamming_mean"
            stdkey = f"{qdist}diff_hamming_std"
            plot_beta_line(ax, mnow, dfdb[ykey], dfdb[stdkey], beta, _b)
            # ax.errorbar(mnow, dfdb[ykey], yerr=dfdb[stdkey], label=f"beta={int(beta)}", fmt='o:', color=cmap(norm(_b)), alpha=balpha)

            # N = len(dfdb['Eguess_abserr'])
            # Eguess_mean = np.array(dfdb['Eguess_abserr'].apply(lambda x: jnp.mean(x)))
            # Eguess_std = np.array(dfdb['Eguess_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(N)))
            # ax.errorbar(mnow, Eguess_mean, yerr=Eguess_std, fmt='x:', color='black', alpha=1.)

        ax.set_xticks(mticks)
        ax.set_xticklabels([])
        # ax.set_xticklabels(ms)

        ax.set_ylim(hamming_range)
        plot_ref_lines(ax)
        ax.relim()

        if axcol == 0:
            ax.set_ylabel(f"D={d}")
        if axcol != 0:
            ax.set_yticklabels([])

        if _d == 0:
            ax.set_title(f"Hamming err at {qdist}")
        
            # if axcol == 0:
            #     ax.legend(loc=(0.37, 0.3))

        if _d == (len(dims) - 1):
            # ax.set_xlabel("Y")
            ax.set_xticklabels(mshow, rotation=-90)


        # ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

# fig.tight_layout(w_pad=0.)
fig.tight_layout()
outdir = Path("figs/QUANT1b")
outdir.mkdir(exist_ok=True, parents=True)
fig.savefig(outdir / "QUANT1b_retrieval.png", bbox_inches='tight', dpi=350)
