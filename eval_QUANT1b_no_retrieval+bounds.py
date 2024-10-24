#%%
import pandas as pd
import numpy as np
import jax.numpy as jnp
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import *
from dataclasses import dataclass
import matplotlib as mpl
from matplotlib.colors import Normalize
from scipy import interpolate


@dataclass
class Args:
    ckpt_dir: str # Where to load the checkpoints
    device: str = "0" # Which device to run on. One of {"cpu", "0", "1", ...}, where numbers represent the GPU

    def get_device(self): return "" if self.device.lower() in ["", "cpu"] else self.device

args = Args(ckpt_dir="results/QUANT1b_near_0.1_retrieval/ckpts")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]=args.get_device()

resultsdir = Path(args.ckpt_dir)
fnames = list(resultsdir.glob("*.pkl"))

df = pd.concat([pd.read_pickle(f) for f in fnames])
df = df.drop_duplicates(subset=["dataset", "d", "beta", "m", "kernel"], inplace=False)

dims = sorted(df["d"].unique())
# ms = [5e3, 1e4, 4e4, 8e4, 1.2e5, 1.6e5, 2e5, 3e5, 4e5, 5e5]
ms = [4e4, 8e4, 1.2e5, 1.6e5, 2e5, 3e5, 4e5, 5e5]
ms = [int(m) for m in ms]
# mticks = [m for m in ms if m not in [500, 10_000]]
mticks = [m for m in ms if m not in [500, 10_000]]
mshow = [f"{m/1000:0.0f}k" for m in mticks]

# Choose betas we want to see
betas = df["beta"].unique()
betas = [10., 30., 40., 50.]
new_betas = list(sorted([b for b in betas if not (isinstance(b, str) or b == 1)], reverse=True))
betas = new_betas

norm = Normalize(vmin=0., vmax=len(betas)-1+0.25)
# norm = Normalize(vmin=-(len(betas)-1), vmax=0.25)
cmap = mpl.colormaps['plasma']

Ecols = ["E0", "Emem", "Enear"]
gradEcols = ["gradE0", "gradEmem", "gradEnear"]
Efig, Eaxs = plt.subplots(len(dims), len(Ecols), figsize=(len(Ecols)*3, len(dims)*3), squeeze=False)
gradEfig, gradEaxs = plt.subplots(len(dims), len(gradEcols), figsize=(len(gradEcols)*3, len(dims)*3), squeeze=False)

Eaxcol = 0
gradEaxcol = 0
nQ = df.iloc[0].q0.shape[0]
balpha = 1.0

# plot_fmt = 'o-'
plot_fmt = 'o'

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

# Compute random guess expectations
df['Eguess_abserr_mean'] = df['Eguess_abserr'].apply(lambda x: jnp.mean(x))
Erandom_guess = df.groupby("beta")['Eguess_abserr_mean'].mean().max()
df['gradEguess_abserr_mean'] = df['gradEguess_abserr'].apply(lambda x: jnp.mean(x))
gradErandom_guess = df.groupby("beta")['gradEguess_abserr_mean'].mean().max()

def plot_rand_line(ax, kind):
    if kind == "E":
        ax.axhline(Erandom_guess, color='r', linestyle='--', label="Rand guess")
    elif kind == "gradE":
        # ax.axhline(gradErandom_guess, color='r', linestyle='--', label="Rand guess")
        ax.axhline(gradErandom_guess, color='r', linestyle='--')
    else:
        raise ValueError(f"Invalid kind {kind}")

# Plot uppoer bound
xx = np.linspace(min(ms), max(ms), 100)
xtrue = np.array(mticks)
def boundfn(x, b):
    return b /np.sqrt(x)

def plot_gap(ax, ytrue, beta_idx, 
             rand_val=np.inf, # Only plot errors less than this value
):
    if any(ytrue > rand_val):
        return
    b_fit = np.max(np.sqrt(xtrue) * ytrue)
    sqrtY_curves.append((b_fit, ytrue))
    f = interpolate.interp1d(xtrue, ytrue, kind='linear', fill_value='extrapolate')
    yy = boundfn(xx, b_fit)
    ytrue_interp = f(xx)
    color = get_color(beta_idx)
    ax.plot(xx, yy, color=color, linestyle='-', linewidth=2)
    ax.fill_between(xx, ytrue_interp, yy, alpha=0.25, color=color)



for _d, d in enumerate(dims):
    dfd = df[df["d"] == d]

    dfdb_all = dfd[dfd['beta'].isin(set(betas))]
    normalized_Es = [
        dfdb_all['E0_abserr'].apply(lambda x: jnp.mean(x)), 
        dfdb_all['Enear_abserr'].apply(lambda x: jnp.mean(x)), 
        dfdb_all['Emem_abserr'].apply(lambda x: jnp.mean(x))
    ]
    Erange_max = max([E.max() for E in normalized_Es])
    Erange_max = Erange_max + 0.1 * Erange_max
    Erange_min = min([E.min() for E in normalized_Es])
    Erange_min = Erange_min / 2
    Erange = [Erange_min, Erange_max]
    Erange = [Erange_min, Erandom_guess + 0.1]

    normalized_gradEs = [
        dfdb_all['gradE0_abserr'].apply(lambda x: jnp.mean(x)), 
        dfdb_all['gradEnear_abserr'].apply(lambda x: jnp.mean(x)), 
        dfdb_all['gradEmem_abserr'].apply(lambda x: jnp.mean(x))
    ]
    gradErange_max = max([gradE.max() for gradE in normalized_gradEs]) * 2
    # gradErange_max = gradErange_max + 0.7 * gradErange_max
    gradErange_min = min([gradE.min() for gradE in normalized_gradEs]) / 2
    # gradErange_min = gradErange_min - 0.05 * gradErange_max
    gradErange = [gradErange_min, gradErange_max]

    ### Emem plots
    ax = Eaxs[_d, Eaxcol % len(Ecols)]
    Eaxcol+=1
    for _b, beta in enumerate(betas):
        dfdb = dfd[dfd["beta"] == beta].sort_values(by="m")
        dfdb = dfdb[dfdb["m"].isin(ms)]
        mnow = dfdb['m']
        Emem_mean = np.array([x for x in dfdb['Emem_abserr'].apply(lambda x: jnp.mean(x))])
        Emem_std = np.array(dfdb['Emem_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(nQ)))

        plot_beta_line(ax, mnow, Emem_mean, Emem_std, beta, _b)
        plot_gap(ax, Emem_mean, _b, rand_val=-np.inf)

    ax.set_xticks(mticks)
    ax.set_xticklabels([])

    ax.set_ylabel(f"D={d}")
    ax.set_ylim(Erange)
    print("Mem range: ", Erange)
    plot_rand_line(ax, "E")
    ax.set_yscale('log')
    ax.relim()

    if _d == 0:
        ax.set_title(r"E$\Delta$, at mems")
        # ax.legend(loc=(0.37, 0.3))

    if _d == (len(dims) - 1):
        ax.set_xlabel("Y")
        ax.set_xticklabels(mshow, rotation=-90)

    ax = Eaxs[_d, Eaxcol % len(Ecols)]
    Eaxcol+=1
    sqrtY_curves = []
    for _b, beta in enumerate(betas):
        dfdb = dfd[dfd["beta"] == beta].sort_values(by="m")
        dfdb = dfdb[dfdb["m"].isin(ms)]
        mnow = dfdb['m']

        Enear_mean = np.array([x for x in dfdb['Enear_abserr'].apply(lambda x: jnp.mean(x))])
        Enear_std = np.array(dfdb['Enear_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(nQ)))

        plot_beta_line(ax, mnow, Enear_mean, Enear_std, beta, _b)
        plot_gap(ax, Enear_mean, _b, rand_val=-np.inf)

    if _d == 0:
        ax.set_title(r"E$\Delta$, near mems")

    ax.set_yscale("log")
    ax.set_ylim(Erange)
    ax.set_yticklabels([])
    plot_rand_line(ax, "E")
    ax.relim()
    ax.set_xticks(mticks)
    ax.set_xticklabels([])

    if _d == (len(dims) - 1):
        ax.set_xlabel("Y")
        ax.set_xticklabels(mshow, rotation=-90)

    ### E0 plots
    ax = Eaxs[_d, Eaxcol % len(Ecols)]
    Eaxcol += 1
    for _b, beta in enumerate(betas):
        dfdb = dfd[dfd["beta"] == beta].sort_values(by="m")
        dfdb = dfdb[dfdb["m"].isin(ms)]
        mnow = dfdb['m']
# 
        # E0_mean = np.array(dfdb['E0_abserr'].apply(lambda x: jnp.mean(x)))
        E0_mean = np.array([x for x in dfdb['E0_abserr'].apply(lambda x: jnp.mean(x))])
        E0_std = np.array(dfdb['E0_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(nQ)))

        # ax.errorbar(mnow, E0_mean, yerr=E0_std, label=f"beta={beta}", fmt=plot_fmt, color=get_color(_b), alpha=balpha, linewidth=get_linewidth(_b), elinewidth=1.5)
        plot_beta_line(ax, mnow, E0_mean, E0_std, beta, _b)
        plot_gap(ax, E0_mean, _b, rand_val=-np.inf)

    if _d == 0:
        ax.set_title(r"E$\Delta$, at rand")

    ax.set_xticks(mticks)
    ax.set_xticklabels([])
    ax.set_ylim(Erange)
    ax.set_yscale('log')
    plot_rand_line(ax, "E")
    ax.set_yticklabels([])
    ax.relim()

    if _d == (len(dims) - 1):
        ax.set_xlabel("Y")
        ax.set_xticklabels(mshow, rotation=-90)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    # ### gradEmem plots
    ax = gradEaxs[_d, gradEaxcol % len(gradEcols)]
    gradEaxcol+=1
    for _b, beta in enumerate(betas):
        dfdb = dfd[dfd["beta"] == beta].sort_values(by="m")
        dfdb = dfdb[dfdb["m"].isin(ms)]
        mnow = dfdb['m']
        gradEmem_mean = np.array([x for x in dfdb['gradEmem_abserr'].apply(lambda x: jnp.mean(x))])
        gradEmem_std = np.array(dfdb['gradEmem_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(nQ)))

        # ax.errorbar(mnow, gradEmem_mean, yerr=gradEmem_std, label=f"beta={beta}", fmt=plot_fmt, color=get_color(_b), alpha=balpha, linewidth=get_linewidth(_b), elinewidth=1.5)
        plot_beta_line(ax, mnow, gradEmem_mean, gradEmem_std, beta, _b)
        plot_gap(ax, gradEmem_mean, _b, rand_val=gradErandom_guess)

        # if _b == (len(betas) - 1):
        #     # N = len(dfdb['Eguess_abserr'])
        #     gradEguess_mean = np.array(dfdb['gradEguess_abserr'].apply(lambda x: jnp.mean(x)))
        #     gradEguess_std = np.array(dfdb['gradEguess_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(x.shape[0])))
        #     ax.errorbar(mnow, gradEguess_mean, yerr=gradEguess_std, fmt='x:', color='red', alpha=1.)

    ax.set_xticks(mticks)
    ax.set_xticklabels([])
    plot_rand_line(ax, "gradE")
    ax.set_ylim(gradErange)
    ax.set_yscale('log')
    ax.relim()
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    if _d == 0:
        ax.set_title(r"gradE$\Delta$, at mems")
        # ax.legend(loc=(0.37, 0.32))

    if _d == (len(dims) - 1):
        ax.set_xlabel("Y")
        ax.set_xticklabels(mshow, rotation=-90)

    # ### gradEnear plots
    ax = gradEaxs[_d, gradEaxcol % len(gradEcols)]
    gradEaxcol+=1
    for _b, beta in enumerate(betas):
        dfdb = dfd[dfd["beta"] == beta].sort_values(by="m")
        dfdb = dfdb[dfdb["m"].isin(ms)]
        mnow = dfdb['m']

        gradEnear_mean = np.array([x for x in dfdb['gradEnear_abserr'].apply(lambda x: jnp.mean(x))])
        gradEnear_std = np.array(dfdb['gradEnear_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(nQ)))

        # ax.errorbar(mnow, gradEnear_mean, yerr=gradEnear_std, label=f"beta={beta}", fmt=plot_fmt, color=get_color(_b), alpha=balpha, linewidth=get_linewidth(_b), elinewidth=1.5)
        plot_beta_line(ax, mnow, gradEnear_mean, gradEnear_std, beta, _b)
        plot_gap(ax, gradEnear_mean, _b, rand_val=gradErandom_guess)

    if _d == 0:
        ax.set_title(r"gradE$\Delta$, near mems")

    ax.set_xticks(mticks)
    ax.set_xticklabels([])
    plot_rand_line(ax, "gradE")
    ax.set_ylim(gradErange)
    ax.set_yscale('log')
    ax.set_yticklabels([])
    ax.relim()
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    if _d == (len(dims) - 1):
        ax.set_xlabel("Y")
        ax.set_xticklabels(mshow, rotation=-90)

    # ### gradE0 plots
    ax = gradEaxs[_d, gradEaxcol % len(gradEcols)]
    gradEaxcol+=1
    for _b, beta in enumerate(betas):
        dfdb = dfd[dfd["beta"] == beta].sort_values(by="m")
        dfdb = dfdb[dfdb["m"].isin(ms)]
        mnow = dfdb['m']
        gradE0_mean = np.array([x for x in dfdb['gradE0_abserr'].apply(lambda x: jnp.mean(x))])
        gradE0_std = np.array(dfdb['gradE0_abserr'].apply(lambda x: jnp.std(x) / jnp.sqrt(nQ)))

        # ax.errorbar(mnow, gradE0_mean, yerr=gradE0_std, label=f"beta={beta}", fmt=plot_fmt, color=get_color(_b), alpha=balpha, linewidth=get_linewidth(_b), elinewidth=1.5)
        plot_beta_line(ax, mnow, gradE0_mean, gradE0_std, beta, _b)
        plot_gap(ax, gradE0_mean, _b, rand_val=gradErandom_guess)

    if _d == 0:
        ax.set_title(r"gradE$\Delta$, at rand")

    ax.set_xticks(mticks)
    ax.set_xticklabels([])
    plot_rand_line(ax, "gradE")
    ax.set_ylim(gradErange)
    ax.set_yscale('log')
    ax.set_yticklabels([])
    ax.relim()

    if _d == (len(dims) - 1):
        ax.set_xlabel("Y")
        ax.set_xticklabels(mshow, rotation=-90)

#%%

nqueries = df.iloc[0].q0.shape[0]
nmems = df.iloc[0].memories.shape[0]
Efig.tight_layout()
# Efig.savefig(f"figs/QUANT1b_E+bounds.png", bbox_inches='tight', dpi=350) # Technically, we never described the bounds of this one
Efig.show()

gradEfig.tight_layout()
outdir = Path("figs/QUANT1b")
outdir.mkdir(exist_ok=True, parents=True)
gradEfig.savefig(outdir / "QUANT1b_gradE+bounds.png", bbox_inches='tight', dpi=350)
gradEfig.show()

# %%
