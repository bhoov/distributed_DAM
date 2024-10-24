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

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# resultsdir = Path("results/QUANT2--data=letter_nobias/ckpts")
resultsdir = Path("results/QUANT2-manybeta/ckpts")

fnames = list(resultsdir.glob("*.pkl"))
df = pd.concat([pd.read_pickle(f) for f in fnames])

def get_deltas(row, fix_offset_energies=False):
    def get_mae(x, y, fix_offset=False):
        if fix_offset:
            # polyfit and shift
            coefs = np.polyfit(x, y, 1)
            y = y - coefs[-1]

        mae = jnp.abs(x - y)
        return np.mean(mae), np.std(mae) / np.sqrt(row.nqueries) 
    
    delta_Emems_mean, delta_Emems_std = get_mae(row.Emems, row.Emems_kernel, fix_offset=fix_offset_energies)
    delta_logEmems_mean, delta_logEmems_std = get_mae(row.logEmems, row.logEmems_kernel, fix_offset=fix_offset_energies)
    delta_gradEmems_mean, delta_gradEmems_std = get_mae(row.gradEmems, row.gradEmems_kernel, fix_offset=False)
    delta_gradlogEmems_mean, delta_gradlogEmems_std = get_mae(row.gradlogEmems, row.gradlogEmems_kernel, fix_offset=False)

    delta_Equeries_mean, delta_Equeries_std = get_mae(row.Equeries, row.Equeries_kernel, fix_offset=fix_offset_energies)
    delta_logEqueries_mean, delta_logEqueries_std = get_mae(row.logEqueries, row.logEqueries_kernel, fix_offset=fix_offset_energies)
    delta_gradEqueries_mean, delta_gradEqueries_std = get_mae(row.gradEqueries, row.gradEqueries_kernel, fix_offset=False)
    delta_gradlogEqueries_mean, delta_gradlogEqueries_std = get_mae(row.gradlogEqueries, row.gradlogEqueries_kernel, fix_offset=False)

    return pd.Series([ 
        delta_Emems_mean, delta_Emems_std,
        delta_logEmems_mean, delta_logEmems_std,
        delta_gradEmems_mean, delta_gradEmems_std,
        delta_gradlogEmems_mean, delta_gradlogEmems_std,

        delta_Equeries_mean, delta_Equeries_std, 
        delta_logEqueries_mean, delta_logEqueries_std,
        delta_gradEqueries_mean, delta_gradEqueries_std,
        delta_gradlogEqueries_mean, delta_gradlogEqueries_std,
        ])

delta_cols = [    "delta_Emems_mean", "delta_Emems_std",
    "delta_logEmems_mean", "delta_logEmems_std",
    "delta_gradEmems_mean", "delta_gradEmems_std",
    "delta_gradlogEmems_mean", "delta_gradlogEmems_std",
    "delta_Equeries_mean", "delta_Equeries_std", 
    "delta_logEqueries_mean", "delta_logEqueries_std",
    "delta_gradEqueries_mean", "delta_gradEqueries_std",
    "delta_gradlogEqueries_mean", "delta_gradlogEqueries_std"
]

df[delta_cols] = df.apply(lambda row: get_deltas(row, fix_offset_energies=True), axis=1)
drop_columns = ['queries', 'memories', 'Emems', 'logEmems', 'gradEmems', 'gradlogEmems',
    'Equeries', 'logEqueries', 'gradEqueries', 'gradlogEqueries',
    'Emems_kernel', 'logEmems_kernel', 'gradEmems_kernel',
    'gradlogEmems_kernel', 'Equeries_kernel', 'logEqueries_kernel',
    'gradEqueries_kernel', 'gradlogEqueries_kernel']

dfsum = df.drop(columns=drop_columns)
ms = sorted(dfsum['m'].unique())
print("ALL MS", ms)
ms = [40000, 80000, 160000, 400000, 800000]
mticks = ms
mshow = [f"{m/1000:0.0f}k" for m in mticks]
betas = sorted(dfsum['beta'].unique())
print("ALL BETAS: ", betas)
betas = [1, 10, 30, 60]
kernels = sorted(dfsum['kernel'].unique())
kernels = ["CosL2DAM", "SinCosL2DAM", "ExpL2DAM", "ExpExpL2DAM"]

cols = [
    { "val": "logEmems", "name": r"$\Delta \log E_\rho$" },
    { "val": "logEqueries", "name": r"$\Delta \log E_q$" },
    { "val": "gradlogEmems", "name": r"$\Delta (\nabla \log E_\rho)$" },
    { "val": "gradlogEqueries", "name": r"$\Delta (\nabla \log E_q)$" },
]
rows = [{
    "val": beta, 
    # "name": r"$\beta^\star =" + f"{beta:0.2f}$" if beta > 100. else r"$\beta_\text{low} =" + f"{beta:0.2f}$"
    "name": r"$\beta =" + f"{beta:0.0f}$"} for beta in betas]

rose_of_sharon = {
        '50': '#fffbeb',
        '100': '#fef3c7',
        '200': '#fde58a',
        '300': '#fbd24e',
        '400': '#fabe25',
        '500': '#f49d0c',
        '600': '#d87607',
        '700': '#bc560a',
        '800': '#923f0e',
        '900': '#78340f',
        '950': '#451a03',
    }


aquamarine = {
        '50': '#effef9',
        '100': '#c9feee',
        '200': '#90fbde',
        '300': '#56f2cd',
        '400': '#24ddb8',
        '500': '#0bc19f',
        '600': '#069b82',
        '700': '#097c6a',
        '800': '#0d6256',
        '900': '#105148',
        '950': '#02312c',
    }


kernel_colors = {
    "CosL2DAM": aquamarine['300'],
    "SinCosL2DAM": aquamarine['600'],
    "ExpL2DAM": rose_of_sharon['300'],
    "ExpExpL2DAM": rose_of_sharon['600']
}

kdot_colors = {
    "CosL2DAM": "#1DE6A5",
    "SinCosL2DAM": "#005E42",
    "ExpL2DAM": "#F7AD18",
    "ExpExpL2DAM": "#B73700",
}

nrows = len(rows)
ncols = len(cols)

# fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))

dfm = dfsum[dfsum['m'].isin(set(ms))]

normalized_Es = [dfm["delta_logEmems_mean"], dfm["delta_logEqueries_mean"]]
Erange_max = max([E.max() for E in normalized_Es])
Erange_max = Erange_max + 0.2 * Erange_max
Erange_min = min([E.min() for E in normalized_Es])
Erange_min = Erange_min /2 
Erange = [Erange_min, Erange_max]

normalized_gradEs = [dfm["delta_gradlogEmems_mean"], dfm["delta_gradlogEqueries_mean"]]
gradErange_max = max([E.max() for E in normalized_gradEs])
gradErange_max = gradErange_max + 0.2 * gradErange_max
gradErange_min = min([E.min() for E in normalized_gradEs])
gradErange_min = gradErange_min / 2.
gradErange = [gradErange_min, gradErange_max]

for i, row in enumerate(rows):
    dfnow = dfm.query(f"beta == {row['val']}")
    for j, col in enumerate(cols):
        for kname in kernels:
            dfnowk = dfnow.query(f"kernel == '{kname}'").sort_values(by="m")
            ms = dfnowk['m']
            ax = axs[i, j]
            beta = row["val"]
            valmean = f"delta_{col['val']}_mean"
            valstd = f"delta_{col['val']}_std"

            meanvals = dfnowk[valmean]
            stdvals = dfnowk[valstd]

            ax.set_xticks(mticks)
            ax.set_xticklabels([])
            mc = kdot_colors[kname]
            ax.errorbar(ms, meanvals, yerr=stdvals, fmt='o-', alpha=1., color=kernel_colors[kname], mec=mc, mfc=mc, label=kname)
            if i == 0:
                ax.set_title(col['name'], y=1.04)
            if i == len(rows) - 1:
                ax.set_xticklabels(mshow, rotation=-90)
                ax.set_xlabel("Y")
            if j == 0:
                ax.set_ylabel(row['name'])
            else:
                ax.set_yticklabels([])

            if "grad" in col['val']:
                ax.set_ylim(gradErange)
            else:
                ax.set_ylim(Erange)

            if i==0 and j==0:
                ax.legend()
            ax.set_yscale('log')
            ax.relim()
        # ax.set_title(f"{name} {row['name']}")

dataset = dfsum['data'].unique()[0]
nqueries = dfsum['nqueries'].unique()[0]
nmemories = dfsum['nmemories'].unique()[0]
add_bias = dfsum['add_bias'].unique()[0]
fig.suptitle(f"{dataset} data, nqueries={nqueries}, nmemories={nmemories}{', with bias' if add_bias else ', no bias'}", fontsize=14)
fig.tight_layout()
fig.show()

used_all_kernels = len(kernels) == len(dfsum['kernel'].unique())
outdir = Path("figs/QUANT2")
outdir.mkdir(exist_ok=True, parents=True)
fig.savefig(outdir / "QUANT2--ablations.png", dpi=350)

# %%
