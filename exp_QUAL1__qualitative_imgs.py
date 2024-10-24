""" Plot the retrieval results from the energy dynamics """

#%% Testing inference code

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from data_utils import get_data, DataOpts
from typing import *
from kernel_sims import SIM_REGISTRY as KernelOpts
from tqdm.auto import tqdm
import pandas as pd
import path_fixes as pf
import os
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%%
if __name__ == "__main__":
    C, H, W = 3, 32, 32
    d = C * H * W
    im2pattern = lambda im: rearrange(im, "... h w c -> ... (h w c)") / jnp.sqrt(d)
    pattern2im = lambda pat: rearrange(pat, "... (h w c) -> ... h w c", h=32, w=32, c=3) * jnp.sqrt(d)

    nmems = 10
    seed = 56
    data = get_data(DataOpts.cifar10)
    idxs = jr.choice(jr.PRNGKey(seed), jnp.arange(data.shape[0]), (nmems,), replace=False)
    qidx = 6 # For seed 56, indexes into the memories

    memories = jnp.array(data[idxs])
    query = jnp.array(memories[qidx])

    # # Visualize memories
    # fig, axs = plt.subplots(10, 10, figsize=(20,20))
    # k = 0
    # for i in range(10):
    #     for j in range(10):
    #         ax = axs[i,j]
    #         ax.imshow(memories[k])
    #         ax.axis('off')
    #         ax.set_title(f"{k}")
    #         k += 1
    # fig.tight_layout()
    # fig.suptitle(f"SEED {seed}")
    # fig.show()

    memories = im2pattern(memories)

    ckpt_dir = pf.RESULTS / "QUAL1_big"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    im_og_f = ckpt_dir / "im_og.npy"
    im_corrupted_f = ckpt_dir / "im_corrupted.npy"
    qout_standard_f = ckpt_dir / "standard_results.pkl"
    qout_kernel_f = ckpt_dir / "kernel_results.pkl"

    betas = [ 40., 60., 70., 75., 80.]
    ms = [10_000, 40_000, 90_000, 160_000, 360_000]

    kernel_name = "SinCosL2DAM"
    mask_after = d//2

    im_og = im2pattern(query)
    im_corrupted = im_og.at[mask_after:].set(0)

    np.save(im_og_f, im_og)
    np.save(im_corrupted_f, im_corrupted)

    df_standards = pd.read_pickle(qout_standard_f) if qout_standard_f.exists() else pd.DataFrame()
    df_kernels = pd.read_pickle(qout_kernel_f) if qout_kernel_f.exists() else pd.DataFrame()
    qouts_standard = df_standards.to_dict('records')
    qouts_kernel = df_kernels.to_dict('records')

    if len(df_standards):
        existing_betas = set(df_standards['beta'].values)
        existing_beta_ms = set([(b, m) for b, m in zip(df_kernels['beta'].values, df_kernels['m'].values)])
    else:
        existing_betas = set([])
        existing_beta_ms = set([])
        

    with tqdm(total=len(betas) * len(ms)) as pbar:
        for beta in betas:
            q = im2pattern(query).at[mask_after:].set(0)
            clamp_idxs = jnp.arange(d) < mask_after
            if beta not in existing_betas:
                kdam = KernelOpts[kernel_name](jr.PRNGKey(seed), d, 10, beta=beta, add_bias=False)
                qout, logs = kdam.recall(q, memories, clamp_idxs=clamp_idxs, depth=400)
                qouts_standard.append({
                    "beta": beta,
                    "qout": qout,
                    "energies": logs['energies'].T
                })

            for m in ms:
                pbar.set_description(f"beta={beta}, m={m}")
                if (beta, m) in existing_beta_ms:
                    pbar.update(1)
                    continue

                kdam = KernelOpts[kernel_name](jr.PRNGKey(seed), d, m, beta=beta, add_bias=False)

                T = kdam.kernelize_memories(memories)
                qout_kernel, logs_kernel = kdam.kernel_recall(q, T, clamp_idxs=clamp_idxs, depth=3600)
                qouts_kernel.append({
                    "m": m,
                    "beta": beta,
                    "qout": qout_kernel,
                    "energies": logs_kernel['energies'].T
                })
                pbar.update(1)

                # Slowly build up
                df_standards = pd.DataFrame(qouts_standard)
                df_standards.to_pickle(qout_standard_f)

                df_kernels = pd.DataFrame(qouts_kernel)
                df_kernels.to_pickle(qout_kernel_f)


    #%% Plotting
    # qouts_standard
    # df_standards
    from einops import rearrange, repeat

    im_og = np.load(im_og_f)
    im_corrupted = np.load(im_corrupted_f)
    qouts_standard = pd.read_pickle(qout_standard_f).to_dict("records")
    qouts_kernel = pd.read_pickle(qout_kernel_f).to_dict("records")

    betas = sorted(list(set([qk['beta'] for qk in qouts_kernel])))
    betas = [b for b in betas if b not in set([1., 10., 20., 50., 90., 120.])]

    ms = sorted(list(set([qk['m'] for qk in qouts_kernel])))
    ms = [m for m in ms if m not in set([360.])]

    standard_recons = sorted(qouts_standard, key=lambda x: x['beta'])
    standard_recons = [srecon for srecon in standard_recons if srecon['beta'] in set(betas)]
    standard_recons = rearrange([pattern2im(qout['qout']) for qout in standard_recons], "kh h w c -> (kh h) w c")

    im_og_show = repeat(pattern2im(im_og), "h w c -> (kh h) w c", kh=len(betas))
    im_corrupted_show = repeat(pattern2im(im_corrupted), "h w c -> (kh h) w c", kh=len(betas))

    kernel_recons = sorted(qouts_kernel, key=lambda x: (x['beta'], x['m']))
    kernel_recons_all = []
    for beta in betas:
        krecons = sorted(list(filter(lambda x: x['beta'] == beta, kernel_recons)), key=lambda x: x['m'])
        krecons = [krecon for krecon in krecons if krecon['m'] in set(ms)]
        kernel_recons_all.append([pattern2im(krecon['qout']) for krecon in krecons])

    [[k.shape for k in kr] for kr in kernel_recons_all]
    kernel_recons = np.array(kernel_recons_all)
    kernel_recons = rearrange(kernel_recons, "kb km h w c -> (kb h) (km w) c")
    kernel_recons.shape

    # imtotal = jnp.concatenate([im_corrupted_show, kernel_recons, standard_recons, im_og_show], axis=1)
    imtotal = jnp.concatenate([im_corrupted_show, kernel_recons, standard_recons], axis=1)

    fig, ax = plt.subplots(1, figsize=(7,7))

    beta_ticks = [int(16 + 32 * i) for i in range(len(betas))]

    m_ticklabels = ["Corrupted"] + [f"Y={m/1000:0.0f}k" for m in ms] + [f"Standard"] #+ ["Original"]
    m_ticks = [int(16 + 32 * i) for i in range(len(m_ticklabels))]

    ax.imshow(imtotal)
    # ax.axis('off')
    ax.set_yticks(beta_ticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_yticks([16 + 32 * i for i in range(len(betas))], minor=True)
    ax.set_yticklabels([f"$\\beta=${beta:0.0f}" for beta in betas], fontsize=12)

    # ax.axis('off')
    # ax.set_ylabel("Beta")

    ax.set_xticks(m_ticks)
    ax.set_xticklabels(m_ticklabels, fontsize=9)
    plt.show()

    outdir = Path("figs/QUAL1")
    outdir.mkdir(exist_ok=True, parents=True)
    fig.savefig(outdir / "QUAL1__cifar10__reconstructions.png")
    # fig.savefig("figs/QUAL1__cifar10__reconstructions.png")

# %%
