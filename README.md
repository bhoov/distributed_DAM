# Distributed Representations for Dense Associative Memory (**DrDAM**)

> Codebase accompanying the NeurIPS 2024 paper ["Distributed Memory through the Lens of Random Features"]()

**DrDAM** is the first technique to show how memories can be distributed in Dense Associative Memories (DenseAMs), much like how memories were distributed in the original Hopfield network. The traditional **Memory Representation for Dense Associative Memory (MrDAM)** is a "slot-based" associative memory where each memory is represented as a "slot" (i.e., row or column) of a weight matrix. New memories are added by concatenating a new vector to an existing weight matrix. DrDAM takes advantage of Random Features to store patterns via summation into a weight tensor of constant size rather than concatenation. See Figure 1 below (from the paper) for an illustration. DrDAM closely approximates both the energy and fixed-point dynamics of the traditional **M**emory **R**epresentation for **D**ense **A**ssociative **M**emory (MrDAM) while having a parameter space of constant size.

<figure>
  <img src="assets/Fig1.png" alt="Figure 1">
  <figcaption>Figure 1: The Distributed Representation for Dense Associative Memory (DrDAM) approximates both the energy and fixed-point dynamics of the traditional Memory Representation for Dense Associative Memory (MrDAM) while having a parameter space of constant size.</figcaption>
</figure>

This repository contains the code for recreating all experiments of the main paper. All reported results were created on a single L40s GPU with ~48GB of VRAM.

## Installation
```bash
conda env create -f environment.yml
conda activate distributed_DAM
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]" # Match CUDA version to your GPU
make data # Download data. Takes ~10 min depending on internet speed
```

## Experiment Descriptions

We follow a "shallow" directory structure that obviates the need for submodules and editable pip installations. Run all commands from the root directory.

### (Fig 1) "Pure INFerence on Tiny Imagenet" [PINF]

Recreate the trajectories shown in the right half of Fig 1 of our paper by running the following code:

```
python exp_PINF.py
```

- Runs on GPU 0 by default.
- Output figures saved in `figs/FIG1`

### (Fig 2) "Pure INFerence on Tiny Imagenet: a compression scenario" [PINF2]