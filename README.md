
# EMRI Search

This repository implements a search pipeline for detecting Extreme Mass Ratio Inspirals (EMRIs) in simulated LISA data. Given input LISA time-series, the pipeline returns the best-matching instantaneous frequency evolution (track) found in the data. The code accompanies the related paper: [Ab uno disce omnes: Single-harmonic search for extreme mass-ratio inspirals]().

For a quick, simplified EMRI search, open and run [QuickStartEMRIsearch.ipynb](QuickStartEMRIsearch.ipynb). The notebook walks through the pipeline and produces example outputs, including:

- Injected waveform and three harmonics with their corresponding detection statistic values.

  ![Waveform Injection](quick_start_results/true_frequency_track.png)

- Top panel: the injected EMRI signal with noise. Bottom panel: the recovered signal after masking the best-fit track of the dominant harmonic.

  ![Data Comparison](quick_start_results/data_comparison.png)

- Visualizations of the optimization process and the recovered track evolution.

<p align="center">
  <img src="quick_start_results/frequency_evolution_2panel_animation.gif" alt="Track Evolution" style="width:100%;"/>
  <img src="quick_start_results/optimization_animation.gif" alt="Search Scatter" style="width:100%;"/>
</p>

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Core Utilities](#core-utilities)
- [Citation](#citation)
- [Contact](#contact)

## Installation

### Create a conda environment
Recommended minimal environment:
```bash
conda create -n emri -y python=3.12
conda activate emri
pip install eryn fastemriwaveforms multiprocess optax matplotlib scipy jupyter interpax matplotlib-label-lines lisaanalysistools astropy corner pastamarkers
```

On many Linux systems `fastlisaresponse` installs with:
```bash
pip install fastemriwaveforms
```

On macOS arm64 a working build sequence used by the author:
```bash
git clone https://github.com/mikekatz04/lisa-on-gpu.git
cd lisa-on-gpu

conda install conda-forge::clang_osx-arm64 conda-forge::clangxx_osx-arm64 -y
export CC=$(which clang)
export CXX=$(which clang++)
export CFLAGS="-march=native"
export CXXFLAGS="-march=native"

pip install .
python -m unittest discover
```

Quick test for the response code:
```bash
python da_utils.py
```

## Quick Start

Open [QuickStartEMRIsearch.ipynb](QuickStartEMRIsearch.ipynb) and run the notebook. It guides you through:
- generating EMRI test signals,
- computing Short Fourier Transforms (SFTs),
- computing the detection statistic,
- and running example searches that reproduce the figures above.

## Repository Structure

High-level overview of the repository’s key files and their purposes. Each script can be run to generate a demonstration plot showing typical usage of the functions it defines. You can run all scripts with 

```bash
python da_utils.py; python emri_utils.py; python search_utils.py; python jax_utils.py; python jax_de_utils.py; python draw_population.py;
```

### Core Utilities

Data analysis and waveform helpers:
- `da_utils.py` — SFT computation, frequency-domain inner products, noise generation, SNR calculation.
- `emri_utils.py` — EMRI waveform and trajectory helpers
- `search_utils.py` — detection statistics and search routines
- `draw_population.py` — draw synthetic EMRI populations and parameter transforms.

### JAX / GPU-Accelerated Utilities
- `jax_utils.py` — JAX implementations of core routines for GPU/TPU acceleration (PSDs, detection statistics).
- `jax_de_utils.py` — differential-evolution optimization implemented with JAX for fast population-based searches.

## Citation

If you use this code, please cite the paper []().

## Contact

For questions or issues open an issue on the GitHub repository
