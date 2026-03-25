
# EMRI Search

This repository implements a search pipeline for detecting Extreme Mass Ratio Inspirals (EMRIs) in simulated LISA data. Given input LISA time-series, the pipeline returns the best-matching instantaneous frequency evolution (track) found in the data. The code accompanies the related paper: [Ab uno disce omnes: Single-harmonic search for extreme mass-ratio inspirals](https://arxiv.org/abs/2510.20891). Intermediate search results can be found [here](https://public.spider.surfsara.nl/project/lisa_nlddpc/SearchResults/figures/tdi_search/). 
If you are interested only in which sources we were able to detect check these [interactive visualizations](https://huggingface.co/spaces/lorenzsp/emrisearch).

For a quick, simplified EMRI search, open and run [QuickStartEMRIsearch.ipynb](examples/QuickStartEMRIsearch.ipynb). The notebook walks through the pipeline and produces example outputs, including:

- Injected waveform and three harmonics with their corresponding detection statistic values.

  ![Waveform Injection](examples/quick_start_results/true_frequency_track.png)

- Top panel: the injected EMRI signal with noise. Bottom panel: the recovered signal after masking the best-fit track of the dominant harmonic.

  ![Data Comparison](examples/quick_start_results/data_comparison.png)

- Visualizations of the optimization process and the recovered track evolution.

<p align="center">
  <img src="examples/quick_start_results/frequency_evolution_2panel_animation.gif" alt="Track Evolution" style="width:100%;"/>
  <img src="examples/quick_start_results/optimization_animation.gif" alt="Search Scatter" style="width:100%;"/>
</p>

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Core Utilities](#core-utilities)
- [Citation](#citation)
- [Contact](#contact)

## Installation

<!-- `emrisearch` should be available under PyPI and thus should be easily instalable using
```bash
pip install emrisearch
``` -->

### Create a conda environment
Recommended minimal environment:
```bash
conda create -n emri -y python=3.12
conda activate emri
pip install -e .
```
with the flag `-e` for development.

There is one package that needs to be installed from source `fastlisaresponse`. 
On macOS arm64 a working build sequence used by the author:
```bash
git clone https://github.com/mikekatz04/lisa-on-gpu.git
cd lisa-on-gpu
git checkout v1.2.1a0

conda install conda-forge::clang_osx-arm64 conda-forge::clangxx_osx-arm64 -y
export CC=$(which clang)
export CXX=$(which clang++)
export CFLAGS="-march=native"
export CXXFLAGS="-march=native"

pip install .
python -m unittest discover
```

Test most of the functions of the package with:
```bash
python -m unittest tests.test_demo
```

## Quick Start

Open [QuickStartEMRIsearch.ipynb](examples/QuickStartEMRIsearch.ipynb) and run the notebook. It guides you through:
- generating EMRI test signals,
- computing Short Fourier Transforms (SFTs),
- computing the detection statistic,
- and running example searches that reproduce the figures above.

## Repository Structure

High-level overview of the modules's key files and their purposes. 
Each script can be run to generate a demonstration plot showing typical usage of the functions it defines.


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

If you use this code, please cite the paper:
```
@ARTICLE{2026PhRvD.113b4061S,
       author = {{Speri}, Lorenzo and {Tenorio}, Rodrigo and {Chapman-Bird}, Christian and {Gerosa}, Davide},
        title = "{Single-harmonic search for extreme mass-ratio inspirals}",
      journal = {\prd},
     keywords = {General relativity, alternative theories of gravity, General Relativity and Quantum Cosmology, High Energy Astrophysical Phenomena, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = jan,
       volume = {113},
       number = {2},
          eid = {024061},
        pages = {024061},
          doi = {10.1103/dh3j-ksfl},
archivePrefix = {arXiv},
       eprint = {2510.20891},
 primaryClass = {gr-qc},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026PhRvD.113b4061S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contact

For questions or issues open an issue on the GitHub repository.
