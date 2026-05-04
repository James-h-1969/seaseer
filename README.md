<div align="center">

# 🌊 SeaSeer

**Neural ODE-based spatiotemporal forecasting for ocean & climate data**

[![CI](https://github.com/James-h-1969/thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/James-h-1969/thesis/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

</div>

---

## Overview

SeaSeer is a deep learning model that uses **Neural ODEs** to learn transport dynamics for spatiotemporal ocean/climate forecasting. It combines a convolutional velocity network with optional attention and emission (uncertainty) branches built on a ResNet backbone.

## Getting Started

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (package manager)

### Installation

```bash
git clone git@github.com:James-h-1969/thesis.git
cd thesis
uv sync
pre-commit install
```

### Training

```bash
cd seaseer
make train
```

### Evaluation

```bash
cd seaseer
make eval
```

### Data Generation

Before downloading data, you need to set up credentials for the two data providers:

1. **Copernicus Climate Data Store (ERA5)**: Register at [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu), then create `~/.cdsapirc` with:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-api-key>
   ```

2. **Copernicus Marine (CMEMS)**: Register at [https://data.marine.copernicus.eu/register](https://data.marine.copernicus.eu/register), then log in once:
   ```bash
   uv run copernicusmarine login
   ```

Then generate the data:
```bash
make generate_data
```
This downloads data covering the Great Barrier Reef region (10°S–25°S, 142°E–154°E) from 1993–2018.

## Project Structure

```
seaseer/
├── model.py              # SeaSeer model definition
├── train.py              # Training loop
├── eval.py               # Evaluation script
└── README.md             # SeaSeer-specific docs
helpers/
├── models/
│   ├── ResidualBlock.py  # Residual block module
│   └── ResidualNetwork.py# ResNet backbone
└── scripts/
    └── generate_data.py  # Data download & generation
tests/
├── conftest.py           # Shared test fixtures
└── test_models.py        # Test suite
Makefile                  # Train/eval/data shortcuts
```

## Thesis
This is part of the work done by James Hocking for his thesis 'Seaseer: A Neural ODE for predicting Marine Heat Waves.
