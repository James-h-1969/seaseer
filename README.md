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

## Project Structure

```
seaseer/
├── model.py              # SeaSeer model definition
├── train.py              # Training loop
├── eval.py               # Evaluation script
└── Makefile              # Train/eval shortcuts
helper_models/
├── ResidualBlock.py      # Residual block module
└── ResidualNetwork.py    # ResNet backbone
tests/
└── test.py               # Test suite
```