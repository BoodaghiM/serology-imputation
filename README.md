# Serology Imputation Framework

![CI](https://github.com/BoodaghiM/serology-imputation/actions/workflows/ci.yml/badge.svg)

A modular, reproducible framework for **missing-value imputation in serological datasets**, supporting both classical statistical methods and deep-learning approaches.

Designed for **research and production-grade pipelines**, with scalable execution via **local Python, Docker, Ray, and Nextflow (HPC/workflow engines)**.

---

## Key Features

- **Multiple imputation per file** (MI-style outputs)
- **Deterministic, reproducible execution**
- **Parallelized execution** with Ray
- **Workflow orchestration** via Nextflow
- **Containerized execution** with Docker

### Supported Imputation Methods
- **Iterative Random Forest**
- **Iterative Bayesian Ridge**
- **Masked-loss Autoencoder (PyTorch)**

### Input / Execution Modes
- Single CSV file
- Directory trees of datasets
- Local Python
- Docker
- Nextflow (HPC / workflow engines)

---

## Installation (Local / Development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
