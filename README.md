Serology Imputation Framework

A modular, reproducible framework for missing-value imputation in serological datasets, supporting classical statistical methods and deep-learning approaches.
Designed for research use, with scalable execution via Docker, Ray, and Nextflow.

Features

Multiple imputation per file (MI-style outputs)

Supported methods:

Iterative Random Forest

Iterative Bayesian Ridge

Masked-loss Autoencoder (PyTorch)

Deterministic output naming

Works on:

Single CSV files

Directory trees of datasets

Execution modes:

Local Python

Docker

Nextflow (HPC / workflow engines)

Installation (Local / Development)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .


(Optional) Run tests:

pytest -q

Quickstart — Single File (Local or Docker)
Example
serology-impute \
  --file data/nan_added_40.csv \
  --method iterative_bayesridge \
  --seed 1 \
  --out-dir outputs/


Output naming:

nan_added_40_imp1_imputed.csv

Bulk Imputation (Docker)
Build image
docker build -t serology-imputation:latest .

Run with mounted data
docker run --rm \
  -v /path/to/Data:/data \
  -v /path/to/Outputs:/outputs \
  serology-imputation:latest \
  --config /configs/params.yaml

Configuration File (YAML)

Example params.yaml:

input_dirs:
  - /data/Numerical_data_MAR_1/data
  - /data/Numerical_data_MAR_2/data
  - /data/Numerical_data_MAR_3/data

output_root: /outputs

imputation_settings:
  iterative_rf: 2
  iterative_bayesridge: 2
  autoencoder: 2

# Optional
# epochs: 200

Output structure
outputs/
└── Numerical_data_MAR_1
    └── iterative_bayesridge
        ├── nan_added_40_imp1_imputed.csv
        └── nan_added_40_imp2_imputed.csv

Nextflow Execution
Run
nextflow run main.nf \
  -params-file configs/params.yaml \
  -with-docker

Resume cached runs
nextflow run main.nf \
  -params-file configs/params.yaml \
  -with-docker \
  -resume

Methods Overview
Method	Description
Iterative RF	IterativeImputer with RandomForestRegressor
Bayesian Ridge	IterativeImputer with BayesianRidge
Autoencoder	Masked-loss autoencoder (PyTorch)

All methods operate only on numeric columns, preserving ID columns unchanged.

Reproducibility Notes

Each imputation index (imp1, imp2, …) maps to a deterministic random seed

CPU usage is controlled to avoid oversubscription

Docker + Nextflow ensure environment consistency

License

MIT License (see LICENSE).

Status

Research-grade software.
API stability is not guaranteed, but outputs and workflows are reproducible.
