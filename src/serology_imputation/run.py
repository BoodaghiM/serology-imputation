from __future__ import annotations

import os
# ---- ENVIRONMENT SETUP (Must be before Ray/NumPy/Torch) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "fatal")

import argparse
from pathlib import Path
import yaml
import ray
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

# Package imports
from serology_imputation.imputers import (
    IterativeRFImputer,
    IterativeBayesRidgeImputer,
    AutoencoderImputer,
    AutoencoderConfig,
)
from serology_imputation.utils import list_csv_files, write_df, ensure_dir

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Serology Imputation CLI (Ray + Nextflow compatible)")

    # 1) Config Mode (Bulk)
    p.add_argument("--config", help="Path to YAML config for bulk Ray mode")

    # 2) Worker Mode (Nextflow/Single-File)
    p.add_argument("--file", help="Path to a single input CSV (triggers worker mode)")
    p.add_argument(
        "--method",
        choices=["iterative_rf", "iterative_bayesridge", "autoencoder"],
        help="Method for single-file mode",
    )

    # We keep --seed for backward compatibility, but treat it as the "imputation index" (imp #)
    p.add_argument("--seed", type=int, help="Imputation index (imp #) for single-file mode (kept as --seed)")

    # 3) Overrides (Works for both modes)
    p.add_argument("--input-dirs", nargs="+", help="Explicit list of input directories (overrides config)")
    p.add_argument("--output-root", help="Root directory for results (overrides config)")
    p.add_argument("--epochs", type=int, help="Optional epochs for Autoencoder (overrides config)")
    p.add_argument("--out-dir", help="Specific output dir (used primarily by Nextflow)")

    return p.parse_args()


def build_imputer(method: str, seed: int, epochs: int | None = None):
    """Factory to build the imputer with optional hyperparameter overrides."""
    if method == "iterative_rf":
        return IterativeRFImputer(seed=seed)

    if method == "iterative_bayesridge":
        return IterativeBayesRidgeImputer(seed=seed)

    if method == "autoencoder":
        if epochs is None:
            return AutoencoderImputer(seed=seed)
        cfg = AutoencoderConfig(epochs=epochs)
        return AutoencoderImputer(seed=seed, config=cfg)

    raise ValueError(f"Unknown method: {method}")


def make_output_name(csv_path: Path, imp: int) -> str:
    """
    Professional naming:
      <original_stem>_imp<k>_imputed.csv
    Example:
      nan_added_40_imp1_imputed.csv
    """
    return f"{csv_path.stem}_imp{imp}_imputed{csv_path.suffix}"


def process_single_file(csv_path: Path, method: str, imp: int, out_path: Path, epochs: int | None = None):
    """
    Atomic task: read one CSV, impute numeric columns, write output.

    Notes:
      - `imp` is both the imputation index and the RNG seed (deterministic per imp).
      - We assume the first column is an ID column and all others are numeric-ish.
    """
    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError(f"CSV has <2 columns; expected ID + numeric columns: {csv_path}")

    id_col = df.columns[0]
    X_df = df.drop(columns=[id_col])
    X = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    m_before = int(np.isnan(X).sum())

    imputer = build_imputer(method, seed=imp, epochs=epochs)
    X_imp = imputer.fit_transform(X)

    m_after = int(np.isnan(X_imp).sum())

    out_df = pd.DataFrame(X_imp, columns=X_df.columns)
    out_df.insert(0, id_col, df[id_col])

    ensure_dir(out_path.parent)
    write_df(out_path, out_df)
    return m_before, m_after


# Make each Ray task explicitly consume 1 CPU to avoid runaway parallelism on small machines.
@ray.remote(num_cpus=1)
def run_one_job(csv_path_str: str, out_path_str: str, method: str, imp: int, epochs: int | None):
    """Ray remote wrapper for process_single_file."""
    m_b, m_a = process_single_file(Path(csv_path_str), method, imp, Path(out_path_str), epochs)
    return {"m_before": m_b, "m_after": m_a}


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # MODE 1: NEXTFLOW / WORKER MODE (Single File)
    # ---------------------------------------------------------
    if args.file:
        if not args.method or args.seed is None:
            raise ValueError("--file mode requires --method and --seed (used as imputation index)")

        csv_path = Path(args.file)

        out_dir = Path(args.out_dir) if args.out_dir else Path(".")
        imp = int(args.seed)

        out_name = make_output_name(csv_path, imp)
        out_path = out_dir / out_name

        print(f"Worker Task: {csv_path.name} | Method: {args.method} | Imp: {imp} | Epochs: {args.epochs}")
        m_b, m_a = process_single_file(csv_path, args.method, imp=imp, out_path=out_path, epochs=args.epochs)
        print(f"Success. Missing: {m_b} -> {m_a}")
        return

    # ---------------------------------------------------------
    # MODE 2: RAY / BULK MODE
    # ---------------------------------------------------------
    cfg: dict = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # Hierarchy: CLI Flag > YAML Config > Default
    input_dirs_raw = args.input_dirs or cfg.get("input_dirs", [])
    input_dirs = [Path(d) for d in input_dirs_raw]

    output_root = Path(args.output_root or cfg.get("output_root", "outputs"))

    # Per-method number of imputations (multiple imputation count)
    imp_settings: dict[str, int] = cfg.get(
        "imputation_settings",
        {"iterative_rf": 1, "iterative_bayesridge": 1, "autoencoder": 1},
    )

    # Global epochs override
    epochs = args.epochs or cfg.get("epochs")

    if not input_dirs:
        raise ValueError("No input directories found. Use --input-dirs or --config.")

    valid_methods = {"iterative_rf", "iterative_bayesridge", "autoencoder"}

    print(f"Ray Mode initialized. Processing {len(input_dirs)} directories.")
    ray.init(ignore_reinit_error=True)

    jobs: list[tuple[str, str, str, int, int | None]] = []
    for d in input_dirs:
        if not d.exists():
            print(f"Warning: Directory {d} not found. Skipping.")
            continue

        csv_files = list_csv_files(d)
        for csv_path in csv_files:
            for method, n_imps in imp_settings.items():
                if method not in valid_methods:
                    print(f"Warning: unknown method '{method}' in imputation_settings. Skipping.")
                    continue

                for imp in range(1, int(n_imps) + 1):
                    out_name = make_output_name(csv_path, imp)

                    # output_root / <input_dir_name> / <method> / <file_impK_imputed.csv>
                    out_path = output_root / d.name / method / out_name

                    jobs.append((str(csv_path), str(out_path), method, imp, epochs))

    print(f"Submitting {len(jobs)} jobs to Ray cluster...")

    console = Console(force_terminal=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Imputing...", total=len(jobs))

        futures = [run_one_job.remote(*j) for j in jobs]
        remaining = futures

        total_m_b = 0
        total_m_a = 0

        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            result = ray.get(done[0])
            total_m_b += result["m_before"]
            total_m_a += result["m_after"]
            progress.advance(task)

    print("\nBulk run complete.")
    print(f"Total Missing Values: {total_m_b} -> {total_m_a}")
    ray.shutdown()


if __name__ == "__main__":
    main()
