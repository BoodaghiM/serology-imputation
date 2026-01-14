from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def list_csv_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])


def read_csv_numeric(path: Path) -> pd.DataFrame:
    """
    Reads CSV and returns a numeric dataframe.
    Assumes the first column could be an ID column; if it's non-numeric, keep it separate in run.py.
    """
    return pd.read_csv(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_df(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

