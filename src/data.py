from __future__ import annotations
import numpy as np
import pandas as pd

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str)
    required = {"concurso", "data", "dezenas"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    df["concurso"] = pd.to_numeric(df["concurso"], errors="raise").astype(int)
    df = df.sort_values("concurso").drop_duplicates("concurso").reset_index(drop=True)
    parsed = []
    for raw in df["dezenas"].astype(str):
        nums = [int(x) for x in raw.replace(",", " ").split()]
        if len(nums) != 20 or len(set(nums)) != 20 or not all(0 <= x <= 99 for x in nums):
            raise ValueError(f"Resultado inválido: {raw}")
        parsed.append(sorted(nums))
    df["numbers"] = parsed
    return df

def incidence(df: pd.DataFrame) -> np.ndarray:
    x = np.zeros((len(df), 100), dtype=np.int8)
    for i, nums in enumerate(df["numbers"]):
        x[i, nums] = 1
    return x
