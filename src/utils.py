# src/utils.py
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_corpus_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize common header variants → required columns
    df = df.rename(columns={
        "ID":"id","Id":"id",
        "Headline":"title","headline":"title",
        "Publisher":"source","publisher":"source",
        "Text":"content","text":"content"
    })
    for c in ["id","title","source","category","date","content"]:
        if c not in df.columns:
            df[c] = ""
    df["id"] = df["id"].astype(str)
    return df

def compact_text(title: str, content: str) -> str:
    return re.sub(r"\s+", " ", f"{title or ''}. {content or ''}").strip()
