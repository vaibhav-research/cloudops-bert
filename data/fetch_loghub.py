#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch + preprocess CloudOpsBERT datasets (LogHub HDFS & BGL).

What it does
------------
- Downloads Zenodo zips (LogHub v8, ISSRE'23)
- Extracts archives
- For each dataset:
  HDFS:
    * If a structured CSV is found -> convert to JSONL (text,label).
    * Else if 'preprocessed' CSVs (templates+traces+labels) exist -> reconstruct text from event sequences.
  BGL:
    * If a structured CSV is found -> convert to JSONL.
    * Else if kaggle CLI & creds available -> downloads 'ayush2222/structured-bgl-logs-csv' and uses it.
- Normalizes labels to {0,1}, shuffles, splits into train/test
- Emits JSONL at data/processed/<dataset>/{train,test}.jsonl

Usage
-----
  python data/fetch_data.py --dataset all
  python data/fetch_data.py --dataset hdfs
  python data/fetch_data.py --dataset bgl
  python data/fetch_data.py --test-ratio 0.2 --seed 42 --outdir data/processed
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ----------------------------
# Config: Zenodo LogHub v8 (ISSRE'23)
# Record page: https://zenodo.org/records/8196385
# ----------------------------
DATASETS: Dict[str, Tuple[str, str]] = {
    # name: (URL, kind)
    "hdfs": ("https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1", "zip"),
    "bgl":  ("https://zenodo.org/records/8196385/files/BGL.zip?download=1", "zip"),
}

# Kaggle fallback for BGL structured CSV
KAGGLE_BGL_DS = "ayush2222/structured-bgl-logs-csv"   # contains BGL.log_structured.csv

# Default repo-relative paths
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "processed"


# ----------------------------
# IO helpers
# ----------------------------
def _stream_download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise SystemExit(
                f"[!] HTTP error while downloading:\nURL: {url}\n{e}\n"
                "Tip: Open the URL in a browser to confirm availability.\n"
            )
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def _find_structured_csv(extracted_dir: Path, dataset_name: str) -> Optional[Path]:
    """
    Heuristic search for a structured CSV inside the extracted dataset.
    Prefers files with 'structured' in the name; otherwise picks the largest CSV.
    """
    candidates = list(extracted_dir.rglob("*.csv"))
    print(f"[=] ({dataset_name}) CSV candidates under {extracted_dir}: {len(candidates)}")
    if candidates:
        for p in sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[:10]:
            print("    -", p.relative_to(extracted_dir))

    if not candidates:
        return None

    # Prefer names containing 'structured'
    structured = [p for p in candidates if "structured" in p.name.lower()]
    if structured:
        structured.sort(key=lambda p: p.stat().st_size, reverse=True)
        return structured[0]

    # Known names
    preferred = {
        "bgl.log_structured.csv",
        "hdfs_2k.log_structured.csv",
        "hdfs.log_structured.csv",
    }
    for p in candidates:
        if p.name.lower() in preferred:
            return p

    # Fallback: choose the largest CSV
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


# ----------------------------
# Label & text helpers
# ----------------------------
def _normalize_label_series(s: pd.Series) -> pd.Series:
    def map_one(x):
        if pd.isna(x):
            return 0
        try:
            v = int(float(x))
            return 1 if v != 0 else 0
        except Exception:
            pass
        if isinstance(x, bool):
            return 1 if x else 0
        xs = str(x).strip().lower()
        if xs in {"0", "normal", "ok", "benign", "healthy", "clean"}:
            return 0
        if xs in {"1", "anomaly", "abnormal", "error", "fault", "failure", "fail"}:
            return 1
        return 1 if any(k in xs for k in ["anomal", "error", "abnorm", "fault", "fail"]) else 0
    return s.apply(map_one).astype(int)


def _pick_text_col(df: pd.DataFrame) -> str:
    if "Content" in df.columns:
        return "Content"
    if "EventTemplate" in df.columns:
        return "EventTemplate"
    text_like = []
    for c in df.columns:
        if df[c].dtype == object:
            try:
                avg_len = df[c].astype(str).str.len().mean()
            except Exception:
                avg_len = 0
            text_like.append((avg_len, c))
    if text_like:
        text_like.sort(reverse=True)
        return text_like[0][1]
    return df.columns[0]


def _pick_label_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["Label", "Anomaly", "label", "y", "target"]:
        if c in df.columns:
            return c
    return None


def _split_train_test(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    df = df.iloc[idx].reset_index(drop=True)
    n_test = max(1, int(len(df) * test_ratio))
    test_df = df.iloc[:n_test].copy()
    train_df = df.iloc[n_test:].copy()
    return train_df, test_df


def _to_jsonl(df: pd.DataFrame, out_path: Path, text_col: str, label_col: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {"text": str(row[text_col]), "label": int(row[label_col])}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------
# CSV → JSONL (generic)
# ----------------------------
def preprocess_from_structured_csv(csv_path: Path, out_dir: Path, test_ratio: float, seed: int) -> None:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise SystemExit(f"[!] Empty CSV: {csv_path}")

    text_col = _pick_text_col(df)
    label_col = _pick_label_col(df)

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[~df[text_col].isna() & (df[text_col].str.len() > 0)].copy()

    if label_col is None:
        raise SystemExit(
            f"[!] No label column found in {csv_path.name}. "
            "Expected one of {Label, Anomaly, label, y, target}.\n"
            "Tip: Inspect the CSV columns and adjust _pick_label_col or add dataset-specific logic."
        )

    df[label_col] = _normalize_label_series(df[label_col])

    train_df, test_df = _split_train_test(df, test_ratio=test_ratio, seed=seed)
    _to_jsonl(train_df[[text_col, label_col]], out_dir / "train.jsonl", text_col, label_col)
    _to_jsonl(test_df[[text_col, label_col]], out_dir / "test.jsonl", text_col, label_col)
    print(f"[+] Wrote {len(train_df)} train / {len(test_df)} test to {out_dir}")


# ----------------------------
# HDFS (preprocessed trio → reconstruct text)
# ----------------------------
def _load_hdfs_templates(templates_csv: Path):
    df = pd.read_csv(templates_csv)
    # choose template col
    tpl_col = "EventTemplate" if "EventTemplate" in df.columns else _pick_text_col(df)
    # choose id col
    id_col = None
    for c in ["EventId", "EventID", "event_id", "ID", "Id", "eid"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        non_text = [c for c in df.columns if df[c].dtype != object]
        id_col = non_text[0] if non_text else df.columns[0]
    ids = df[id_col].astype(str).str.strip()
    # normalize to E<number>
    if ids.str.fullmatch(r"\d+").all():
        ids = "E" + ids
    ids = ids.str.replace(r".*?(\d+)$", r"E\1", regex=True).str.upper()
    return dict(zip(ids, df[tpl_col].astype(str)))


def _load_hdfs_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    blk_col = None
    for c in ["BlockId", "BlockID", "Block", "blk_id", "blockid", "block"]:
        if c in df.columns:
            blk_col = c
            break
    if blk_col is None:
        blk_col = df.columns[0]
    lbl_col = None
    for c in ["Label", "label", "Anomaly", "y", "target"]:
        if c in df.columns:
            lbl_col = c
            break
    if lbl_col is None:
        raise SystemExit("[!] Could not find label column in anomaly_label.csv")
    out = df[[blk_col, lbl_col]].rename(columns={blk_col: "BlockId", lbl_col: "label"})
    out["label"] = _normalize_label_series(out["label"])
    return out


def _detect_hdfs_trace_cols(traces_df: pd.DataFrame, id2tpl_keys):
    # block id
    blk_col = None
    for c in ["BlockId", "BlockID", "Block", "blk_id", "blockid", "block"]:
        if c in traces_df.columns:
            blk_col = c
            break
    if blk_col is None:
        blk_col = traces_df.columns[0]

    # choose sequence column that contains IDs matching templates
    obj_cols = [c for c in traces_df.columns if traces_df[c].dtype == object]
    candidates = obj_cols if obj_cols else list(traces_df.columns)
    keys = set(id2tpl_keys)

    def score_col(c):
        vals = traces_df[c].astype(str).head(50).tolist()
        hits = 0
        for v in vals:
            toks = re.findall(r"[A-Za-z]*\d+", v)  # E12, 23, id7
            toks = [("E" + t) if t.isdigit() else re.sub(r".*?(\d+)$", r"E\1", t.upper())
                    for t in toks]
            if any(t in keys for t in toks):
                hits += 1
        return hits

    best = max(candidates, key=score_col)
    return blk_col, best


def _parse_hdfs_sequence(cell: str, id2tpl: dict) -> str:
    s = str(cell).strip()
    if not s:
        return ""
    s = s.strip("[](){}")
    s = s.replace(",", " ").replace(";", " ")
    toks = re.findall(r"[A-Za-z]*\d+", s)  # E1, e23, 45, id7
    out = []
    for t in toks:
        key = ("E" + t) if t.isdigit() else re.sub(r".*?(\d+)$", r"E\1", t.upper())
        out.append(id2tpl.get(key, f"<{key}>"))
    return " ".join(out)


def preprocess_hdfs(extracted_dir: Path, out_dir: Path, test_ratio: float, seed: int) -> None:
    # 1) Try structured CSV first
    csv_path = _find_structured_csv(extracted_dir, "hdfs")
    if csv_path is not None and "preprocessed" not in str(csv_path.parent).lower():
        print(f"[=] Using HDFS structured CSV: {csv_path.relative_to(extracted_dir)}")
        preprocess_from_structured_csv(csv_path, out_dir, test_ratio, seed)
        return

    # 2) If not, check for preprocessed trio
    pre = extracted_dir / "preprocessed"
    tpl = pre / "HDFS.log_templates.csv"
    trc = pre / "Event_traces.csv"
    lbl = pre / "anomaly_label.csv"
    if all(p.exists() for p in [tpl, trc, lbl]):
        print("[=] Using HDFS preprocessed trio (templates+traces+labels)")
        id2tpl = _load_hdfs_templates(tpl)
        labels = _load_hdfs_labels(lbl)
        traces = pd.read_csv(trc)
        blk_col, seq_col = _detect_hdfs_trace_cols(traces, id2tpl.keys())
        print(f"[traces] chose sequence column: {seq_col}")

        df = traces[[blk_col, seq_col]].rename(columns={blk_col: "BlockId", seq_col: "EventSeq"})
        df = df.merge(labels, on="BlockId", how="inner")
        df["text"] = df["EventSeq"].apply(lambda x: _parse_hdfs_sequence(x, id2tpl))
        df = df[~df["text"].isna() & (df["text"].str.len() > 0)][["text", "label"]].copy()

        train_df, test_df = _split_train_test(df, test_ratio=test_ratio, seed=seed)
        _to_jsonl(train_df, out_dir / "train.jsonl", "text", "label")
        _to_jsonl(test_df, out_dir / "test.jsonl", "text", "label")
        print(f"[+] Wrote {len(train_df)} train / {len(test_df)} test to {out_dir}")
        return

    raise SystemExit(
        "[!] HDFS: neither structured CSV nor preprocessed trio found.\n"
        f"    Looked under: {extracted_dir}\n"
        "    Expected a '*structured*.csv' OR 'preprocessed/{HDFS.log_templates.csv, Event_traces.csv, anomaly_label.csv}'."
    )


# ----------------------------
# BGL (Kaggle fallback if no structured CSV)
# ----------------------------
def _have_kaggle_cli_and_creds() -> bool:
    if shutil.which("kaggle") is None:
        return False
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _download_bgl_from_kaggle(target_dir: Path) -> Optional[Path]:
    """Downloads structured BGL CSV via kaggle CLI. Returns CSV path or None."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print("[=] Attempting Kaggle download for BGL structured CSV…")
    try:
        # download zip
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_BGL_DS, "-p", str(target_dir)],
            check=True
        )
        # unzip
        zips = list(target_dir.glob("*.zip"))
        if not zips:
            print("[!] Kaggle download completed but no zip found.")
            return None
        for z in zips:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(target_dir)
        # look for BGL.log_structured.csv
        csv = next((p for p in target_dir.rglob("BGL.log_structured.csv")), None)
        if csv is None:
            # fallback: largest CSV
            csvs = list(target_dir.rglob("*.csv"))
            if csvs:
                csv = max(csvs, key=lambda p: p.stat().st_size)
        return csv
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        return None


def preprocess_bgl(extracted_dir: Path, raw_dataset_dir: Path, out_dir: Path, test_ratio: float, seed: int) -> None:
    csv_path = _find_structured_csv(extracted_dir, "bgl")
    if csv_path is not None:
        print(f"[=] Using BGL structured CSV: {csv_path.relative_to(extracted_dir)}")
        preprocess_from_structured_csv(csv_path, out_dir, test_ratio, seed)
        return

    # Try Kaggle fallback
    if _have_kaggle_cli_and_creds():
        kaggle_dir = raw_dataset_dir / "kaggle"
        csv_path = _download_bgl_from_kaggle(kaggle_dir)
        if csv_path and csv_path.exists():
            print(f"[=] Using Kaggle CSV: {csv_path}")
            preprocess_from_structured_csv(csv_path, out_dir, test_ratio, seed)
            return
        else:
            print("[!] Kaggle fallback did not yield a usable CSV.")

    raise SystemExit(
        "[!] BGL: no structured CSV found in the Zenodo archive, and Kaggle fallback not available.\n"
        "    To enable Kaggle fallback: install kaggle (`pip install kaggle`), place credentials at ~/.kaggle/kaggle.json,\n"
        f"    then re-run, or manually download {KAGGLE_BGL_DS} and point this script to its CSV."
    )


# ----------------------------
# Fetch dispatcher
# ----------------------------
def fetch_and_extract(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise SystemExit(f"[!] Unknown dataset: {dataset}. Options: {list(DATASETS.keys())}")

    url, kind = DATASETS[dataset]
    dataset_dir = RAW_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1].split("?")[0]
    zip_path = dataset_dir / filename

    if not zip_path.exists():
        print(f"[+] Downloading {dataset} from:\n    {url}")
        _stream_download(url, zip_path)
    else:
        print(f"[=] Already downloaded: {zip_path.name}")

    if kind == "zip":
        extract_dir = dataset_dir / "extracted"
        if not extract_dir.exists():
            print(f"[+] Extracting to: {extract_dir}")
            _extract_zip(zip_path, extract_dir)
        else:
            print(f"[=] Already extracted: {extract_dir}")
        return extract_dir

    raise SystemExit(f"[!] Unsupported archive kind for {dataset}: {kind}")


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch and preprocess CloudOpsBERT datasets (LogHub).")
    ap.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all",
                    help="Dataset to fetch/preprocess.")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (0-1).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Output directory for processed JSONL.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in targets:
        print(f"\n=== [{name.upper()}] ===")
        extracted = fetch_and_extract(name)
        out_dir = outdir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if name == "hdfs":
            preprocess_hdfs(extracted, out_dir, args.test_ratio, args.seed)
        elif name == "bgl":
            preprocess_bgl(extracted, RAW_DIR / "bgl", out_dir, args.test_ratio, args.seed)
        else:
            raise SystemExit(f"[!] Unknown dataset handler: {name}")

    print("\n[✓] Done. Processed datasets at:", outdir.resolve())
    for name in targets:
        base = outdir / name
        print(f"    - {base / 'train.jsonl'}")
        print(f"    - {base / 'test.jsonl'}")


if __name__ == "__main__":
    pd.options.display.max_colwidth = 200
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.", file=sys.stderr)
        sys.exit(130)
