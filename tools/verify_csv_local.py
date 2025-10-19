# path: tools/verify_csv_local.py
#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import pandas as pd

CANDS = ["utf-8","utf-8-sig","cp1252","latin-1"]
def sniff(p: Path):
    b = p.read_bytes()[:200_000]
    for enc in CANDS:
        try:
            b.decode(enc); return enc
        except: pass
    return "latin-1"

ap = argparse.ArgumentParser(); ap.add_argument("--path", required=True); ap.add_argument("--sep", default=";"); ap.add_argument("--rows", type=int, default=20)
a = ap.parse_args(); p = Path(a.path)
enc = sniff(p); print(f"encoding={enc}")
df = pd.read_csv(p, sep=a.sep, encoding=enc, engine="python", nrows=a.rows, on_bad_lines="skip")
with pd.option_context("display.max_columns", 50, "display.width", 200):
    print(df.head(a.rows).to_string(index=False))

