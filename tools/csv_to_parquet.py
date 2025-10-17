from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


def eprint(*a: object) -> None:
    s = " ".join(str(x) for x in a) + "\n"
    enc = sys.stderr.encoding or "utf-8"
    sys.stderr.buffer.write(s.encode(enc, errors="replace"))
    sys.stderr.flush()


def detect_encoding(path: str, user_encoding: Optional[str]) -> str:
    if user_encoding:
        return user_encoding
    # Essai rapide: on lit un petit blob bytes et on essaie de décoder
    try:
        blob = Path(path).read_bytes()[:200_000]
    except Exception as e:
        eprint(f"[fatal] impossible d'ouvrir {path}: {e}")
        raise
    for enc in CANDIDATE_ENCODINGS:
        try:
            _ = blob.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"


def to_parquet_stream(
    src: str,
    dst: str,
    sep: str = ";",
    encoding: Optional[str] = None,
    chunksize: int = 200_000,
) -> int:
    enc = detect_encoding(src, encoding)
    eprint(f"[info] using encoding={enc}, sep='{sep}', chunksize={chunksize}")

    out_path = Path(dst)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Construire kwargs sans low_memory (non supporté par engine='python')
    read_kwargs = dict(
        sep=sep,
        encoding=enc,
        engine="python",
        chunksize=chunksize,
        on_bad_lines="skip",   # ignore les lignes invalides
        dtype=None,
    )

    # Lecture en chunks
    try:
        reader = pd.read_csv(src, **read_kwargs)
    except Exception as e:
        eprint(f"[fatal] pandas.read_csv a échoué: {e}")
        return 1

    first = True
    written_rows = 0
    try:
        for i, chunk in enumerate(reader, start=1):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            pq.write_table(
                table,
                out_path,
                compression="snappy",
                use_dictionary=True,
                append=not first,
            )
            first = False
            written_rows += len(chunk)
            if i % 5 == 0:
                eprint(f"[info] wrote {written_rows} rows so far…")
    except Exception as e:
        eprint(f"[fatal] échec pendant la conversion: {e}")
        return 1

    if written_rows == 0:
        eprint("[warn] 0 lignes écrites (CSV vide ou entièrement illisible?)")
        return 2

    print(f"Parquet écrit -> {dst} (rows={written_rows})")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sep", default=";")
    ap.add_argument("--encoding", default=None, help="Force un encodage (sinon auto)")
    ap.add_argument("--chunksize", type=int, default=200_000)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    return to_parquet_stream(args.src, args.dst, args.sep, args.encoding, args.chunksize)


if __name__ == "__main__":
    raise SystemExit(main())

