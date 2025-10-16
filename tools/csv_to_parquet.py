# path: tools/csv_to_parquet.py
#!/usr/bin/env python3
"""
CSV (;) -> Parquet en streaming, faible RAM.
"""
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sep", default=";")
    ap.add_argument("--batch-rows", type=int, default=200_000)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    read_opts = pv.ReadOptions(block_size=1<<20)  # 1MB blocks
    parse_opts = pv.ParseOptions(delimiter=args.sep)
    convert_opts = pv.ConvertOptions(auto_dict_encode=True)

    reader = pv.open_csv(
        src,
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    )

    writer = None
    try:
        for batch in reader.read_next_batches():
            if writer is None:
                writer = pq.ParquetWriter(dst, schema=batch.schema, compression="snappy")
            table = pa.Table.from_batches([batch])
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    print(f"✔ Written parquet: {dst}")

if __name__ == "__main__":
    main()

