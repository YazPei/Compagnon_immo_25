#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, sys, time, mimetypes, pathlib, tempfile, csv, re, shutil
from typing import Optional, List, Tuple
import boto3
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

# ---------- Utils: env ----------
def _env(name: str, default: Optional[str]=None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in ("", None) else default

# ---------- CSV normalize ----------
_NUMERIC_FR = re.compile(r"^\s*[+-]?\d{1,3}(\s?\d{3})*(,\d+)?\s*$")

def _is_numeric_fr(v: str) -> bool:
    return bool(_NUMERIC_FR.match(v.strip()))

def _fr_to_dot(v: str) -> str:
    s = v.strip().replace("\u00A0", "").replace(" ", "")
    return s.replace(",", ".") if _is_numeric_fr(s) else v

def _sniff_delimiter(sample: str) -> Optional[str]:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"]).delimiter
    except Exception:
        return None

def convert_delimiter_to_tmp(
    in_path: str,
    *,
    to: str,                 # "csv" or "tsv"
    decimal_fr: bool,
    encoding: str,
    from_delim: Optional[str] = None,
) -> Tuple[str, str]:
    """Retourne (tmp_path, content_type). Fichier écrit avec \n LF."""
    out_delim = "," if to == "csv" else "\t"
    ctype = "text/csv" if to == "csv" else "text/tab-separated-values"

    with open(in_path, "r", encoding=encoding, newline="") as fin:
        if from_delim is None:
            sample = fin.read(64_000); fin.seek(0)
            from_delim = _sniff_delimiter(sample) or ";"
        reader = csv.reader(fin, delimiter=from_delim, quotechar='"', doublequote=True, skipinitialspace=False)
        fd, tmp_path = tempfile.mkstemp(suffix=f".{to}")
        os.close(fd)
        with open(tmp_path, "w", encoding=encoding, newline="") as fout:
            writer = csv.writer(fout, delimiter=out_delim, quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            for row in reader:
                if decimal_fr:
                    row = [_fr_to_dot(x) for x in row]
                writer.writerow(row)
    return tmp_path, ctype

def should_normalize(path: str) -> bool:
    ext = pathlib.Path(path).suffix.lower()
    return ext in {".csv", ".tsv", ".txt"}  # safe: cible fichiers tabulaires

# ---------- S3 client ----------
def make_client(path_style: bool=True, no_verify_ssl: bool=False):
    endpoint = _env("AWS_S3_ENDPOINT")
    region = _env("AWS_DEFAULT_REGION", "us-east-1")
    ak = _env("AWS_ACCESS_KEY_ID")
    sk = _env("AWS_SECRET_ACCESS_KEY")
    st = _env("AWS_SESSION_TOKEN")
    if not ak or not sk:
        print("ERROR: Missing AWS_ACCESS_KEY_ID/SECRET_ACCESS_KEY.", file=sys.stderr); sys.exit(2)
    cfg = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "adaptive"},
        signature_version="s3v4",
        s3={"addressing_style": "path" if path_style else "virtual"},
        connect_timeout=10,
        read_timeout=120,
    )
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=(endpoint or None),
        region_name=region,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=st,
        config=cfg,
        verify=(False if no_verify_ssl else True),
    )

def bucket_name() -> str:
    b = _env("DAGSHUB_BUCKET")
    if not b:
        print("ERROR: Missing DAGSHUB_BUCKET.", file=sys.stderr); sys.exit(2)
    return b

# ---------- Upload with retry ----------
def upload_with_retry(
    client, local_path: str, bucket: str, key: str, *,
    force_single: bool, chunk_mb: int, threshold_mb: int, max_conc: int,
    content_type: Optional[str], retries: int, verbose: bool
) -> None:
    excs = (S3UploadFailedError, ClientError, EndpointConnectionError, ReadTimeoutError, TimeoutError, OSError)

    def _xfer(single: bool) -> S3Transfer:
        if single:
            cfg = TransferConfig(multipart_threshold=64 * 1024**2 * 1024, max_concurrency=1)  # énorme seuil => single
        else:
            cfg = TransferConfig(
                multipart_threshold=threshold_mb * 1024**2,
                multipart_chunksize=chunk_mb * 1024**2,
                max_concurrency=max_conc,
                use_threads=True,
            )
        return S3Transfer(client, config=cfg)

    xfer = _xfer(force_single)
    attempt, backoff, last, fell_back = 0, 1.5, None, force_single
    while attempt <= retries:
        try:
            extra = {"ContentType": content_type} if content_type else None
            xfer.upload_file(local_path, bucket, key, extra_args=extra)
            if verbose: print(f"Upload OK (attempt {attempt+1})")
            return
        except excs as e:
            last = e; attempt += 1
            if verbose: print(f"Attempt {attempt} failed: {repr(e)}")
            if attempt > retries: break
            status = int(getattr(e, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0) if isinstance(e, ClientError) else None
            if (status and 500 <= status < 600) and not fell_back:
                if verbose: print("Fallback to single-part to bypass UploadPart 5xx.")
                xfer = _xfer(True); fell_back = True
            time.sleep(min(30.0, backoff ** attempt))
    raise RuntimeError(f"Upload failed after {retries} retries: {last}") from last

# ---------- Ops ----------
def cmd_sanity(args):
    s3 = make_client(); b = bucket_name()
    resp = s3.list_objects_v2(Bucket=b, MaxKeys=1)
    print("Sanity OK. KeyCount:", resp.get("KeyCount", 0))

def _maybe_normalize(local_path: str, args, key_hint: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Retourne (path_to_upload, content_type, tmp_path_for_cleanup)
    - Ne convertit que si --normalize-csv et fichier tabulaire.
    - Si --to=tsv, renomme la clé distante (.tsv) si key_hint semble .csv.
    """
    if not args.normalize_csv or not should_normalize(local_path):
        return local_path, (args.content_type or mimetypes.guess_type(local_path)[0]), None

    target_fmt = args.to
    tmp_path, ctype = convert_delimiter_to_tmp(
        local_path, to=target_fmt, decimal_fr=args.decimal_fr, encoding=args.encoding
    )
    return tmp_path, ctype, tmp_path  # tmp à nettoyer après upload

def cmd_upload(args):
    s3 = make_client(); b = bucket_name()
    f = args.file
    if not os.path.isfile(f):
        print(f"File not found: {f}", file=sys.stderr); sys.exit(2)

    upload_path, ctype, tmp = _maybe_normalize(f, args, args.key)
    key = args.key
    # Si conversion vers TSV et clé finit en .csv → corriger l’extension distante
    if args.normalize_csv and args.to == "tsv" and key.lower().endswith(".csv"):
        key = key[:-4] + ".tsv"

    try:
        upload_with_retry(
            s3, upload_path, b, key,
            force_single=args.force_single,
            chunk_mb=args.chunk_size_mb,
            threshold_mb=args.multipart_threshold_mb,
            max_conc=args.max_concurrency,
            content_type=(args.content_type or ctype),
            retries=args.retries,
            verbose=args.verbose,
        )
    finally:
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def cmd_list(args):
    s3 = make_client(); b = bucket_name()
    resp = s3.list_objects_v2(Bucket=b, Prefix=(args.prefix or ""), MaxKeys=min(args.max_keys, 1000))
    for o in resp.get("Contents", []) or []:
        print(o["Key"])

def _ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def cmd_download(args):
    s3 = make_client(); b = bucket_name()
    dest_dir = args.out; _ensure_dir(dest_dir)
    dest = os.path.join(dest_dir, os.path.basename(args.key))
    s3.download_file(b, args.key, dest); print(f"Downloaded -> {dest}")

def cmd_sync_up(args):
    s3 = make_client(); b = bucket_name()
    root = pathlib.Path(args.dir)
    if not root.is_dir():
        print(f"Local dir not found: {root}", file=sys.stderr); sys.exit(2)
    prefix = args.prefix or ""
    for p in root.rglob("*"):
        if not p.is_file(): continue
        rel = p.relative_to(root).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel

        upload_path, ctype, tmp = _maybe_normalize(str(p), args, key)
        # Harmoniser extension distante si TSV demandé
        up_key = key
        if args.normalize_csv and args.to == "tsv" and up_key.lower().endswith(".csv"):
            up_key = up_key[:-4] + ".tsv"

        try:
            upload_with_retry(
                s3, upload_path, b, up_key,
                force_single=args.force_single,
                chunk_mb=args.chunk_size_mb,
                threshold_mb=args.multipart_threshold_mb,
                max_conc=args.max_concurrency,
                content_type=(args.content_type or ctype or mimetypes.guess_type(up_key)[0]),
                retries=args.retries,
                verbose=args.verbose,
            )
            if args.verbose: print(f"Uploaded {p} -> s3://{b}/{up_key}")
        finally:
            if tmp and os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass

def cmd_sync_down(args):
    s3 = make_client(); b = bucket_name()
    prefix, outdir = (args.prefix or ""), args.out
    _ensure_dir(outdir)
    token = None
    while True:
        resp = s3.list_objects_v2(Bucket=b, Prefix=prefix, ContinuationToken=token, MaxKeys=1000)
        for o in resp.get("Contents", []) or []:
            key = o["Key"]
            rel = key[len(prefix):].lstrip("/") if prefix else key
            dest = os.path.join(outdir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(b, key, dest)
            if args.verbose: print(f"Downloaded s3://{b}/{key} -> {dest}")
        if not resp.get("IsTruncated"): break
        token = resp.get("NextContinuationToken")

def cmd_presign(args):
    s3 = make_client(); b = bucket_name()
    url = s3.generate_presigned_url(
        ClientMethod="get_object", Params={"Bucket": b, "Key": args.key}, ExpiresIn=int(args.expires)
    ); print(url)

def cmd_cat(args):
    s3 = make_client(); b = bucket_name()
    obj = s3.get_object(Bucket=b, Key=args.key)
    body = obj["Body"].read(1024 * 1024)
    try: text = body.decode("utf-8", errors="replace")
    except Exception: text = str(body)
    for line in text.splitlines()[: args.lines]:
        print(line)

def cmd_rm(args):
    s3 = make_client(); b = bucket_name()
    if args.key:
        s3.delete_object(Bucket=b, Key=args.key); print(f"Deleted s3://{b}/{args.key}"); return
    prefix = args.prefix or ""
    if not prefix:
        print("ERROR: --key ou --prefix requis.", file=sys.stderr); sys.exit(2)
    token, to_delete = None, []
    while True:
        resp = s3.list_objects_v2(Bucket=b, Prefix=prefix, ContinuationToken=token, MaxKeys=1000)
        for o in resp.get("Contents", []) or []:
            to_delete.append({"Key": o["Key"]})
            if len(to_delete) == 1000:
                s3.delete_objects(Bucket=b, Delete={"Objects": to_delete}); to_delete.clear()
        if not resp.get("IsTruncated"): break
        token = resp.get("NextContinuationToken")
    if to_delete: s3.delete_objects(Bucket=b, Delete={"Objects": to_delete})
    print(f"Deleted all under s3://{b}/{prefix}")

# ---------- CLI ----------
def _add_common_norm_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--normalize-csv", action="store_true", help="Convertit ; -> ',' (CSV) ou '\\t' (TSV) avant upload")
    p.add_argument("--to", choices=["csv","tsv"], default="csv", help="Format de sortie si normalisation")
    p.add_argument("--decimal-fr", action="store_true", help="Convertit 1,23 -> 1.23 sur champs numériques")
    p.add_argument("--encoding", default="utf-8", help="Encodage lecture/écriture (défaut: utf-8)")

def main():
    ap = argparse.ArgumentParser(prog="dagshub_s3", description="DagsHub S3 export/import CLI (+ CSV normalize)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sanity")

    up = sub.add_parser("upload")
    up.add_argument("--file", required=True)
    up.add_argument("--key", required=True)
    up.add_argument("--force-single", action="store_true")
    up.add_argument("--chunk-size-mb", type=int, default=8)
    up.add_argument("--multipart-threshold-mb", type=int, default=8)
    up.add_argument("--max-concurrency", type=int, default=4)
    up.add_argument("--content-type", default=None)
    up.add_argument("--retries", type=int, default=6)
    up.add_argument("--verbose", action="store_true")
    _add_common_norm_flags(up)

    ls = sub.add_parser("list")
    ls.add_argument("--prefix", default="")
    ls.add_argument("--max-keys", type=int, default=100)

    dl = sub.add_parser("download")
    dl.add_argument("--key", required=True)
    dl.add_argument("--out", required=True)

    su = sub.add_parser("sync-up")
    su.add_argument("--dir", required=True)
    su.add_argument("--prefix", default="")
    su.add_argument("--force-single", action="store_true")
    su.add_argument("--chunk-size-mb", type=int, default=8)
    su.add_argument("--multipart-threshold-mb", type=int, default=8)
    su.add_argument("--max-concurrency", type=int, default=4)
    su.add_argument("--retries", type=int, default=6)
    su.add_argument("--verbose", action="store_true")
    _add_common_norm_flags(su)

    sd = sub.add_parser("sync-down")
    sd.add_argument("--prefix", required=True)
    sd.add_argument("--out", required=True)
    sd.add_argument("--verbose", action="store_true")

    ps = sub.add_parser("presign")
    ps.add_argument("--key", required=True)
    ps.add_argument("--expires", type=int, default=3600)

    ct = sub.add_parser("cat")
    ct.add_argument("--key", required=True)
    ct.add_argument("--lines", type=int, default=20)

    rm = sub.add_parser("rm")
    rm.add_argument("--key", default="")
    rm.add_argument("--prefix", default="")

    args = ap.parse_args()
    if args.cmd == "sanity": cmd_sanity(args)
    elif args.cmd == "upload": cmd_upload(args)
    elif args.cmd == "list": cmd_list(args)
    elif args.cmd == "download": cmd_download(args)
    elif args.cmd == "sync-up": cmd_sync_up(args)
    elif args.cmd == "sync-down": cmd_sync_down(args)
    elif args.cmd == "presign": cmd_presign(args)
    elif args.cmd == "cat": cmd_cat(args)
    elif args.cmd == "rm": cmd_rm(args)
    else: ap.error("unknown command")

if __name__ == "__main__":
    raise SystemExit(main())
