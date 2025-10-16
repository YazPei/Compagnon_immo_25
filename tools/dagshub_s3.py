# path: tools/dagshub_s3.py
#!/usr/bin/env python3
"""
DagsHub S3 helper CLI: export/import sans douleur.
- Auth via ~/.dagshub.env (token = access key = secret).
- Single script: sanity, upload (single/multipart), list, download, sync-up/down, presign, cat, rm.
- Robust retries & fallback on 5xx for multipart.
"""
from __future__ import annotations
import argparse, os, sys, time, mimetypes, pathlib
from typing import Optional, Iterable, List, Tuple
import boto3
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

# ---------- Client ----------
def _env(name: str, default: Optional[str]=None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in ("", None) else default

def make_client(path_style: bool=True, no_verify_ssl: bool=False):
    endpoint = _env("AWS_S3_ENDPOINT")
    region = _env("AWS_DEFAULT_REGION", "us-east-1")
    ak = _env("AWS_ACCESS_KEY_ID")
    sk = _env("AWS_SECRET_ACCESS_KEY")
    st = _env("AWS_SESSION_TOKEN")
    if not ak or not sk:
        print("ERROR: Missing AWS_ACCESS_KEY_ID/SECRET_ACCESS_KEY (DagsHub token).", file=sys.stderr)
        sys.exit(2)
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
        endpoint_url=endpoint or None,
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
        print("ERROR: Missing DAGSHUB_BUCKET (repo name) in ~/.dagshub.env", file=sys.stderr)
        sys.exit(2)
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
            cfg = TransferConfig(multipart_threshold=64 * 1024**2 * 1024, max_concurrency=1)
        else:
            cfg = TransferConfig(
                multipart_threshold=threshold_mb * 1024**2,
                multipart_chunksize=chunk_mb * 1024**2,
                max_concurrency=max_conc,
                use_threads=True,
            )
        return S3Transfer(client, config=cfg)

    xfer = _xfer(force_single)
    attempt = 0
    backoff = 1.5
    last: Optional[Exception] = None
    fell_back = force_single
    while attempt <= retries:
        try:
            extra = {"ContentType": content_type} if content_type else None
            xfer.upload_file(local_path, bucket, key, extra_args=extra)
            if verbose:
                print(f"Upload OK (attempt {attempt+1})")
            return
        except excs as e:
            last = e
            attempt += 1
            if verbose:
                print(f"Attempt {attempt} failed: {repr(e)}")
            if attempt > retries:
                break
            status = None
            if isinstance(e, ClientError):
                status = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
            if (status and 500 <= status < 600) and not fell_back:
                if verbose:
                    print("Fallback to single-part to bypass UploadPart 5xx.")
                xfer = _xfer(True)
                fell_back = True
            time.sleep(min(30.0, backoff ** attempt))
    raise RuntimeError(f"Upload failed after {retries} retries: {last}") from last

# ---------- Ops ----------
def cmd_sanity(args):
    s3 = make_client()
    b = bucket_name()
    resp = s3.list_objects_v2(Bucket=b, MaxKeys=1)
    print("Sanity OK. KeyCount:", resp.get("KeyCount", 0))

def cmd_upload(args):
    s3 = make_client()
    b = bucket_name()
    f = args.file
    if not os.path.isfile(f):
        print(f"File not found: {f}", file=sys.stderr); sys.exit(2)
    key = args.key
    ctype = args.content_type or mimetypes.guess_type(f)[0]
    upload_with_retry(
        s3, f, b, key,
        force_single=args.force_single,
        chunk_mb=args.chunk_size_mb,
        threshold_mb=args.multipart_threshold_mb,
        max_conc=args.max_concurrency,
        content_type=ctype,
        retries=args.retries,
        verbose=args.verbose,
    )

def cmd_list(args):
    s3 = make_client()
    b = bucket_name()
    prefix = args.prefix or ""
    # Appel simple sans pagination (évite ContinuationToken=None)
    resp = s3.list_objects_v2(Bucket=b, Prefix=prefix, MaxKeys=min(args.max_keys, 1000))
    for o in resp.get("Contents", []) or []:
        print(o["Key"])

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cmd_download(args):
    s3 = make_client()
    b = bucket_name()
    key = args.key
    outdir = args.out
    _ensure_dir(outdir)
    dest = os.path.join(outdir, os.path.basename(key))
    s3.download_file(b, key, dest)
    print(f"Downloaded -> {dest}")

def cmd_sync_up(args):
    s3 = make_client()
    b = bucket_name()
    root = pathlib.Path(args.dir)
    if not root.is_dir():
        print(f"Local dir not found: {root}", file=sys.stderr); sys.exit(2)
    prefix = args.prefix or ""
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            ctype = mimetypes.guess_type(str(p))[0]
            upload_with_retry(
                s3, str(p), b, key,
                force_single=args.force_single,
                chunk_mb=args.chunk_size_mb,
                threshold_mb=args.multipart_threshold_mb,
                max_conc=args.max_concurrency,
                content_type=ctype,
                retries=args.retries,
                verbose=args.verbose,
            )
            if args.verbose:
                print(f"Uploaded {p} -> s3://{b}/{key}")

def cmd_sync_down(args):
    s3 = make_client()
    b = bucket_name()
    prefix = args.prefix or ""
    outdir = args.out
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
            if args.verbose:
                print(f"Downloaded s3://{b}/{key} -> {dest}")
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

def cmd_presign(args):
    s3 = make_client()
    b = bucket_name()
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": b, "Key": args.key},
        ExpiresIn=int(args.expires),
    )
    print(url)

def cmd_cat(args):
    s3 = make_client()
    b = bucket_name()
    obj = s3.get_object(Bucket=b, Key=args.key)
    body = obj["Body"].read(1024 * 1024)  # 1MB max affiché
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        text = str(body)
    lines = text.splitlines()
    for line in lines[: args.lines]:
        print(line)

def cmd_rm(args):
    s3 = make_client()
    b = bucket_name()
    # Supprime clé ou tous objets sous un préfixe
    if args.key:
        s3.delete_object(Bucket=b, Key=args.key)
        print(f"Deleted s3://{b}/{args.key}")
        return
    prefix = args.prefix or ""
    if not prefix:
        print("ERROR: --key ou --prefix requis.", file=sys.stderr); sys.exit(2)
    token = None
    to_delete = []
    while True:
        resp = s3.list_objects_v2(Bucket=b, Prefix=prefix, ContinuationToken=token, MaxKeys=1000)
        for o in resp.get("Contents", []) or []:
            to_delete.append({"Key": o["Key"]})
            if len(to_delete) == 1000:
                s3.delete_objects(Bucket=b, Delete={"Objects": to_delete})
                to_delete.clear()
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    if to_delete:
        s3.delete_objects(Bucket=b, Delete={"Objects": to_delete})
    print(f"Deleted all under s3://{b}/{prefix}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(prog="dagshub_s3", description="DagsHub S3 export/import CLI")
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
    else:
        ap.error("unknown command")

if __name__ == "__main__":
    raise SystemExit(main())

