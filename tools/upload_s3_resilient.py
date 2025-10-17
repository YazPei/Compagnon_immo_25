import argparse, os, sys, time, tempfile, csv, re, mimetypes
from typing import Optional
import boto3
from boto3.s3.transfer import TransferConfig, S3Transfer
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from boto3.exceptions import S3UploadFailedError

_NUMERIC_FR = re.compile(r"^\s*[+-]?\d{1,3}(\s?\d{3})*(,\d+)?\s*$")
def _is_numeric_fr(v: str) -> bool: return bool(_NUMERIC_FR.match(v.strip()))
def _fr_to_dot(v: str) -> str:
    s = v.strip().replace("\u00A0","").replace(" ","")
    return s.replace(",", ".") if _is_numeric_fr(s) else v
def _sniff_delimiter(sample: str) -> Optional[str]:
    try: return csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"]).delimiter
    except Exception: return None

def convert_to_tmp(path: str, to: str, decimal_fr: bool, encoding: str) -> str:
    out_delim = "," if to == "csv" else "\t"
    with open(path, "r", encoding=encoding, newline="") as fin:
        sample = fin.read(64_000); fin.seek(0)
        from_delim = _sniff_delimiter(sample) or ";"
        reader = csv.reader(fin, delimiter=from_delim, quotechar='"', doublequote=True)
        fd, tmp_path = tempfile.mkstemp(suffix=f".{to}"); os.close(fd)
        with open(tmp_path, "w", encoding=encoding, newline="") as fout:
            w = csv.writer(fout, delimiter=out_delim, quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            for row in reader:
                if decimal_fr: row = [_fr_to_dot(x) for x in row]
                w.writerow(row)
    return tmp_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resilient S3 uploader")
    p.add_argument("file"); p.add_argument("bucket"); p.add_argument("key")
    p.add_argument("--endpoint-url", default=os.getenv("AWS_S3_ENDPOINT"))
    p.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--access-key", default=os.getenv("AWS_ACCESS_KEY_ID"))
    p.add_argument("--secret-key", default=os.getenv("AWS_SECRET_ACCESS_KEY"))
    p.add_argument("--session-token", default=os.getenv("AWS_SESSION_TOKEN"))
    p.add_argument("--force-single", action="store_true")
    p.add_argument("--chunk-size-mb", type=int, default=8)
    p.add_argument("--multipart-threshold-mb", type=int, default=8)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--path-style", action="store_true")
    p.add_argument("--content-type", default=None)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--verbose", action="store_true")
    # Normalisation
    p.add_argument("--normalize-csv", action="store_true")
    p.add_argument("--to", choices=["csv","tsv"], default="csv")
    p.add_argument("--decimal-fr", action="store_true")
    p.add_argument("--encoding", default="utf-8")
    return p.parse_args()

def make_client(region: str, endpoint_url: Optional[str], access_key: Optional[str], secret_key: Optional[str],
                session_token: Optional[str], path_style: bool):
    cfg = Config(
        region_name=region, retries={"max_attempts": 10, "mode": "adaptive"},
        signature_version="s3v4", s3={"addressing_style": "path" if path_style else "virtual"},
        connect_timeout=10, read_timeout=120,
    )
    session = boto3.session.Session()
    return session.client("s3",
        endpoint_url=(endpoint_url or None), region_name=region,
        aws_access_key_id=access_key, aws_secret_access_key=secret_key, aws_session_token=session_token, config=cfg)

def upload_with_retry(client, local_path: str, bucket: str, key: str, force_single: bool, chunk_mb: int,
                      threshold_mb: int, max_conc: int, content_type: Optional[str], retries: int, verbose: bool) -> None:
    exceptions = (S3UploadFailedError, ClientError, EndpointConnectionError, ReadTimeoutError, TimeoutError, OSError)
    def _make_transfer(single: bool) -> S3Transfer:
        if single:
            cfg = TransferConfig(multipart_threshold=64 * 1024**2 * 1024, max_concurrency=1)
        else:
            cfg = TransferConfig(multipart_threshold=threshold_mb * 1024**2, multipart_chunksize=chunk_mb * 1024**2,
                                 max_concurrency=max_conc, use_threads=True)
        return S3Transfer(client, config=cfg)
    transfer = _make_transfer(force_single)
    attempt, backoff, last_exc, single_done = 0, 1.5, None, force_single
    while attempt <= retries:
        try:
            extra_args = {"ContentType": content_type} if content_type else None
            transfer.upload_file(local_path, bucket, key, extra_args=extra_args)
            if verbose: print(f"Upload succeeded (attempt {attempt+1})."); return
        except exceptions as e:
            last_exc = e; attempt += 1
            if verbose: print(f"Attempt {attempt} failed: {repr(e)}")
            if attempt > retries: break
            status = int(getattr(e, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0) if isinstance(e, ClientError) else None
            if (status and 500 <= status < 600) and not single_done:
                if verbose: print("Falling back to single-part upload."); transfer = _make_transfer(True); single_done = True
            time.sleep(min(30.0, backoff ** attempt))
    raise RuntimeError(f"Upload failed after {retries} retries: {last_exc}") from last_exc

def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}", file=sys.stderr); return 2

    path_for_upload = args.file
    tmp: Optional[str] = None
    content_type = args.content_type or mimetypes.guess_type(args.file)[0]

    # WHY: Normaliser pour que DagsHub prévisualise correctement (évite les ';' affichés).
    if args.normalize_csv and pathlib.Path(args.file).suffix.lower() in {".csv",".tsv",".txt"}:
        tmp = convert_to_tmp(args.file, to=args.to, decimal_fr=args.decimal_fr, encoding=args.encoding)
        path_for_upload = tmp
        content_type = "text/csv" if args.to == "csv" else "text/tab-separated-values"
        # Si on convertit vers TSV mais key termine en .csv → harmoniser
        key = args.key[:-4] + ".tsv" if (args.to == "tsv" and args.key.lower().endswith(".csv")) else args.key
    else:
        key = args.key

    client = make_client(args.region, args.endpoint_url, args.access_key, args.secret_key, args.session_token, args.path_style)
    try:
        if args.verbose:
            print("=== Config ===")
            print(f"Endpoint URL: {args.endpoint_url or 'AWS S3 default'}")
            print(f"Uploading: {path_for_upload} -> s3://{args.bucket}/{key}")
            print(f"Content-Type: {content_type or 'auto'}"); print("================")
        upload_with_retry(
            client=client, local_path=path_for_upload, bucket=args.bucket, key=key,
            force_single=args.force_single, chunk_mb=args.chunk_size_mb, threshold_mb=args.multipart_threshold_mb,
            max_conc=args.max_concurrency, content_type=content_type, retries=args.retries, verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1
    finally:
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

if __name__ == "__main__":
    raise SystemExit(main())
