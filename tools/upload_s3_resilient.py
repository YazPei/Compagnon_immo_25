#!/usr/bin/env python3
# path: tools/upload_s3_resilient.py
"""
Resilient S3/S3-compatible uploader that avoids UploadPart 500s.
Treats empty --endpoint-url as None to prevent 'Invalid endpoint' errors.
"""

import argparse
import os
import sys
import time
from typing import Optional

import boto3
from boto3.s3.transfer import TransferConfig, S3Transfer
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from boto3.exceptions import S3UploadFailedError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resilient S3 uploader")
    p.add_argument("file", help="Local file path")
    p.add_argument("bucket", help="Bucket name")
    p.add_argument("key", help="Object key (remote path)")
    p.add_argument("--endpoint-url", default=os.getenv("AWS_S3_ENDPOINT"),
                   help="Custom S3 endpoint (omit for AWS S3)")
    p.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--access-key", default=os.getenv("AWS_ACCESS_KEY_ID"))
    p.add_argument("--secret-key", default=os.getenv("AWS_SECRET_ACCESS_KEY"))
    p.add_argument("--session-token", default=os.getenv("AWS_SESSION_TOKEN"))
    p.add_argument("--force-single", action="store_true",
                   help="Force single-part upload (avoids UploadPart entirely)")
    p.add_argument("--chunk-size-mb", type=int, default=8,
                   help="Multipart chunk size in MB (when not forcing single)")
    p.add_argument("--multipart-threshold-mb", type=int, default=8,
                   help="Multipart threshold in MB (when not forcing single)")
    p.add_argument("--max-concurrency", type=int, default=4,
                   help="Max threads for transfer (lower for flaky endpoints)")
    p.add_argument("--path-style", action="store_true",
                   help="Use path-style addressing (S3-compatible services)")
    p.add_argument("--content-type", default=None, help="Optional content-type")
    p.add_argument("--retries", type=int, default=6, help="Max upload retries")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_client(
    region: str,
    endpoint_url: Optional[str],
    access_key: Optional[str],
    secret_key: Optional[str],
    session_token: Optional[str],
    path_style: bool,
):
    cfg = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "adaptive"},
        signature_version="s3v4",
        s3={"addressing_style": "path" if path_style else "virtual"},
        connect_timeout=10,
        read_timeout=120,
    )
    session = boto3.session.Session()
    # IMPORTANT: coerce empty string to None (prevents 'Invalid endpoint' ValueError)
    endpoint_url = endpoint_url or None
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        config=cfg,
    )


def upload_with_retry(
    client,
    local_path: str,
    bucket: str,
    key: str,
    force_single: bool,
    chunk_mb: int,
    threshold_mb: int,
    max_conc: int,
    content_type: Optional[str],
    retries: int,
    verbose: bool,
) -> None:
    exceptions = (S3UploadFailedError, ClientError, EndpointConnectionError, ReadTimeoutError, TimeoutError, OSError)

    def _make_transfer(single: bool) -> S3Transfer:
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

    transfer = _make_transfer(force_single)
    attempt = 0
    backoff = 1.5
    last_exc: Optional[Exception] = None
    single_fallback_done = force_single

    while attempt <= retries:
        try:
            extra_args = {"ContentType": content_type} if content_type else None
            transfer.upload_file(local_path, bucket, key, extra_args=extra_args)
            if verbose:
                print(f"Upload succeeded (attempt {attempt+1}).")
            return
        except exceptions as e:
            last_exc = e
            attempt += 1
            if verbose:
                print(f"Attempt {attempt} failed: {repr(e)}")
            if attempt > retries:
                break

            status = None
            if isinstance(e, ClientError):
                status = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)

            if (status and 500 <= status < 600) and not single_fallback_done:
                if verbose:
                    print("Falling back to single-part upload to avoid UploadPart failures.")
                transfer = _make_transfer(True)
                single_fallback_done = True

            time.sleep(min(30.0, backoff ** attempt))

    raise RuntimeError(f"Upload failed after {retries} retries: {last_exc}") from last_exc


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}", file=sys.stderr)
        return 2

    client = make_client(
        region=args.region,
        endpoint_url=(args.endpoint_url or None),  # also coerce here for safety
        access_key=args.access_key,
        secret_key=args.secret_key,
        session_token=args.session_token,
        path_style=args.path_style,
    )

    if args.verbose:
        print("=== Config ===")
        print(f"Endpoint URL: {args.endpoint_url or 'AWS S3 default'}")
        print(f"Region: {args.region}")
        print(f"Path-style: {args.path_style}")
        print(f"Force single-part: {args.force_single}")
        print(f"Chunk size MB: {args.chunk_size_mb}, Threshold MB: {args.multipart_threshold_mb}")
        print(f"Max concurrency: {args.max_concurrency}")
        print(f"Uploading: {args.file} -> s3://{args.bucket}/{args.key}")
        print("================")

    try:
        upload_with_retry(
            client=client,
            local_path=args.file,
            bucket=args.bucket,
            key=args.key,
            force_single=args.force_single,
            chunk_mb=args.chunk_size_mb,
            threshold_mb=args.multipart_threshold_mb,
            max_conc=args.max_concurrency,
            content_type=args.content_type,
            retries=args.retries,
            verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

