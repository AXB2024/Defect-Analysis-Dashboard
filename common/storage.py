"""Object-store abstraction so every component runs against S3 or a local directory.

All I/O goes through boto3 (not pyarrow's native S3 filesystem) so that moto can
mock it in tests and the Lambda needs no extra filesystem dependencies.

URIs:
    s3://bucket[/prefix]   -> S3Store
    local path / file://   -> LocalStore
"""

from __future__ import annotations

import gzip
import io
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


class Store(ABC):
    @abstractmethod
    def get_bytes(self, key: str) -> bytes | None:
        """Return object bytes, or None if the key does not exist."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @property
    @abstractmethod
    def uri(self) -> str: ...

    # JSON helpers
    def get_json(self, key: str) -> Any | None:
        raw = self.get_bytes(key)
        if raw is None:
            return None
        if key.endswith(".gz"):
            raw = gzip.decompress(raw)
        return json.loads(raw)

    def put_json(self, key: str, obj: Any) -> None:
        body = json.dumps(obj, default=str, separators=(",", ":")).encode()
        if key.endswith(".gz"):
            self.put_bytes(key, gzip.compress(body), "application/gzip")
        else:
            self.put_bytes(key, body, "application/json")

    # Parquet helpers
    def get_parquet(self, key: str) -> pd.DataFrame | None:
        raw = self.get_bytes(key)
        if raw is None:
            return None
        return pd.read_parquet(io.BytesIO(raw))

    def put_parquet(self, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, compression="snappy")
        self.put_bytes(key, buf.getvalue(), "application/vnd.apache.parquet")

    def read_parquet_prefix(self, prefix: str) -> pd.DataFrame:
        keys = [k for k in self.list_keys(prefix) if k.endswith(".parquet")]
        frames = [self.get_parquet(k) for k in sorted(keys)]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


class LocalStore(Store):
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def uri(self) -> str:
        return str(self.root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def get_bytes(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # atomic on POSIX, mirrors S3 PUT semantics

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        search_root = base if base.is_dir() else base.parent
        if not search_root.exists():
            return []
        out = []
        for p in search_root.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                key = p.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


class S3Store(Store):
    def __init__(self, bucket: str, prefix: str = "", client=None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = client or boto3.client("s3")

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}".rstrip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _strip(self, full: str) -> str:
        return full[len(self.prefix) + 1 :] if self.prefix else full

    def get_bytes(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type)

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            out.extend(self._strip(o["Key"]) for o in page.get("Contents", []))
        return sorted(out)

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self._key(key))


def open_store(uri: str | None = None) -> Store:
    """Open a store from a URI, defaulting to $DATA_URI then ./data."""
    uri = uri or os.environ.get("DATA_URI") or "data"
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        return S3Store(bucket, prefix)
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return LocalStore(uri)


def iter_chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
