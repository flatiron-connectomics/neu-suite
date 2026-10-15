"""Destination/source locations: local paths, ``s3://``/``gs://`` URLs, kvstore dicts.

Everything downstream speaks TensorStore *kvstore specs* (plain dicts), so the ops
behave identically for local files and object stores. This module normalizes a
user-facing location into a kvstore and joins subpaths uniformly.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Mapping

_SCHEMES = {"s3://": "s3", "gs://": "gcs"}


def _resolve_local(path: str) -> str:
    """A filesystem path as tensorstore will accept it: absolute, with ``~`` expanded.

    tensorstore's file driver rejects relative paths, and an absolute path also means the
    same thing on a worker with a different working directory. A trailing slash is
    dropped; callers that need one append it.
    """
    if not path:
        return path        # a degenerate location, left alone rather than made the cwd
    return os.path.abspath(os.path.expanduser(path))


def to_kvstore(location: str | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a location to a TensorStore kvstore spec.

    Accepts a local path (made absolute, ``~`` expanded), an ``s3://bucket/prefix`` /
    ``gs://bucket/prefix`` URL, or an existing kvstore dict.
    """
    if isinstance(location, Mapping):
        kv = dict(location)
        if kv.get("driver") == "file":
            # A hand-written spec is as likely to carry a relative path as a string is.
            kv["path"] = _resolve_local(str(kv.get("path", "")))
        return kv
    s = str(location)
    for scheme, driver in _SCHEMES.items():
        if s.startswith(scheme):
            bucket, _, prefix = s[len(scheme):].partition("/")
            return {"driver": driver, "bucket": bucket, "path": prefix}
    return {"driver": "file", "path": _resolve_local(s)}


def join(kvstore: Mapping[str, Any], *parts: str) -> dict[str, Any]:
    """Append path segments to a kvstore's ``path`` (leading '/' kept for files)."""
    kv = dict(kvstore)
    base = str(kv.get("path", ""))
    absolute = kv.get("driver") == "file" and base.startswith("/")
    segs = [base.strip("/")] if base.strip("/") else []
    segs += [str(p).strip("/") for p in parts if str(p).strip("/")]
    joined = "/".join(segs)
    kv["path"] = ("/" + joined) if absolute else joined
    return kv


def as_directory(kvstore: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``kvstore`` whose ``path`` ends in ``/``, for a driver given it as a root."""
    kv = dict(kvstore)
    if "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    return kv


def spec_kvstore(spec: Mapping[str, Any]) -> dict[str, Any]:
    """kvstore for a backend spec (handles 'kvstore', 'path', and scheme URLs)."""
    if "kvstore" in spec:
        return dict(spec["kvstore"])
    if "path" in spec:
        return to_kvstore(spec["path"])
    raise ValueError("spec needs 'kvstore' or 'path'")


def is_local(location: str | Mapping[str, Any]) -> bool:
    """True if ``location`` resolves to an ordinary filesystem path, for callers that need
    POSIX semantics (sqlite, appended logs) to refuse an object store up front."""
    return to_kvstore(location).get("driver") == "file"


def local_path(location: str | Mapping[str, Any]) -> str:
    """The filesystem path for a local location; raises if it is remote."""
    kv = to_kvstore(location)
    if kv.get("driver") != "file":
        raise ValueError(
            f"{location!r} is not a local path (driver={kv.get('driver')!r}); "
            "this operation needs a filesystem")
    return str(kv.get("path", ""))


# --------------------------------------------------------------------------- #
# Byte / JSON I/O over a kvstore: one code path for local files and object stores.
# The file driver creates missing parent directories on write.
# --------------------------------------------------------------------------- #
def ensure_credentials(kvstore: Mapping[str, Any]) -> dict[str, Any]:
    """Bootstrap object-store credentials for a kvstore that is about to be opened.

    Call it wherever a store is opened, not where a spec is built: tensorstore's S3
    profile provider cannot read ``~/.aws/credentials``, so the profile is copied into
    ``AWS_*`` environment variables, per process. A dask worker that skipped it would get
    403s while others succeed.
    """
    kv = dict(kvstore)
    if kv.get("driver") == "s3":
        from .aws import ensure_aws_credentials
        ensure_aws_credentials()
    return kv


# Opened stores, keyed by their prefix spec, so every read and write under a prefix
# shares one: each S3 open rebuilds the credential chain and logs two benign ERROR
# lines. Unbounded; a run touches a handful of prefixes. Per process.
_STORES: dict[str, Any] = {}


def _open_store(kvstore: Mapping[str, Any]):
    import json as _json

    import tensorstore as ts

    cache_key = _json.dumps(dict(kvstore), sort_keys=True, default=str)
    store = _STORES.get(cache_key)
    if store is None:
        store = ts.KvStore.open(ensure_credentials(kvstore)).result()
        _STORES[cache_key] = store
    return store


def _kv(location: str | Mapping[str, Any], *parts: str):
    kv = join(to_kvstore(location), *parts) if parts else to_kvstore(location)
    key = ""
    base = str(kv.get("path", ""))
    if base:
        # The final segment is the key, so the store opens on the containing prefix.
        head, _, tail = base.rstrip("/").rpartition("/")
        kv["path"] = (head + "/") if head else ("/" if base.startswith("/") else "")
        key = tail
    return _open_store(kv), key


def read_bytes(location: str | Mapping[str, Any], *parts: str) -> bytes | None:
    """Read one object; ``None`` if it does not exist."""
    store, key = _kv(location, *parts)
    result = store.read(key).result()
    return bytes(result.value) if result.state == "value" else None


def read_range(location: str | Mapping[str, Any], start: int, end: int,
               *parts: str) -> bytes | None:
    """Read the half-open byte range ``[start, end)`` of one object; ``None`` if absent.

    A range reaching past the end of the object raises ``OUT_OF_RANGE`` (a ``ValueError``)
    on both the file and S3 drivers. On an object store this is an HTTP Range request.
    """
    store, key = _kv(location, *parts)
    result = store.read(key, byte_range=slice(int(start), int(end))).result()
    return bytes(result.value) if result.state == "value" else None


def write_bytes(location: str | Mapping[str, Any], data: bytes, *parts: str) -> None:
    """Write one object, creating parent directories for the file driver."""
    store, key = _kv(location, *parts)
    store.write(key, bytes(data)).result()


def list_keys(location: str | Mapping[str, Any], *parts: str,
              limit: int | None = None) -> list[str]:
    """Key names directly under a prefix.

    ``limit`` truncates the result, not the request: tensorstore enumerates the whole
    prefix first, so a prefix with millions of objects costs the same either way. To ask
    about one object, probe it by name with :func:`exists`; to bound a listing, list a
    narrower prefix.
    """
    kv = join(to_kvstore(location), *parts) if parts else dict(to_kvstore(location))
    store = _open_store(as_directory(kv))
    out: list[str] = []
    for key in store.list().result():
        out.append(key.decode() if isinstance(key, bytes) else str(key))
        if limit is not None and len(out) >= limit:
            break
    return out


def exists(location: str | Mapping[str, Any], *parts: str) -> bool:
    """True if the object exists. One request; safe against object stores."""
    store, key = _kv(location, *parts)
    return store.read(key).result().state == "value"


def read_json(location: str | Mapping[str, Any], *parts: str):
    """Read and parse a JSON object; ``None`` if it does not exist."""
    import json

    raw = read_bytes(location, *parts)
    return None if raw is None else json.loads(raw.decode("utf-8"))


def write_json(location: str | Mapping[str, Any], obj: Any, *parts: str,
               indent: int | None = 2) -> None:
    """Serialize ``obj`` as JSON and write it."""
    import json

    write_bytes(location, json.dumps(obj, indent=indent).encode("utf-8"), *parts)


def default_progress_path(dst: str | Mapping[str, Any]) -> str:
    """Default manifest path for a destination (local: alongside it; remote: cwd).

    A manifest is appended to and fsynced, so for a remote destination it lands in the
    working directory. Its name then carries a digest of the whole location beside the
    last component (``seg.4f2a91c7.progress.jsonl``), so two destinations ending in the
    same word never share a manifest, which would make each resume skip the other's
    blocks. The digest covers driver, bucket and path only, trailing slash removed, so one
    destination always maps to one manifest.
    """
    kv = to_kvstore(dst)
    if kv.get("driver") == "file":
        return kv["path"].rstrip("/") + ".progress.jsonl"
    path = str(kv.get("path", "")).rstrip("/")
    name = path.split("/")[-1] or str(kv.get("bucket", "volume"))
    identity = f"{kv.get('driver', '')}|{kv.get('bucket', '')}|{path}"
    digest = hashlib.blake2b(identity.encode(), digest_size=4).hexdigest()
    return f"{name}.{digest}.progress.jsonl"
