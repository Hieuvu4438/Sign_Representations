"""Streaming verified downloads with explicit receipts and no fallback initialization."""
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import yaml

from .cache.hashing import file_hash
from .config import UniqueLoader


def fetch_asset(registry, model):
    registry = Path(registry).resolve()
    values = yaml.load(registry.read_text(), Loader=UniqueLoader)
    if values.get("schema_version") != 1 or model not in values.get("models", {}):
        raise ValueError(f"Unknown asset {model} in {registry}")
    entry = values["models"][model]
    target = Path(entry["local_path"])
    if not target.is_absolute():
        target = registry.parent.parent / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        actual = file_hash(target)
        receipt_path = target.with_suffix(target.suffix + ".receipt.json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        verified_receipt = receipt.get("sha256") == actual and receipt.get("weights_url") == entry["weights_url"]
        if entry.get("sha256") != actual and not (entry.get("sha256") is None and verified_receipt):
            raise ValueError(f"Asset exists without a matching registry SHA-256: {target}")
        return {"registry_key": model, "path": str(target), "sha256": actual, "status": "already_verified"}
    with tempfile.NamedTemporaryFile(prefix=target.name + ".", suffix=".part", dir=target.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            request = urllib.request.Request(entry["weights_url"], headers={"User-Agent": "SignPathLab/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/" in content_type or "html" in content_type or "json" in content_type:
                    raise ValueError(f"Asset download returned an error/login page: {content_type}")
                head = response.read(1024)
                if head.lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml", b"<error")):
                    raise ValueError("Asset download returned markup instead of weights")
                stream.write(head)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            stream.flush()
            size = temp.stat().st_size
            if size < 1024 or (entry.get("expected_size_bytes") is not None and size != entry["expected_size_bytes"]):
                raise ValueError("Asset download size mismatch or incomplete file")
            actual = file_hash(temp)
            if entry.get("sha256") is not None and entry["sha256"] != actual:
                raise ValueError("Asset SHA-256 mismatch")
            # Hard link publishes atomically and refuses to replace a concurrent download.
            os.link(temp, target)
            receipt = {"registry_key": model, "path": str(target), "sha256": actual,
                       "size_bytes": size, "weights_url": entry["weights_url"],
                       "checksum_trust": "registry_verified" if entry.get("sha256") else "computed_after_download"}
            from .cli import write_json
            write_json(target.with_suffix(target.suffix + ".receipt.json"), receipt)
            return receipt
        finally:
            temp.unlink(missing_ok=True)
