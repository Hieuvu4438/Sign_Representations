import hashlib
import io
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml
from signpath.assets import fetch_asset


class Response(io.BytesIO):
    def __init__(self, content, content_type="application/octet-stream"):
        super().__init__(content)
        self.headers = {"Content-Type": content_type}


def registry(tmp_path, sha=None):
    path = tmp_path / "registries/assets.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump({"schema_version": 1, "models": {"test": {
        "local_path": "checkpoints/test.pt", "weights_url": "https://example.invalid/test.pt", "sha256": sha}}}))
    return path


def test_download_receipt_atomic_publish_and_repeat(tmp_path):
    content = b"PK" + b"x" * 2000
    path = registry(tmp_path)
    with patch("urllib.request.urlopen", return_value=Response(content)):
        result = fetch_asset(path, "test")
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert Path(result["path"]).read_bytes() == content
    assert not list((tmp_path / "checkpoints").glob("*.part"))
    with patch("urllib.request.urlopen") as network:
        assert fetch_asset(path, "test")["status"] == "already_verified"
        network.assert_not_called()


@pytest.mark.parametrize("content,ctype,sha", [
    (b"<html>login</html>", "text/html", None),
    (b"<!DOCTYPE html>" + b"x" * 2000, "application/octet-stream", None),
    (b"PK" + b"x" * 2000, "application/octet-stream", "a" * 64),
    (b"PKshort", "application/octet-stream", None),
])
def test_invalid_download_is_never_published(tmp_path, content, ctype, sha):
    path = registry(tmp_path, sha)
    with patch("urllib.request.urlopen", return_value=Response(content, ctype)):
        with pytest.raises(ValueError):
            fetch_asset(path, "test")
    assert not (tmp_path / "checkpoints/test.pt").exists()
    assert not list((tmp_path / "checkpoints").glob("*.part"))
