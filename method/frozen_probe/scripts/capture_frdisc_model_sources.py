"""Capture prospectively authorized, exact pinned primary source files only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    start = time.monotonic()
    commit = 'c8115c09c54d5480bf0adf972f66988bb2274c2b'
    wanted = {
        'models/safsar.py': (15509, '4e81bdd237c1c0d6c84c702c3d36dc033e1c91650adc673b5acaa3c1aeadbb62'),
        'train.py': (25579, '5e46af5c88acf2040f2f1deb68e6a118cb35bad676889b76717455e00165f440'),
        'videoloader.py': (18560, '006fea1349181eaf0aead6155961ff20c045840f94a802c08379144685c65613'),
        'configs/SAFSAR.json': (671, '15a2f593bf09910c9b95d813e214fc2f3200419942c6fd161635929215b1e5bf'),
        'LICENSE': (1518, 'd3522fc7814fc79c74f64c68dc9cb591b062b5e90a5bddd50e444e5b3f727711'),
    }
    target = Path('method/frozen_probe/references/fsosar_c8115c09_model')
    assert not target.exists()
    assert sum(size for size, _ in wanted.values()) < 128 << 10
    sources = {}
    for name, (size, digest) in wanted.items():
        url = f'https://raw.githubusercontent.com/HSP-IIT/fsosar/{commit}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(size + 1)
        assert len(data) == size and hashlib.sha256(data).hexdigest() == digest
        sources[name] = data
    target.mkdir()
    files = {}
    for name, data in sources.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(data)
        files[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    record = dict(captured_utc=datetime.now(timezone.utc).isoformat(), decision='D086',
                  commit=commit, source_bytes=sum(len(b) for b in sources.values()),
                  source_ceiling_bytes=128 << 10, files=files,
                  captured_wall_seconds=time.monotonic() - start,
                  scope='Exact pinned files and full BSD license; selected symbols read, not full-paper/model/training execution or benchmark reproduction.')
    with Path('method/research_lab/records/inventory/FRDISC_MODEL_SOURCE_CAPTURE_r1.json').open('x') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
