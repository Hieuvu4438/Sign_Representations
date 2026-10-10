import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from signrel.data.manifest import sha256
from signrel.eval.retrieval import retrieval
from signrel.eval.linear_probe import linear_probe
from signrel.eval.phonology import load_metadata,phonological_probes
from signrel.eval.signer_probe import signer_probe
from signrel.eval.minimal_pairs import minimal_pairs


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return z["features"], [json.loads(r) for r in z["rows_json"].tolist()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",required=True,help="Held-out query features")
    p.add_argument("--gallery",required=True,help="Gallery/train features")
    p.add_argument("--config",required=True)
    p.add_argument("--out",required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    q,qr = load(args.features)
    g,gr = load(args.gallery)
    from signrel.eval.frozen import evaluate_features
    report = evaluate_features(q,qr,g,gr,cfg)
    report.update(feature_sha256=sha256(args.features),gallery_sha256=sha256(args.gallery),
                  config_sha256=sha256(args.config))
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report["retrieval"]))


if __name__ == "__main__":
    main()
