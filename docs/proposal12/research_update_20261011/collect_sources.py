"""Download public primary papers and record integrity/provenance; never runs external code."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import requests
import fitz

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "sjepa": "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf",
    "signrep": "https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.pdf",
    "signseek": "https://arxiv.org/pdf/2609.03695v1",
    "slim": "https://arxiv.org/pdf/2603.10648v3",
    "skeleton2vec": "https://arxiv.org/pdf/2401.00921v1",
    "mamp": "https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.pdf",
    "gfp": "https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Towards_Efficient_General_Feature_Prediction_in_Masked_Skeleton_Modeling_ICCV_2025_paper.pdf",
    "amr": "https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_Exploring_Adaptive_Masked_Reconstruction_for_Self-Supervised_Skeleton-Based_Action_Recognition_CVPR_2026_paper.pdf",
    "best": "https://ojs.aaai.org/index.php/AAAI/article/download/25470/25242",
    "masa": "https://arxiv.org/pdf/2405.20666v1",
    "stc": "https://arxiv.org/pdf/2406.10501v1",
    "signbert": "https://arxiv.org/pdf/2110.05382",
    "signbertplus": "https://arxiv.org/pdf/2305.04868",
    "motionbert": "https://arxiv.org/pdf/2210.06551",
    "shubert": "https://aclanthology.org/2025.acl-long.1397.pdf",
    "signdino": "https://arxiv.org/pdf/2609.06296v1",
    "ijepa": "https://arxiv.org/pdf/2301.08243",
    "lejepa": "https://arxiv.org/pdf/2511.08544",
    "vjepa21": "https://arxiv.org/pdf/2603.14482",
}

def collect(item):
    key, url = item
    rec = {"id": key, "requested_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    folder = ROOT / "sources"
    folder.mkdir(exist_ok=True)
    pdf = folder / (key + ".pdf")
    try:
        if not pdf.exists():
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("Response is not a PDF")
            pdf.write_bytes(r.content)
            rec["resolved_url"] = r.url
        rec["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        rec["bytes"] = pdf.stat().st_size
        pre = run_preflight(pdf)
        (folder / (key + ".preflight.json")).write_text(json.dumps(pre, indent=2))
        rec["preflight"] = pre["verdict"]
        with fitz.open(pdf) as doc:
            rec["pages"] = len(doc)
            text = "\n".join(f"\n=== PDF_PAGE_{i+1} ===\n" + p.get_text() for i, p in enumerate(doc))
        (folder / (key + ".txt")).write_text(text.replace("\x00", ""))
        rec["status"] = "downloaded_and_extracted"
    except Exception as exc:
        rec["status"] = "unavailable"
        rec["error"] = str(exc)
    print(key, rec["status"], rec.get("pages"), rec.get("preflight"), flush=True)
    return rec

if __name__ == "__main__":
    sys.path.insert(0, "/tmp/signrep_research_pdfdeps")
    skill_scripts = Path("/home/haipd/.codex/skills/academic-research-suite/ars/scripts")
    sys.path.insert(0, str(skill_scripts))
    from pdf_read_preflight import run_preflight
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(collect, SOURCES.items()))
    (ROOT / "source_downloads.json").write_text(json.dumps(records, indent=2))
