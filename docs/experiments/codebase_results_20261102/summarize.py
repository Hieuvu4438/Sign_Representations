"""Export existing ASL-Citizen evidence; no model loading or evaluation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sources = {}


def read(path):
    p = ROOT / path
    data = p.read_bytes()
    sources[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return json.loads(data) if p.suffix == ".json" else data.decode("utf-8-sig")


def export(name, rows):
    (OUT / (name + ".json")).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    if rows:
        with (OUT / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                              for k, v in r.items()} for r in rows])


def table(rows, columns):
    result = ["| " + " | ".join(title for _, title in columns) + " |",
              "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _ in columns:
            v = row.get(key)
            cells.append("—" if v is None or v == "" else f"{v:.3f}" if isinstance(v, float) else str(v))
        result.append("| " + " | ".join(cells) + " |")
    return "\n".join(result)


old = "docs/experiments/method_comparison_20261019/"
audit = []
for name, spec in read(old + "sources.json").items():
    bundle = old + "source_results/" + name
    read(bundle)
    read(spec["original"])
    record = dict(source=spec["original"], bundle=bundle,
                  bundle_matches=sources[bundle]["sha256"] == spec["sha256"],
                  original_matches=sources[spec["original"]]["sha256"] == spec["sha256"])
    assert record["bundle_matches"] and record["original_matches"], record
    audit.append(record)
export("source_audit", audit)
official = read(old + "official_test_retrieval.json")
export("official_test_retrieval", official)
recognition = read(old + "recognition_test.json")
export("recognition_test", recognition)

# Re-read every model JSON rather than trusting the earlier CSV's numbers.
pilot_index = list(csv.DictReader(read("docs/experiments/method_comparison_20261018/pilot_ASLC300_white512.csv").splitlines()))
pilots = []
for row in pilot_index:
    d = read(row["source_json"])
    a, l = d["aslc300"]["retrieval_white512"], d["loso"]["white512"]
    r = dict(model=row["model"], train_clips=row["pretraining_clips"], steps=row["optimizer_steps"],
             ASLC300_R1=100*a["R@1"], ASLC300_R5=100*a["R@5"], ASLC300_DCG_x100=100*a["DCG"],
             LOSO_R1=100*l["R@1"], LP_macro=100*d["aslc300"]["linear_probe"]["macro_top1"],
             source=row["source_json"])
    assert abs(r["ASLC300_R1"] - float(row["ASLC300_Rec1_pct"])) < 1e-8
    assert abs(r["LP_macro"] - float(row["ASLC300_LP_macro_pct"])) < 1e-8
    pilots.append(r)
export("pilot_ASLC300_white512", pilots)

raw_data = read("method/obs_jepa/runtime/eval/comparison_avg_aslc300/summary.json")
raw = []
for name, d in raw_data["models"].items():
    m = d["metrics"]
    raw.append(dict(model=name, input=d["input"], R1=100*m["Rec@1"], R5=100*m["Rec@5"],
                    DCG_x100=100*m["DCG"], initialization=d["initialization"]))
raw.sort(key=lambda x: -x["R1"])
export("pilot_ASLC300_raw_avg", raw)

legacy = []
for name in ("J", "JP", "M", "MP", "S", "JREL", "JNM", "JMOT"):
    src = f"method/jepa_targets/runtime/eval_{name}.json"
    d = read(src)[name]
    cfg = read(f"method/jepa_targets/runtime/{name}/config.json")
    for pooling in ("mean", "bins2", "white512", "white512_bins2"):
        if pooling not in d:
            continue
        m = d[pooling]
        legacy.append(dict(model=name, pooling=pooling, R1=100*m["R@1"], R5=100*m["R@5"],
                           DCG_x100=100*m["DCG"], queries=m["n"], steps=cfg["steps"],
                           target=f"latent={cfg['latent']}; pixel={cfg['pixel']}; priors={cfg['priors']}", source=src))
export("jepa_targets_LOSO", legacy)

asc = {}
for src in ("method/asc_jepa/runtime/eval_ex13.json", "method/asc_jepa/runtime/eval_ex13b.json"):
    for name, d in read(src).items():
        if "wmean" in d:
            asc[name] = dict(model=name, weighted_R1=100*d["wmean"]["R@1"],
                             bins2_R1=100*d["bins2"]["R@1"], queries=d["wmean"]["n"], source=src)
export("asc_validation", list(asc.values()))
rel_src = "method/signrel_jepa/reports/controlled_pilot/summary_test.json"
rel = []
for name, d in read(rel_src)["summary"].items():
    rel.append(dict(model=name, R1_mean=100*d["Recall@1"]["mean"], R1_seed_SD=100*d["Recall@1"]["std"],
                    R5_mean=100*d["Recall@5"]["mean"], seeds=d["Recall@1"]["seeds"], source=rel_src))
export("signrel_40gloss_pilot", rel)

metrics = [("method", "Method"), ("readout", "Readout"), ("R1", "R@1 %"), ("R5", "R@5 %"), ("DCG_x100", "DCG×100")]
parts = ["# Thống kê kết quả ASL-Citizen trong codebase — 02/11/2026", "",
"Rà soát các số đo đã lưu tại workspace. Không train, không chạy inference hoặc chấm lại test. Ngày rà soát không phải ngày thực nghiệm; phần lớn kết quả chính được lưu trong tháng 10/2026.", "",
"## Kết quả chính", "",
"- Skeleton: **PhonoJEPA PLctx full160**, official test R@1 **16,678%**, R@5 **36,896%**, DCG×100 **39,618**; train scratch trên 40.154 clip, 160 epochs / 200.640 updates, một seed.",
"- Pipeline RGB: **SignRep + Chamfer20**, R@1 **58,420%**. Đây là frozen encoder pretrain bên ngoài và set matching; bins2/Chamfer là readout, không phải các encoder được train độc lập trên ASLC.",
"- Head SSL **H008**: single vector 1024D trên frozen SignRep; ba seed đăng ký 57,28 / 57,31 / 57,31%, trung bình **57,300%**. Đây là học head không nhãn trên ASLC, khác train encoder từ đầu.",
"- ASLC300 white512: JH3D R@1 **56,800%**; PLctx16k **48,933%**; PLctx full160 **47,667%**. Linear probe full160 **68,867%** chỉ thuộc pilot 300 gloss.", "",
"## Official test: frozen dictionary retrieval", "",
"Gallery danh nghĩa: official train 40.154 clip; query: official test 32.941 clip; 2.731 gloss. Điểm gloss là max similarity trên các video gallery cùng gloss. R@K là tỷ lệ rank≤K; DCG = mean[1/log2(1+rank)]. Mọi DCG trong báo cáo được nhân 100. Các bảng giữ readout chính/phụ đã khai báo trước, không chọn lại theo test.", ""]
for group, title in (("skeleton_raw", "Skeleton: raw readout"), ("skeleton_postprocessed", "Skeleton: PCA/whitening"), ("RGB_SignRep_pipeline", "RGB: external pretraining / SignRep")):
    rs = sorted([r for r in official if r["group"] == group], key=lambda x: -x["R1"])
    parts += ["### " + title, "", table(rs, metrics), ""]
parts += ["PhonoPath A1 có readout **chính grp+P**: epoch40 8,567%, epoch160 6,928%. grp là readout phụ, tương ứng 11,305% và 11,511%. Pose raw là chính; white512 là hậu xử lý báo riêng. PLctx continuation không cải thiện full160 (−0,0455 điểm). ASM chọn checkpoint 5.000 bằng validation dù train đủ 60.000 steps.", "",
"Pose chỉ có 40.126 gallery dùng được và 4 query lỗi; ASM 40.132 gallery dùng được và 6 query lỗi. Query denominator vẫn là 32.941; PLctx giữ zero vectors. Khác biệt preprocessing/failure policy và pretrain làm các hàng không trở thành đối chứng nhân quả.", "",
"Nguồn: [bảng và source bundle 19/10](../method_comparison_20261019/REPORT_VI.md). 11 source bundle và 11 file nguồn gốc khớp SHA256 đã đăng ký; xem `source_audit.csv`. P4/H008 là kết quả đăng ký còn trong tài liệu, chưa xác minh lại bằng weights/inference.", "",
"## Recognition official test", "", table(recognition, [("method", "Method"), ("regime", "Cách train head"), ("top1", "Top-1 %"), ("top5", "Top-5 %")]), "",
"Linear probe dùng nhãn để train classifier trên frozen encoder. Fine-tune/scratch là supervised downstream. ASM fine-tune hơn scratch 1,606 điểm Top-1 ở một seed; chưa đủ để kết luận ổn định qua seed. Không gộp accuracy classifier với retrieval R@1.", "",
"## Pilot ASLC300: white512 và LOSO", "",
f"{len(pilots)} cấu hình/readout được đọc lại từ JSON gốc, đối chiếu R@1 và LP với bảng lưu trước đó; không tính các bản JSON sao chép là seed độc lập. ASLC300: 3.000 gallery train, 1.500 query validation, 300 gloss. LOSO: 10.304 validation clip, gallery là các signer khác trong validation. LOSO whitening fit trên validation nên mang tính transductive. Hai cột là hai protocol khác nhau.", "",
table(pilots, [("model", "Run"), ("ASLC300_R1", "ASLC300 R@1 %"), ("ASLC300_R5", "R@5 %"), ("ASLC300_DCG_x100", "DCG×100"), ("LOSO_R1", "LOSO R@1 %"), ("LP_macro", "LP macro %")]), "",
"## Pilot ASLC300: raw-average đã tính lại", "",
"Protocol: raw window mean → video L2; không PCA, không activity weighting. Bảng này khác `retrieval_mean` cũ vốn L2 từng window trước pooling. JH3D 45,93% ở đây và 56,80% white512 ở trên đều là số đúng theo từng protocol. SignRep public không cập nhật encoder trên ASLC.", "",
table(raw, [("model", "Run"), ("input", "Input"), ("R1", "R@1 %"), ("R5", "R@5 %"), ("DCG_x100", "DCG×100")]), "",
"## JEPA-targets và ASC-JEPA: validation bổ sung", "",
"Các nhánh RGB J/JP/M/MP/S/JREL/JNM/JMOT có log/config và LOSO validation, 4.000 steps, batch32. CSV `jepa_targets_LOSO.csv` giữ mọi readout đã có; riêng eval_M.json chỉ có mean/bins2, không tự điền whitening. Không gán các điểm này thành official test.", "",
table([r for r in legacy if r["pooling"] == "white512"], [("model", "Run"), ("target", "Target"), ("R1", "LOSO white512 R@1 %"), ("R5", "R@5 %")]), "",
"ASC-JEPA warm-start SignRep public, 3.000 steps (B4 config), đánh giá LOSO validation 10.304 clip. B0 là baseline, asc_B* là continuation. Cả B4 và B2 vẫn thấp hơn baseline ở weighted R@1; bins2 là readout khác.", "",
table(list(asc.values()), [("model", "Run"), ("weighted_R1", "Weighted R@1 %"), ("bins2_R1", "Bins2 R@1 %")]), "",
"## Pilot nhỏ và trạng thái method khác", "",
"SignRel-JEPA: local pilot 40 gloss, ba training seed, không phải full benchmark. Mean ± SD theo seed:", "",
table(rel, [("model", "Run"), ("R1_mean", "R@1 mean %"), ("R1_seed_SD", "SD điểm %"), ("R5_mean", "R@5 mean %"), ("seeds", "Seeds")]), "",
"Nguồn kết luận pilot: `relation_benefit_not_supported_by_this_pilot`. SignRel-Temporal là thí nghiệm riêng: SSL train 47 clip How2Sign + 3 dev, retrieval subset ASLC 40 gloss. A4−A1 macro R@1 trung bình +0,42 điểm qua ba seed, hướng không nhất quán; không phải method pretrain trên ASLC full. Xem [report gốc](../../../method/signrel_temporal/results/confirmatory_report.md).", "",
"| Thư mục / nhóm | Bằng chứng và phạm vi |", "|---|---|",
"| `phonojepa` | PLctx/PLint/PLboth và ablation có pilot; PLctx full160 và continuation có official test. Checkpoint full160 còn local. |",
"| `phonojepa_s1`, `phonojepa_s2` | Scratch ASLC002 3.000 clip, 4.000 steps; có ASLC300 + LOSO, chưa có full official test được xác minh. |",
"| `jepa_targets` | JEPA + prior/ablation trên RGB, gồm JH3D/JV3D/SR3D/SRVJ; pilot và LOSO. |",
"| `obs_jepa` | A00 foundation, các nhánh A adaptation và B10–B13 graph landmark; 4.000 steps trên 3.000 clip. B gate không đạt, P3 chưa chạy. |",
"| `asc_jepa` | SignRep warm-start + state/change; có validation, chưa thấy full official test. |",
"| `jepa_baseline` | ASLC001/002 pilot; full-ASLC reproduction dừng ở 40.000 steps và archive. Best proxy R@1 1,867% tại 10k, 5.462 gallery/2.731 val; không có test. |",
"| `pose_jepa` | Official retrieval + linear probe đã archive; weights/cache/runtime đã xóa, không thể inference từ archive hiện tại. |",
"| `asm_jepa` | Full-ASLC SSL và downstream linear/fine-tune/scratch có official test; một seed. |",
"| `phonopath_jepa` | Epoch40 và epoch160 có official test; metric/ranks/config đã archive, weights không giữ. |",
"| `p4_signssl` | SignRep readout/continuation/SSL head H008; số official test dùng ledger và bảng đăng ký. |",
"| `signrel_jepa` | Controlled local pilot 40 gloss, ba seed; không có lợi ích retrieval rõ của relation. |",
"| `signrel_temporal` | SSL How2Sign; ASLC retrieval subset riêng, không xếp cùng full-ASLC train. |",
"| `signpathlab` | PILOT8: 32 train/16 val, ba seed ×100 updates. E03/E04 cùng PC Top1 27,08%, SD 3,61 điểm. Chỉ tìm thấy evidence khởi động ASLC3000; chưa thấy metric hoàn tất để thống kê. |",
"| `signcoord_ssl` | Smoke/control evidence trên dữ liệu thật, có run 4 updates; chưa xác minh score full ASLC. Không dùng smoke làm benchmark. |",
"| `plctx_pose_fusion` | Đã dừng/dọn runtime 17/10; chưa có full-test fusion để xếp hạng. |",
"| `signrep_spectral`, `frozen_probe`, `stream_reliability`, `research_lab` | Có records/probe/audit/continuation riêng; không xác định là skeleton SSL full-test encoder mới trong bảng chính. |", "",
"## Diễn giải thống kê", "",
"Đa số run chính chỉ một seed. Chênh lệch điểm không mặc nhiên có ý nghĩa thống kê; CI bootstrap theo signer chỉ phản ánh lấy mẫu signer, không thay thế độ dao động do train seed. Không tính trung bình giữa các method/protocol. H008 là ba seed; SignRel pilot ba seed nhưng phạm vi nhỏ.", "",
"Full160 so với PLctx16k: LOSO +4,16 điểm (CI95 [+1,91; +5,12]); LP ASLC300 +8,07 ([+4,06; +12,12]); retrieval ASLC300 −1,27 ([−3,01; +2,25]). Đây là kết quả bootstrap đã lưu trong [báo cáo 13/10](../../proposal12/comparison_plctx_20261013/REPORT_VI.md), một seed/cặp model; không tính lại CI mới.", "",
"## File xuất và tái lập", "",
"`official_test_retrieval.csv/json`, `recognition_test.csv/json`, `pilot_ASLC300_white512.csv/json`, `pilot_ASLC300_raw_avg.csv/json`, `jepa_targets_LOSO.csv/json`, `asc_validation.csv/json`, `signrel_40gloss_pilot.csv/json` và `source_audit.csv/json`. `sources.json` ghim file đọc bằng SHA256; `summarize.py` tạo lại báo cáo bằng Python stdlib. Các cột trống là chưa có giá trị, không phải 0. Script chỉ tổng hợp text/JSON, không mở checkpoint."]
(OUT / "REPORT_VI.md").write_text("\n".join(parts) + "\n")
(OUT / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(dict(official_rows=len(official), pilot_white512_rows=len(pilots), raw_rows=len(raw),
                      legacy_rows=len(legacy), audited_bundles=len(audit), sources=len(sources)), indent=2))
