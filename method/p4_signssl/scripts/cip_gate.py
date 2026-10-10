"""CIP-Sign gate test (proposal5 §9-§10) on ASL Citizen train geometry states.

For each of 5 signer folds: train G0/G1/Gsh on the other 4 folds (5% of their videos held
out as forecaster_val for best-step selection), then evaluate on the held-out fold with a
COMMON example set (same video/anchor/edge/horizon for all models; Gsh partner fixed).
Gate per edge/horizon (geometry block only; no appearance block in this adaptation):
  rel. MSE gain (G0->G1) >= 5%, 95% signer-cluster bootstrap CI of Δ=MSE0-MSE1 > 0,
  G1 better than Gsh with CI > 0, increment RMS >= 0.05 standardized units,
  >= 10 groups and >= 1000 examples.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4 import cip_forecast as CF  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime" / (sys.argv[2] if len(sys.argv) > 2 else "cip")


def train_one(mode, S, V, off, tr_vids, va_vids, seed, steps, dev):
    torch.manual_seed(seed)
    model = CF.Forecaster(D=S.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    smp = CF.Sampler(S, V, off, tr_vids, seed)
    vsmp = CF.Sampler(S, V, off, va_vids, seed + 1)
    vb = [x.to(dev) for x in vsmp.draw(4096, mode)]
    best, best_state, hist = 1e9, None, []
    for step in range(1, steps + 1):
        lr = 3e-4 * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        b = [x.to(dev) for x in smp.draw(256, mode)]
        model.train()
        pred = model(*b[:8])
        err, ok = CF.masked_mse(pred, b[8], b[9])
        loss = err[ok].mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                e, o = CF.masked_mse(model(*vb[:8]), vb[8], vb[9])
                v = e[o].mean().item()
            hist.append((step, loss.item(), v))
            if v < best:
                best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), {"best_val_mse": best, "history": hist}


def eval_set(S, V, off, vids, rng, anchors_per_video=6):
    """Common examples: list of (v, t, partner_v, partner_t) per video; all edges/horizons."""
    out = []
    vids = np.asarray(vids)
    lens = off[vids + 1] - off[vids]
    good = vids[lens >= CF.K + max(CF.HORIZONS)]
    for v in good:
        T = off[v + 1] - off[v]
        for t in rng.choice(np.arange(CF.K - 1, T - max(CF.HORIZONS)), min(anchors_per_video, T - max(CF.HORIZONS) - CF.K + 1), replace=False):
            u = good[rng.integers(len(good))]
            while u == v:
                u = good[rng.integers(len(good))]
            Tu = off[u + 1] - off[u]
            out.append((v, int(t), int(u), int(rng.integers(CF.K - 1, Tu))))
    return out


@torch.no_grad()
def evaluate(models, S, V, off, ex, dev, chunk=8192):
    rows = []
    hist = lambda v, t, p: (S[off[v] + t - CF.K + 1: off[v] + t + 1, p], V[off[v] + t - CF.K + 1: off[v] + t + 1, p])
    batch = []

    def run(batch):
        cols = list(zip(*batch))
        Yb, Mb, Ya, Ma, Yu, Mu, P0, P1, E, H, YT, MT, YP, meta = cols
        T = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x)).to(dt).to(dev)
        yb, mb, ya, ma, yu, mu = T(Yb), T(Mb, torch.bool), T(Ya), T(Ma, torch.bool), T(Yu), T(Mu, torch.bool)
        p0, p1, e, h = T(P0, torch.long), T(P1, torch.long), T(E, torch.long), T(H, torch.long)
        yt, mt, yp = T(YT), T(MT, torch.bool), T(YP)
        g0 = models["g0"](yb, mb, yb, mb, p0, p0, e, h)
        g1 = models["g1"](yb, mb, ya, ma, p0, p1, e, h)
        gs = models["gsh"](yb, mb, yu, mu, p0, p1, e, h)
        out = {}
        for k, pr in (("g0", g0), ("g1", g1), ("gsh", gs), ("persist", yp)):
            out[k] = CF.masked_mse(pr, yt, mt)[0].cpu().numpy()
        inc = ((g1 - g0) ** 2).mean(-1).cpu().numpy()
        for i, m in enumerate(meta):
            rows.append(m + (out["g0"][i], out["g1"][i], out["gsh"][i], out["persist"][i], inc[i]))

    for (v, t, u, tu) in ex:
        for ei, (a, b) in enumerate(CF.EDGES):
            yb, mb = hist(v, t, b)
            ya, ma = hist(v, t, a)
            yu, mu = hist(u, tu, a)
            if mb.any(-1).sum() < 6 or ma.any(-1).sum() < 6 or mu.any(-1).sum() < 6:
                continue
            for hi, d in enumerate(CF.HORIZONS):
                s = off[v] + t + d
                if not V[s, b].any():
                    continue
                yp = np.where(mb[-1], yb[-1], 0)  # persistence (last own state)
                batch.append((yb, mb, ya, ma, yu, mu, b, a, ei, hi, S[s, b], V[s, b], yp, (int(v), ei, hi)))
                if len(batch) >= chunk:
                    run(batch); batch = []
    if batch:
        run(batch)
    return np.array(rows, dtype=np.float64)


def cluster_boot(diff_by_group, n=2000, seed=0):
    g = np.array(list(diff_by_group.values()))
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
    return float(g.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    dev = "cuda"
    meta = json.load(open(RT / "states_meta.json"))
    if meta.get("blocks"):
        CF.BLOCKS = [tuple(b) for b in meta["blocks"]]
    S = np.load(RT / "states.npy").astype(np.float32)
    V = np.load(RT / "valid.npy")
    off = np.load(RT / "offsets.npy")
    vids = [json.loads(l) for l in open(RT / "videos.jsonl")]
    fold = np.array([v["fold"] for v in vids])
    signer = np.array([v["signer"] for v in vids])
    rng = np.random.default_rng(0)
    all_rows, logs = [], {}
    t0 = time.time()
    for f in range(5):
        tr = np.where((fold >= 0) & (fold != f))[0]
        rng.shuffle(tr)
        nva = int(0.05 * len(tr))
        va, tr = tr[:nva], tr[nva:]
        models = {}
        for mode in ("g0", "g1", "gsh"):
            models[mode], logs[f"f{f}_{mode}"] = train_one(mode, S, V, off, tr, va, 100 + f, steps, dev)
            print(f"fold {f} {mode} best val mse {logs[f'f{f}_{mode}']['best_val_mse']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        ex = eval_set(S, V, off, np.where(fold == f)[0], rng)
        r = evaluate(models, S, V, off, ex, dev)
        all_rows.append(r)
        print(f"fold {f}: {len(r)} eval examples", flush=True)
    R = np.concatenate(all_rows)
    np.save(RT / "forecast_errors.npy", R)  # cols: video, edge, hz, g0, g1, gsh, persist, inc2
    gates = {}
    for ei, (a, b) in enumerate(CF.EDGES):
        for hi, d in enumerate(CF.HORIZONS):
            sel = (R[:, 1] == ei) & (R[:, 2] == hi)
            X = R[sel]
            per_v = {}
            for row in X:
                per_v.setdefault(int(row[0]), []).append(row[3:])
            # aggregate per video, then per signer group
            byg = {}
            for v, lst in per_v.items():
                byg.setdefault(signer[v], []).append(np.mean(lst, 0))
            G = {s: np.mean(x, 0) for s, x in byg.items()}
            d01 = cluster_boot({s: x[0] - x[1] for s, x in G.items()})
            dsh = cluster_boot({s: x[2] - x[1] for s, x in G.items()})
            m0 = float(np.mean([x[0] for x in G.values()]))
            m1 = float(np.mean([x[1] for x in G.values()]))
            msh = float(np.mean([x[2] for x in G.values()]))
            mp = float(np.mean([x[3] for x in G.values()]))
            rms = float(np.sqrt(np.mean(X[:, 7]))) if len(X) else 0.0
            rel = d01[0] / max(m0, 1e-8)
            ok = (len(G) >= 10 and len(X) >= 1000 and rel >= 0.05 and d01[1] > 0 and dsh[1] > 0 and rms >= 0.05)
            names = ["LH", "RH", "FACE", "BODY"]
            gates[f"{names[a]}->{names[b]}@{d}"] = {
                "gate": int(ok), "n_examples": int(len(X)), "n_groups": len(G), "mse_g0": m0, "mse_g1": m1,
                "mse_gsh": msh, "mse_persist": mp, "rel_gain": rel, "delta_ci": d01, "g1_vs_gsh_ci": dsh, "increment_rms": rms}
    json.dump({"gates": gates, "train_logs": logs, "steps": steps,
               "n_valid_edges": int(sum(g["gate"] for g in gates.values()))}, open(RT / "gates.json", "w"), indent=1)
    for k, g in gates.items():
        print(f"{k:14s} gate={g['gate']} rel={g['rel_gain']:+.3f} Δci=[{g['delta_ci'][1]:+.4f},{g['delta_ci'][2]:+.4f}] "
              f"g1<gsh ci=[{g['g1_vs_gsh_ci'][1]:+.4f},{g['g1_vs_gsh_ci'][2]:+.4f}] rms={g['increment_rms']:.3f} "
              f"mse g0={g['mse_g0']:.3f} g1={g['mse_g1']:.3f} gsh={g['mse_gsh']:.3f} persist={g['mse_persist']:.3f}")
    print("VALID EDGES", sum(g["gate"] for g in gates.values()), "/", len(gates))


if __name__ == "__main__":
    main()
