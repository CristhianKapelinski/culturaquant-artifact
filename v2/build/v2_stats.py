"""CulturaQuant-v2 statistics: erosion, differential erosion, McNemar, MDE.

Reads the per-(item,rotation) cyclic rows written by scripts/cyclic_position_bias.py
for the cultural set and the rarity-matched control set, and computes, per model:

  - accuracy per (set, precision) and the fp16->quant erosion per set;
  - paired McNemar (continuity-corrected) fp16 vs quant, per set;
  - DIFFERENTIAL erosion  delta = (cult fp16 - cult quant) - (ctrl fp16 - ctrl quant)
    with an item-CLUSTERED bootstrap 95% CI (resample item ids with all 5 rotations);
  - the paired-design MDE (smallest true erosion detectable at 80% power, alpha .05);
  - per-rarity-band erosion.

Then a model-pooled differential (mean delta across models, clustered bootstrap CI):
the design's verdict statistic. CI excludes 0 and negative on cultural => cultural
erodes faster than equally-rare generic (thesis). CI includes 0 with a small MDE =>
conclusive null (quantization is culturally safe; degradation is frequency-driven).

Pure-python (no statsmodels), deterministic (seeded). Run locally on pulled rows:
  python v2_stats.py --cult-dir <dir-with-cyclic__*.jsonl> --ctrl-dir <dir> \
      --models Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 ...
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

PRECS = ["int8", "nf4"]
B = 5000  # bootstrap resamples


def load_rows(d: Path, model: str, prec: str) -> dict:
    """Return {(id, rotation): correct} for one (model, precision)."""
    slug = model.replace("/", "__") + f"__{prec}"
    p = d / f"cyclic__{slug}.jsonl"
    out = {}
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("_summary"):
            continue
        out[(r["id"], r["rotation"])] = int(r["correct"])
    return out


def load_bands(d: Path, model: str) -> dict:
    """Map id -> rarity band, read from any available cyclic file's companion? The
    cyclic rows don't carry band; read from the gate input files instead."""
    return {}


def acc(rows: dict) -> float:
    return sum(rows.values()) / len(rows) if rows else float("nan")


def mcnemar(a: dict, b: dict):
    """Paired McNemar over shared keys. a=fp16, b=quant. Returns (b01,c10,chi2,p)."""
    keys = a.keys() & b.keys()
    b01 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)  # fp16 right, quant wrong
    c10 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)  # fp16 wrong, quant right
    n = b01 + c10
    if n == 0:
        return b01, c10, 0.0, 1.0
    chi2 = (abs(b01 - c10) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))  # 1-CDF chi2_1 = erfc(sqrt(x/2))
    return b01, c10, chi2, p


def mde_paired(n_pairs: int, disc_rate: float, alpha=0.05, power=0.80) -> float:
    """Smallest detectable accuracy drop (proportion) for a paired/McNemar design,
    given n paired observations and the discordant-pair rate. Normal approx."""
    if n_pairs == 0 or disc_rate <= 0:
        return float("nan")
    za, zb = 1.959964, 0.841621
    # drop d ~ (b01-c10)/n; detectable |b01-c10| ~ (za+zb)*sqrt(n*disc_rate)
    return (za + zb) * math.sqrt(disc_rate / n_pairs)


def erosion(rows_fp: dict, rows_q: dict) -> float:
    keys = rows_fp.keys() & rows_q.keys()
    if not keys:
        return float("nan")
    return (sum(rows_fp[k] for k in keys) - sum(rows_q[k] for k in keys)) / len(keys)


def diff_boot(cf, cq, kf, kq, rng: random.Random):
    """Item-clustered bootstrap of differential erosion delta and its 95% CI.
    delta = (cult fp16-quant) - (ctrl fp16-quant), per-instance means."""
    cult_ids = sorted({k[0] for k in cf.keys() & cq.keys()})
    ctrl_ids = sorted({k[0] for k in kf.keys() & kq.keys()})

    def by_item(rows_fp, rows_q, ids):
        m = {}
        for i in ids:
            ks = [(i, r) for r in range(5) if (i, r) in rows_fp and (i, r) in rows_q]
            if ks:
                m[i] = (sum(rows_fp[k] for k in ks), sum(rows_q[k] for k in ks), len(ks))
        return m

    cm = by_item(cf, cq, cult_ids)
    km = by_item(kf, kq, ctrl_ids)

    def delta_from(cids, kids):
        cfp = sum(cm[i][0] for i in cids); cqq = sum(cm[i][1] for i in cids)
        cn = sum(cm[i][2] for i in cids)
        kfp = sum(km[i][0] for i in kids); kqq = sum(km[i][1] for i in kids)
        kn = sum(km[i][2] for i in kids)
        if cn == 0 or kn == 0:
            return None
        return (cfp - cqq) / cn - (kfp - kqq) / kn

    point = delta_from(list(cm), list(km))
    ci = list(cm); ki = list(km)
    boots = []
    for _ in range(B):
        cs = [ci[rng.randrange(len(ci))] for _ in ci]
        ks = [ki[rng.randrange(len(ki))] for _ in ki]
        v = delta_from(cs, ks)
        if v is not None:
            boots.append(v)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cult-dir", required=True)
    ap.add_argument("--ctrl-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--bands-cult", default=None,
                    help="optional midband_cultural.jsonl to report per-band erosion")
    args = ap.parse_args()
    rng = random.Random(20260609)
    cd, kd = Path(args.cult_dir), Path(args.ctrl_dir)

    id_band = {}
    if args.bands_cult and Path(args.bands_cult).exists():
        for ln in Path(args.bands_cult).read_text().splitlines():
            if ln.strip():
                r = json.loads(ln); id_band[r["id"]] = r["rarity"]["band"]

    print("=" * 78)
    print("CulturaQuant-v2 — erosion & differential erosion (cyclic, rarity-matched)")
    print("=" * 78)
    pooled = {p: [] for p in PRECS}
    for model in args.models:
        cf = load_rows(cd, model, "fp16"); kf = load_rows(kd, model, "fp16")
        if not cf or not kf:
            print(f"\n[{model}] MISSING fp16 rows (cult={len(cf)} ctrl={len(kf)}) — skip")
            continue
        print(f"\n### {model}")
        print(f"  cultural fp16 acc={acc(cf)*100:5.1f}  (n_inst={len(cf)}, items={len({k[0] for k in cf})})")
        print(f"  control  fp16 acc={acc(kf)*100:5.1f}  (n_inst={len(kf)}, items={len({k[0] for k in kf})})")
        for p in PRECS:
            cq = load_rows(cd, model, p); kq = load_rows(kd, model, p)
            if not cq or not kq:
                print(f"  [{p}] missing rows — skip"); continue
            ec, ek = erosion(cf, cq), erosion(kf, kq)
            b01c, c10c, chic, pc = mcnemar(cf, cq)
            b01k, c10k, chik, pk = mcnemar(kf, kq)
            point, lo, hi = diff_boot(cf, cq, kf, kq, rng)
            pooled[p].append(point)
            mde_c = mde_paired(len(cf.keys() & cq.keys()), (b01c + c10c) / max(1, len(cf.keys() & cq.keys())))
            print(f"  [{p}] cult_erosion={ec*100:+5.1f}pp (McNemar p={pc:.3f}, b/c={b01c}/{c10c})  "
                  f"ctrl_erosion={ek*100:+5.1f}pp (p={pk:.3f}, b/c={b01k}/{c10k})")
            print(f"        DIFFERENTIAL delta={point*100:+5.1f}pp  95%CI[{lo*100:+5.1f},{hi*100:+5.1f}]  "
                  f"paired-MDE~{mde_c*100:.1f}pp")
            if id_band:
                per = defaultdict(lambda: [0, 0])
                for (i, r), v in cf.items():
                    if (i, r) in cq:
                        per[id_band.get(i, "?")][0] += v - cq[(i, r)]
                        per[id_band.get(i, "?")][1] += 1
                bands = "  ".join(f"{b}:{(s/n)*100:+.0f}pp(n{n//5})" for b, (s, n) in sorted(per.items()) if n)
                print(f"        per-band cult erosion: {bands}")

    print("\n" + "=" * 78)
    print("MODEL-POOLED differential erosion (mean over models)")
    for p in PRECS:
        if pooled[p]:
            mean = sum(pooled[p]) / len(pooled[p])
            print(f"  {p}: mean delta = {mean*100:+.1f}pp over {len(pooled[p])} models  "
                  f"(per-model: {[round(x*100,1) for x in pooled[p]]})")
    print("\n[verdict] delta<0 & CI excludes 0 => rare BR culture erodes MORE than equally-rare")
    print("          generic (thesis). CI includes 0 with small MDE => conclusive NULL")
    print("          (int8/nf4 culturally safe; degradation is frequency-, not culture-driven).")


if __name__ == "__main__":
    main()
