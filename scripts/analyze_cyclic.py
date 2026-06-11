#!/usr/bin/env python3
"""Standalone verdict analyzer reading directly from the saved cyclic__*.jsonl
summaries (robust to the running grid; does not depend on rstd_table.json)."""
import json
import math
import glob
import os

LETTERS = ["A", "B", "C", "D", "E"]
# Defaults to the committed cyclic-bias outputs; override with CQ_CYCLIC_DIR.
OUT = os.environ.get(
    "CQ_CYCLIC_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results", "cyclic_bias"),
)

def binom_two_sided_p(k, n, p=0.5):
    if n == 0: return 1.0
    pmf = lambda i: math.comb(n, i) * p**i * (1-p)**(n-i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n+1) if pmf(i) <= obs + 1e-12))

def wilcoxon(diffs):
    nz = [d for d in diffs if abs(d) > 1e-12]; n = len(nz)
    if n == 0: return 0.0, 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(nz[i])); ranks=[0.0]*n; i=0
    while i < n:
        j=i
        while j+1<n and abs(abs(nz[order[j+1]])-abs(nz[order[i]]))<1e-12: j+=1
        avg=(i+1+j+1)/2.0
        for k in range(i,j+1): ranks[order[k]]=avg
        i=j+1
    wp=sum(ranks[i] for i in range(n) if nz[i]>0); wm=sum(ranks[i] for i in range(n) if nz[i]<0)
    W=min(wp,wm); mean=n*(n+1)/4.0; sd=math.sqrt(n*(n+1)*(2*n+1)/24.0)
    z=(W-mean)/sd if sd>0 else 0.0
    p=2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return W,z,min(1.0,p),n

def load():
    by={}
    for f in glob.glob(os.path.join(OUT,"cyclic__*.jsonl")):
        try:
            last=open(f,encoding="utf-8").read().splitlines()[-1]
            r=json.loads(last)
            if r.get("_summary"): by.setdefault(r["model"],{})[r["precision"]]=r
        except Exception: pass
    return by

def short(m): return m.split("/")[-1]

def main():
    by=load()
    order=["Qwen/Qwen2.5-0.5B-Instruct","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-3B-Instruct",
           "Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","Qwen/Qwen3-4B"]
    models=[m for m in order if m in by]+[m for m in by if m not in order]
    precs=["fp16","int8","nf4"]
    print("="*78)
    print("CYCLIC-ROTATION POSITION-BIAS: RStd table (std of chosen-position probs)")
    print("  0 = perfectly uniform/unbiased ; higher = more position concentration")
    print("="*78)
    print(f"{'model':<24}"+"".join(f"{p:>17}" for p in precs))
    for m in models:
        row=f"{short(m):<24}"
        for p in precs:
            r=by[m].get(p)
            row += (f"{r['rstd']:.3f} {r['favored_position']}{int(round(r['favored_rate']*100)):>2}%".rjust(17)) if r else f"{'-':>17}"
        print(row)
    print("\naccuracy (avg over rotations):")
    print(f"{'model':<24}"+"".join(f"{p:>17}" for p in precs))
    for m in models:
        row=f"{short(m):<24}"
        for p in precs:
            r=by[m].get(p); row += (f"{r['accuracy']:.3f}".rjust(17)) if r else f"{'-':>17}"
        print(row)

    print("\n"+"="*78)
    print("HYPOTHESIS TEST: does quantization INCREASE RStd vs fp16?")
    print("="*78)
    for label,qp in [("nf4 vs fp16","nf4"),("int8 vs fp16","int8")]:
        diffs=[]; rows=[]
        for m in models:
            f=by[m].get("fp16"); q=by[m].get(qp)
            if f and q:
                d=q["rstd"]-f["rstd"]; diffs.append(d)
                rows.append(f"  {short(m):<24} fp16={f['rstd']:.3f} {qp}={q['rstd']:.3f}  delta={d:+.3f}  {'INCREASE' if d>0 else 'decrease' if d<0 else 'flat'}")
        n=len(diffs); pos=sum(1 for d in diffs if d>1e-9); neg=sum(1 for d in diffs if d<-1e-9)
        print(f"\n--- {label}  (n={n} models) ---")
        for r in rows: print(r)
        if n:
            mean=sum(diffs)/n; ps=binom_two_sided_p(pos,pos+neg); W,z,pw,nw=wilcoxon(diffs)
            print(f"  increased(+)={pos}  decreased(-)={neg}  mean delta={mean:+.4f}")
            print(f"  sign test (two-sided) p={ps:.4f}   Wilcoxon z={z:.2f} p~={pw:.4f}")
            verdict = pos>neg and ps<0.05
            print(f"  -> consistent increase? {'YES' if pos==n else 'NO'} ; statistically significant increase? {'YES' if verdict else 'NO'}")

    # Histograms fp16 vs nf4 for the two clearest models
    print("\n"+"="*78); print("PREDICTED-POSITION HISTOGRAMS (fp16 vs nf4)"); print("="*78)
    for m in models:
        if "fp16" in by[m] and "nf4" in by[m]:
            print(f"\n{short(m)}:")
            for p in ("fp16","nf4"):
                pr=by[m][p]["pos_prob"]
                line=" ".join(f"{LETTERS[i]}:{pr[i]*100:4.1f}%" for i in range(5))
                print(f"  {p:<5} {line}")
                for i in range(5):
                    print(f"        {LETTERS[i]} "+"#"*int(round(pr[i]*40))+f" {pr[i]*100:.1f}%")

if __name__=="__main__":
    main()
