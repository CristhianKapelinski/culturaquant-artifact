import json
import random
from pathlib import Path
from collections import defaultdict
R = Path("results_final")
MODELS = ["Qwen__Qwen2.5-7B-Instruct","mistralai__Mistral-7B-Instruct-v0.3","microsoft__Phi-3.5-mini-instruct"]
MNAME = {"Qwen__Qwen2.5-7B-Instruct":"Qwen2.5-7B","mistralai__Mistral-7B-Instruct-v0.3":"Mistral-7B","microsoft__Phi-3.5-mini-instruct":"Phi-3.5-mini"}
PREC = ["fp16","int8","nf4"]
rng = random.Random(20260609)

def load(setdir, model, prec):
    p = R/setdir/f"cyclic__{model}__{prec}.jsonl"
    rows=[]
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        d=json.loads(ln)
        if d.get("_summary"): continue
        rows.append(d)
    return rows

def acc(rows, rot=None):
    rs=[r for r in rows if (rot is None or r["rotation"]==rot)]
    return sum(r["correct"] for r in rs)/len(rs) if rs else float("nan")

def rstd(rows):
    c=[0]*5
    for r in rows: c[r["pred"]]+=1
    n=sum(c); p=[x/n for x in c]; m=sum(p)/5
    return (sum((x-m)**2 for x in p)/5)**0.5

def by_item(rows, rot=None):
    m=defaultdict(lambda:[0,0])
    for r in rows:
        if rot is not None and r["rotation"]!=rot: continue
        m[r["id"]][0]+=r["correct"]; m[r["id"]][1]+=1
    return m

def diff_boot(cf,cq,kf,kq,rot=None):
    # item-clustered bootstrap of differential erosion (cult_drop - ctrl_drop)
    cmf=by_item(cf,rot); cmq=by_item(cq,rot); kmf=by_item(kf,rot); kmq=by_item(kq,rot)
    cids=[i for i in cmf if i in cmq]; kids=[i for i in kmf if i in kmq]
    def delta(ci,ki):
        cfd=sum(cmf[i][0] for i in ci); cfn=sum(cmf[i][1] for i in ci)
        cqd=sum(cmq[i][0] for i in ci) 
        kfd=sum(kmf[i][0] for i in ki); kfn=sum(kmf[i][1] for i in ki)
        kqd=sum(kmq[i][0] for i in ki)
        return (cfd-cqd)/cfn - (kfd-kqd)/kfn
    point=delta(cids,kids)
    bs=[]
    for _ in range(4000):
        cs=[cids[rng.randrange(len(cids))] for _ in cids]
        ks=[kids[rng.randrange(len(kids))] for _ in kids]
        bs.append(delta(cs,ks))
    bs.sort()
    return point, bs[int(.025*len(bs))], bs[int(.975*len(bs))]

out={"models":{},"pooled":{},"calib":{},"ablation":{}}
for m in MODELS:
    rows={(s,p):load(s,m,p) for s in ["vFgrid_cult","vFgrid_ctrl"] for p in PREC}
    md={"name":MNAME[m]}
    for p in PREC:
        md[f"cult_{p}"]=round(acc(rows[("vFgrid_cult",p)])*100,1)
        md[f"ctrl_{p}"]=round(acc(rows[("vFgrid_ctrl",p)])*100,1)
        md[f"cult_{p}_A"]=round(acc(rows[("vFgrid_cult",p)],rot=0)*100,1)
        md[f"ctrl_{p}_A"]=round(acc(rows[("vFgrid_ctrl",p)],rot=0)*100,1)
        md[f"rstd_cult_{p}"]=round(rstd(rows[("vFgrid_cult",p)]),3)
    for q in ["int8","nf4"]:
        pt,lo,hi=diff_boot(rows[("vFgrid_cult","fp16")],rows[("vFgrid_cult",q)],rows[("vFgrid_ctrl","fp16")],rows[("vFgrid_ctrl",q)])
        md[f"diff_{q}"]=[round(pt*100,1),round(lo*100,1),round(hi*100,1)]
        ptA,loA,hiA=diff_boot(rows[("vFgrid_cult","fp16")],rows[("vFgrid_cult",q)],rows[("vFgrid_ctrl","fp16")],rows[("vFgrid_ctrl",q)],rot=0)
        md[f"diff_{q}_A"]=[round(ptA*100,1),round(loA*100,1),round(hiA*100,1)]
    out["models"][m]=md
# pooled differential
for q in ["int8","nf4"]:
    out["pooled"][q]=round(sum(out["models"][m][f"diff_{q}"][0] for m in MODELS)/3,1)
    out["pooled"][q+"_A"]=round(sum(out["models"][m][f"diff_{q}_A"][0] for m in MODELS)/3,1)
# calibration
cal=json.load(open(R/"wide_calibration_table.json"))
for s in cal["results"]:
    out["calib"]=s["per_bucket_acc"]
print(json.dumps(out,indent=2,ensure_ascii=False))
Path("results_final/paper_numbers.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))

# ---- emit results_macros.tex (every paper number lives here) ----
def pp(x): 
    s=f"{x:+.1f}"; return s
def ci(t): return f"[{t[1]:+.1f}, {t[2]:+.1f}]"
L=[]
A=L.append
A("% AUTO-GENERATED from artifact/v2/results_final by analysis_paper.py. Do not edit by hand.")
A("\\newcommand{\\nArm}{128}")
A("\\newcommand{\\nModels}{3}")
A("\\newcommand{\\nRot}{5}")
# calibration gradient
cal=out["calib"]
cmap={"sl0-1":"CalibSLa","sl2-3":"CalibSLb","sl4-7":"CalibSLc","sl8-15":"CalibSLd","sl16-30":"CalibSLe","sl31-50":"CalibSLf","sl51-80":"CalibSLg"}
for k,mc in cmap.items():
    A(f"\\newcommand{{\\{mc}}}{{{cal[k][0]*100:.0f}}}")
# per-model
short={"Qwen__Qwen2.5-7B-Instruct":"Qwen","mistralai__Mistral-7B-Instruct-v0.3":"Mistral","microsoft__Phi-3.5-mini-instruct":"Phi"}
for mk,sh in short.items():
    d=out["models"][mk]
    for p in ["fp16","int8","nf4"]:
        A(f"\\newcommand{{\\{sh}Cult{p.capitalize()}}}{{{d['cult_'+p]:.1f}}}")
        A(f"\\newcommand{{\\{sh}Ctrl{p.capitalize()}}}{{{d['ctrl_'+p]:.1f}}}")
        A(f"\\newcommand{{\\{sh}CultA{p.capitalize()}}}{{{d['cult_'+p+'_A']:.1f}}}")
        A(f"\\newcommand{{\\{sh}Rstd{p.capitalize()}}}{{{d['rstd_cult_'+p]:.3f}}}")
    for q in ["int8","nf4"]:
        t=d['diff_'+q]
        A(f"\\newcommand{{\\{sh}Diff{q.capitalize()}}}{{{t[0]:+.1f}}}")
        A(f"\\newcommand{{\\{sh}Diff{q.capitalize()}CIlo}}{{{t[1]:+.1f}}}")
        A(f"\\newcommand{{\\{sh}Diff{q.capitalize()}CIhi}}{{{t[2]:+.1f}}}")
# pooled
A(f"\\newcommand{{\\PooledDiffIntEight}}{{{out['pooled']['int8']:+.1f}}}")
A(f"\\newcommand{{\\PooledDiffNfFour}}{{{out['pooled']['nf4']:+.1f}}}")
A(f"\\newcommand{{\\PooledDiffIntEightA}}{{{out['pooled']['int8_A']:+.1f}}}")
A(f"\\newcommand{{\\PooledDiffNfFourA}}{{{out['pooled']['nf4_A']:+.1f}}}")
Path("../main_results_macros.tex").write_text("\n".join(L)+"\n")
print("\n".join(L))
print("\n[wrote] papers/culturaquant/main_results_macros.tex" )
