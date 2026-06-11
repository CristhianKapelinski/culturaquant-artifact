import glob
import json
import os
import re
import math
import statistics
import random
import collections
random.seed(20260609)
# Default to the committed run of record (v2/allband_predictions); override with CQ_OUT.
_HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("CQ_OUT", os.path.join(_HERE,"..","allband_predictions"))
SUB={"cult":os.environ.get("CQ_SUB_CULT","cult"),"ctrl":os.environ.get("CQ_SUB_CTRL","ctrl")}
CULT_ITEMS=os.environ.get("CQ_CULT_ITEMS", os.path.join(OUT,"cq_cult_items.jsonl"))
CTRL_ITEMS=os.environ.get("CQ_CTRL_ITEMS", os.path.join(OUT,"cq_ctrl_items.jsonl"))
cband={};creg={}
for ln in open(CULT_ITEMS):
    d=json.loads(ln); cband[d["id"]]=d["rarity"]["band"]; creg[d["id"]]=d["region"]
cult2ctrl={}
for ln in open(CTRL_ITEMS):
    d=json.loads(ln)
    if d.get("matched_br_id"): cult2ctrl[d["matched_br_id"]]=d["id"]
BANDS=["sl0-1","sl2-3","sl4-7","sl8-15","sl16-30","sl31-50","sl51-80"]
def wlo(k,n,z=1.96):
    if n==0:return 0
    p=k/n;d=1+z*z/n;return (p+z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
def iacc(path):
    agg=collections.defaultdict(list)
    if os.path.exists(path):
        for ln in open(path):
            try:o=json.loads(ln)
            except:continue
            if "correct" in o and "id" in o: agg[o["id"]].append(o["correct"])
    return {k:sum(v)/len(v) for k,v in agg.items()}
def L(m,p,w): return iacc(f"{OUT}/{SUB[w]}/cyclic__{m.replace('/','__')}__{p}.jsonl")
MODELS=sorted(re.search(r"cyclic__(.+?)__fp16",os.path.basename(f)).group(1).replace("__","/") for f in glob.glob(OUT+"/"+SUB["cult"]+"/cyclic__*__fp16.jsonl"))
# measurable models
meas=[]; perm={}
for m in MODELS:
    cfp=L(m,"fp16","cult")
    if cfp and wlo(sum(cfp.values()),len(cfp))>0.20: meas.append(m)
def pairdiffs(m,prec):
    cfp=L(m,"fp16","cult");rfp=L(m,"fp16","ctrl");cq=L(m,prec,"cult");rq=L(m,prec,"ctrl")
    out=[]
    for cid in cfp:
        ctid=cult2ctrl.get(cid)
        if ctid is None or ctid not in rfp or ctid not in rq or cid not in cq: continue
        out.append(((cfp[cid]-cq[cid])-(rfp[ctid]-rq[ctid]), cid))
    return out
def boot_ci(vals,B=10000):
    if not vals:return(0,0,0)
    n=len(vals);ms=sorted(sum(vals[random.randrange(n)] for _ in range(n))/n for _ in range(B))
    return (statistics.mean(vals),ms[int(.025*B)],ms[int(.975*B)])
print("=== Per measurable model: differential erosion (pp) with 95%% CI ===")
print("%-24s %18s %18s"%("model","int8 diff [CI]","nf4 diff [CI]"))
allp={"int8":[],"nf4":[]}; mdes=[]
for m in meas:
    line=m.split("/")[-1]
    cell={}
    for prec in ["int8","nf4"]:
        pd=[x[0] for x in pairdiffs(m,prec)]
        mu,lo,hi=boot_ci(pd)
        cell[prec]=(mu*100,lo*100,hi*100)
        allp[prec].append((m,mu))
        if prec=="int8": mdes.append((hi-lo)/2*100)
    print("%-24s %6.1f [%5.1f,%5.1f] %6.1f [%5.1f,%5.1f]"%(line,*cell["int8"],*cell["nf4"]))
# pooled (per-model mean) + CI via pooling all pairs across measurable models
for prec in ["int8","nf4"]:
    allpairs=[x[0] for m in meas for x in pairdiffs(m,prec)]
    mu,lo,hi=boot_ci(allpairs)
    print("POOLED %s: %+.1f pp [%.1f, %.1f]"%(prec,mu*100,lo*100,hi*100))
print("MDE (widest int8 per-model CI half-width): %.1f pp"%max(mdes))
# per-band cult drop pooled over measurable, int8 & nf4
print("\n=== cultural accuracy DROP by rarity band (pooled measurable, pp) ===")
print("%-10s %6s %8s %8s"%("band","n","int8 drop","nf4 drop"))
for b in BANDS:
    ids=[cid for cid in cband if cband[cid]==b]
    di={"int8":[],"nf4":[]}
    for m in meas:
        cfp=L(m,"fp16","cult")
        for prec in ["int8","nf4"]:
            cq=L(m,prec,"cult")
            for cid in ids:
                if cid in cfp and cid in cq: di[prec].append(cfp[cid]-cq[cid])
    print("%-10s %6d %8.1f %8.1f"%(b,len(ids),statistics.mean(di["int8"])*100 if di["int8"] else 0,statistics.mean(di["nf4"])*100 if di["nf4"] else 0))
# per-region cult drop (int8, nf4)
print("\n=== cultural drop by region (pooled measurable, pp) ===")
for grp,regs in [("N+NE",["N","NE"]),("CO",["CO"]),("SE+S",["SE","S"])]:
    ids=[cid for cid in creg if creg[cid] in regs]
    di={"int8":[],"nf4":[]}
    for m in meas:
        cfp=L(m,"fp16","cult")
        for prec in ["int8","nf4"]:
            cq=L(m,prec,"cult")
            for cid in ids:
                if cid in cfp and cid in cq: di[prec].append(cfp[cid]-cq[cid])
    print("  %-6s n=%4d  int8 %+.1f  nf4 %+.1f"%(grp,len(ids),statistics.mean(di["int8"])*100,statistics.mean(di["nf4"])*100))
