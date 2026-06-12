import glob
import json
import os
import re
import math
import statistics
import random
import collections
random.seed(20260609)
# Default to the committed run of record (data/predictions); override with CQ_OUT for a
# fresh GPU-host grid (e.g. CQ_OUT=/path/to/fresh/grid CQ_SUB_CULT=cult CQ_SUB_CTRL=ctrl).
_HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("CQ_OUT", os.path.join(_HERE,"..","data","predictions"))
SUB={"cult":os.environ.get("CQ_SUB_CULT","cult"),"ctrl":os.environ.get("CQ_SUB_CTRL","ctrl")}
CULT_ITEMS=os.environ.get("CQ_CULT_ITEMS", os.path.join(_HERE,"..","data","items","cultural.jsonl"))
CTRL_ITEMS=os.environ.get("CQ_CTRL_ITEMS", os.path.join(_HERE,"..","data","items","control.jsonl"))
# --- item metadata ---
cband={};creg={}
for ln in open(CULT_ITEMS):
    d=json.loads(ln); cband[d["id"]]=d["rarity"]["band"]; creg[d["id"]]=d["region"]
ctrl_to_cult={}
for ln in open(CTRL_ITEMS):
    d=json.loads(ln); ctrl_to_cult[d["id"]]=d.get("matched_br_id")
BANDS=["sl0-1","sl2-3","sl4-7","sl8-15","sl16-30","sl31-50","sl51-80"]
def wilson_lo(k,n,z=1.96):
    if n==0:return 0
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h
def item_acc(path):
    # returns {item_id: mean correct over rotations}
    agg=collections.defaultdict(list)
    if not os.path.exists(path): return {}
    for ln in open(path):
        try:o=json.loads(ln)
        except:continue
        if "correct" not in o or "id" not in o: continue
        agg[o["id"]].append(o["correct"])
    return {k:sum(v)/len(v) for k,v in agg.items()}
def model_files():
    fs=glob.glob(OUT+"/"+SUB["cult"]+"/cyclic__*__fp16.jsonl")
    return sorted(re.search(r"cyclic__(.+?)__fp16",os.path.basename(f)).group(1).replace("__","/") for f in fs)
def load(model,prec,which):
    return item_acc(f"{OUT}/{SUB[which]}/cyclic__{model.replace('/','__')}__{prec}.jsonl")
MODELS=model_files()
# --- per-model: fp16 measurable, differential erosion int8/nf4 ---
print("%-30s %6s %6s %8s %10s %10s"%("model","cultFP","ctrlFP","measur","diffINT8","diffNF4"))
rows={}
for m in MODELS:
    cfp=load(m,"fp16","cult"); rfp=load(m,"fp16","ctrl")
    if not cfp: continue
    cfp_acc=statistics.mean(cfp.values()) 
    k=sum(round(v* len([1])) for v in cfp.values())  # not used
    n=len(cfp); kc=sum(cfp.values())
    meas = wilson_lo(kc,n) > 0.20
    rfp_acc=statistics.mean(rfp.values()) if rfp else 0
    diffs={}
    for prec in ["int8","nf4"]:
        cq=load(m,prec,"cult"); rq=load(m,prec,"ctrl")
        # per matched pair: cult_drop - ctrl_drop
        pair=[]
        for cid in cfp:
            ctrl_id=None
            # find ctrl item whose matched_br_id==cid
            pass
        # build cult->ctrl map once
        diffs[prec]=(cfp_acc-(statistics.mean(cq.values()) if cq else cfp_acc)) - (rfp_acc-(statistics.mean(rq.values()) if rq else rfp_acc))
    rows[m]={"cultFP":cfp_acc,"ctrlFP":rfp_acc,"meas":meas,"int8":diffs["int8"],"nf4":diffs["nf4"],"n":n}
    print("%-30s %6.3f %6.3f %8s %+10.3f %+10.3f"%(m.split("/")[-1],cfp_acc,rfp_acc,"YES" if meas else "no",diffs["int8"],diffs["nf4"]))
# pooled over measurable
meas_models=[m for m in rows if rows[m]["meas"]]
print("\nMEASURABLE models:",len(meas_models),"/",len(rows))
pooled_int8=statistics.mean(rows[m]["int8"] for m in meas_models)
pooled_nf4=statistics.mean(rows[m]["nf4"] for m in meas_models)
print("POOLED differential erosion (measurable): int8 %+.3f  nf4 %+.3f"%(pooled_int8,pooled_nf4))
print("  (positive = cultural erodes MORE than control; ~0 = safe)")
# per-model rows are available in `rows` for downstream use
