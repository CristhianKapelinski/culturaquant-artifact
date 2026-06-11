import glob
import json
import os
import re
import math
import statistics
import collections
OUT=os.path.expanduser("~/cq_data/out")
cband={}
for ln in open("/tmp/cq_cult_items.jsonl"):
    d=json.loads(ln); cband[d["id"]]=d["rarity"]["band"]
BANDS=["sl0-1","sl2-3","sl4-7","sl8-15","sl16-30","sl31-50","sl51-80"]
def wlo(k,n,z=1.96):
    p=k/n;d=1+z*z/n;return (p+z*z/(2*n))/d - z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
def iacc(path):
    agg=collections.defaultdict(list)
    for ln in open(path):
        try:o=json.loads(ln)
        except:continue
        if "correct" in o and "id" in o: agg[o["id"]].append(o["correct"])
    return {k:sum(v)/len(v) for k,v in agg.items()}
def L(m,p): return iacc(f"{OUT}/vAgrid_cult/cyclic__{m.replace('/','__')}__{p}.jsonl")
MODELS=sorted(re.search(r"cyclic__(.+?)__fp16",os.path.basename(f)).group(1).replace("__","/") for f in glob.glob(OUT+"/vAgrid_cult/cyclic__*__fp16.jsonl"))
meas=[m for m in MODELS if (lambda c: c and wlo(sum(c.values()),len(c))>0.20)(L(m,"fp16"))]
print("fp16 cultural accuracy by band, pooled over measurable models (calibration gradient):")
print("%-10s %5s %7s"%("band","n","fp16%"))
for b in BANDS:
    ids=[i for i in cband if cband[i]==b]; vals=[]
    for m in meas:
        c=L(m,"fp16")
        vals+=[c[i] for i in ids if i in c]
    print("%-10s %5d %7.1f"%(b,len(ids),statistics.mean(vals)*100))
# overall fp16 cult/ctrl per measurable model already have; print params guess
print("\nmeasurable models:",[m.split('/')[-1] for m in meas])
print("at-chance:",[m.split('/')[-1] for m in MODELS if m not in meas])
