import json
import os
import statistics
import collections
# Default to the committed run of record (data/predictions); override with CQ_OUT.
_HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get("CQ_OUT", os.path.join(_HERE,"..","data","predictions"))
SUBC=os.environ.get("CQ_SUB_CULT","cult")
meas=["Qwen2.5-1.5B-Instruct","Qwen2.5-3B-Instruct","Qwen2.5-7B-Instruct","Qwen3-1.7B","Qwen3-4B","Phi-3.5-mini-instruct"]
def rstd(path):
    cnt=collections.Counter()
    if not os.path.exists(path): return None
    for ln in open(path):
        try:o=json.loads(ln)
        except:continue
        if "pred" in o: cnt[o["pred"]]+=1
    tot=sum(cnt.values())
    if not tot: return None
    probs=[cnt.get(i,0)/tot for i in range(5)]
    return statistics.pstdev(probs)
fp=[]; nf=[]
for m in meas:
    base=OUT+"/"+SUBC+"/cyclic__Qwen__"+m+"__%s.jsonl" if m.startswith("Qwen") else OUT+"/"+SUBC+"/cyclic__microsoft__"+m+"__%s.jsonl"
    rf=rstd(base%"fp16"); rn=rstd(base%"nf4")
    if rf is not None: fp.append(rf)
    if rn is not None: nf.append(rn)
    print("%-24s fp16=%s nf4=%s"%(m,round(rf,3) if rf else "?",round(rn,3) if rn else "?"))
if fp: print("fp16 RStd range: %.3f to %.3f"%(min(fp),max(fp)))
