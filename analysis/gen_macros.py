M=[]
def c(n,v): M.append("\\newcommand{\\%s}{%s}"%(n,v))
# counts
c("nModels","11"); c("nMeas","6"); c("nArm","700"); c("nRot","5"); c("nPairs","700")
# calibration gradient fp16% by band (pooled measurable)
for k,v in zip("abcdefg",[21.4,20.2,26.0,28.0,35.6,44.0,52.1]): c("CalibSL"+k,"%.1f"%v)
# per measurable model: (tag, params, cultFP, ctrlFP, dInt, dIntLo, dIntHi, dNf, dNfLo, dNfHi)
MM=[("Qwfifteen","1.5B",25.0,20.8,-0.2,-1.1,0.7, 3.5,1.4,5.6),
    ("Qwthree","3B",27.7,22.2, 0.8,-1.0,2.6, 3.1,0.9,5.3),
    ("Qwseven","7B",30.6,23.8, 0.2,-0.8,1.2, 3.1,1.4,4.8),
    ("Qttwo","1.7B",24.7,29.1,-0.7,-2.2,0.7,-3.6,-5.9,-1.2),
    ("Qtfour","4B",26.9,20.6, 0.6,-0.7,1.7, 0.0,-1.8,1.8),
    ("Phi","3.8B",24.9,19.9, 0.5,-1.1,2.1,-1.7,-3.4,0.1)]
for t,p,cf,rf,di,dil,dih,dn,dnl,dnh in MM:
    c(t+"P",p); c(t+"CultFp","%.1f"%cf); c(t+"CtrlFp","%.1f"%rf)
    c(t+"DiffInt","%+.1f"%di); c(t+"DiffIntLo","%.1f"%dil); c(t+"DiffIntHi","%.1f"%dih)
    c(t+"DiffNf","%+.1f"%dn); c(t+"DiffNfLo","%.1f"%dnl); c(t+"DiffNfHi","%.1f"%dnh)
# pooled + MDE
c("PooledDiffInt","+0.2"); c("PooledDiffIntLo","-0.4"); c("PooledDiffIntHi","0.7")
c("PooledDiffNf","+0.7"); c("PooledDiffNfLo","-0.1"); c("PooledDiffNfHi","1.6")
c("mde","1.8")
# nf4 spread
c("nfPosModels","three"); c("nfPosLo","3.1"); c("nfPosHi","3.5")  # Qwen2.5 family erode cultural MORE
c("nfNegModel","Qwen3-1.7B"); c("nfNegVal","-3.6")
# per-band cultural drop (int8, nf4) - shows no growth with rarity
c("BandIntMin","-1.0"); c("BandIntMax","1.3")        # int8 flat range
c("BandNfMin","0.0"); c("BandNfMax","5.9")            # nf4 scattered, not monotone with rarity
# per-band ABSOLUTE cultural accuracy drop (pp), pooled measurable; bands sl0-1..sl51-80
# (cf. analysis/cq_ci.py "cultural accuracy DROP by rarity band"; counts in results/full_analysis.txt)
for k,n,vi,vf in zip("abcdefg",[153,150,146,123,78,36,14],
                     [-0.5,-0.5,0.5,0.4,-0.2,1.3,-1.0],
                     [1.5,0.0,2.1,3.1,1.5,5.9,3.8]):
    c("BandN"+k,str(n)); c("BandDInt"+k,"%+.1f"%vi); c("BandDNf"+k,"%+.1f"%vf)
# per-region ABSOLUTE cultural drop (pp) and item counts
c("RegIntMin","-0.7"); c("RegIntMax","0.3")
c("RegNfNNE","+1.1"); c("RegNfCO","-0.1"); c("RegNfSES","+3.2")
c("RegIntNNE","-0.1"); c("RegIntCO","-0.7"); c("RegIntSES","+0.3")
c("RegNnne","271"); c("RegNco","113"); c("RegNses","316")
# at-chance models
c("chanceModels","Qwen2.5-0.5B, Qwen3-0.6B, Mistral-7B, and both Tucano models")
# total scored instances = 700 items x 5 cyclic rotations
c("nInst","3500")
# cultural items per macro-region (sum=700); see data/items/cultural.jsonl
c("regN","105"); c("regNE","166"); c("regCO","113"); c("regSE","179"); c("regS","137")
# position-choice spread (RStd) for measurable models; see results/rstd_measurable.json (analysis/cq_rstd.py)
c("RstdLo","0.12"); c("RstdHi","0.20"); c("RstdNfHigh","0.33")
import os
import sys
# Default output: the committed reference copy in results/. Pass a path as argv[1]
# (or set CQ_MACRO_OUT) to write elsewhere, e.g. the paper's results_macros.tex.
_HERE=os.path.dirname(os.path.abspath(__file__))
DEFAULT=os.path.join(_HERE,"..","results","results_macros.tex")
out=sys.argv[1] if len(sys.argv)>1 else os.environ.get("CQ_MACRO_OUT",DEFAULT)
open(out,"w").write(
 "% AUTO-GENERATED, all-band 11-model grid (66 runs). Real numbers; see results/full_analysis.txt.\n"
 "% Differential erosion = cultural accuracy drop minus rarity-matched control drop (pp). 95%% item-clustered bootstrap CI, seed 20260609.\n"
 +"\n".join(M)+"\n")
print("wrote",len(M),"macros ->",out)
