import json, numpy as np, sys
R="/home/nishioka/Tmcmc202601/data_5species/main/_runs/paper_gateoff/DS_p2_wide80_p8k_wide2_seed"
seeds=["42","7","123"]
runs={}
for s in seeds:
    d=R+s; rec=json.load(open(d+"/run_record.json"))
    S=np.load(d+"/samples.npy"); L=np.load(d+"/logL.npy")
    runs[s]=(rec,S,L)
names=runs["42"][0]["theta_names"]; free=runs["42"][0]["free_dims"]
print("free_dims:",free)
print("\n## MAP (max logL particle) per seed")
hdr="comp   "+"  ".join(f"s{s:>4}" for s in seeds)+"   range"
print(hdr)
maps={s:runs[s][1][np.argmax(runs[s][2])] for s in seeds}
for i in free:
    v=[maps[s][i] for s in seeds]
    print(f"{names[i]:5s} "+"  ".join(f"{x:+7.2f}" for x in v)+f"   {max(v)-min(v):5.2f}")
print("maxlogL "+"  ".join(f"{runs[s][2].max():+7.2f}" for s in seeds))
print("\n## 5/50/95% per seed for a45 a34 a24 a33 a35")
for nm in ["a45","a34","a24","a33","a35"]:
    i=names.index(nm)
    print(nm," | ".join(f"s{s}: {np.percentile(runs[s][1][:,i],5):+6.2f}/{np.percentile(runs[s][1][:,i],50):+6.2f}/{np.percentile(runs[s][1][:,i],95):+6.2f}" for s in seeds))
pool=np.vstack([runs[s][1] for s in seeds]); poolL=np.concatenate([runs[s][2] for s in seeds])
print("\n## pooled histograms (24000 particles), 30 bins")
for nm in ["a45","a34","a24"]:
    i=names.index(nm); x=pool[:,i]
    h,e=np.histogram(x,bins=30)
    print(f"\n{nm}  min {x.min():+.2f} max {x.max():+.2f}")
    mx=h.max()
    for c,lo in zip(h,e[:-1]):
        print(f"  {lo:+7.2f} {'#'*int(50*c/mx):50s} {c}")
    # mode split per seed at a crude threshold guess: look for lowest-density bin between peaks
print("\n## a45 per seed: fraction a45<0, a45>5; a34 fraction >0")
for s in seeds:
    S=runs[s][1]; L=runs[s][2]
    a45=S[:,names.index("a45")]; a34=S[:,names.index("a34")]
    print(f"s{s}: a45<0 {np.mean(a45<0):.2f}  0..5 {np.mean((a45>=0)&(a45<5)):.2f}  >5 {np.mean(a45>=5):.2f} | maxlogL a45<0 {L[a45<0].max() if (a45<0).any() else float('nan'):+.2f}  a45>=5 {L[a45>=5].max() if (a45>=5).any() else float('nan'):+.2f} | a34>0 {np.mean(a34>0):.2f}")
