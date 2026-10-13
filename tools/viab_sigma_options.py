#!/usr/bin/env python3
"""生きた割合（LIVE/DEAD）の σ の候補と、尤度の中での通路ごとの情報の割合（2026-10-13）。

割合 = λ_ch Σ w/σ²（組成は λ_Pg・終盤・珍しい種の重みつき）を 3 通路で足したうちの生きた割合の分。
候補: 今の σ（画像 SD の日平均、スカラー）、日ごとの画像 SD、SD/√3（バイオフィルム 3 個と仮定）、
SD/√n_画像（画像が独立と仮定）。出力は生データの数を含むので標準出力だけ（git に入れない）。
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.argv=['x','--device','cpu']; sys.path.insert(0,str(ROOT/'data_5species/main'))
import logging; logging.disable(logging.INFO)
import estimate_paper_jax as E
ld=pd.read_csv(ROOT/'data_5species/experiment_data/raw_heine/livedead_long.csv')
C={"CS":("Commensal","Static"),"CH":("Commensal","HOBIC"),"DS":("Dysbiotic","Static"),"DH":("Dysbiotic","HOBIC")}
L3={"CS":1.5,"CH":1.5,"DS":2.0,"DH":3.0}
def info(src,tag,sv):
    D=E.DATA_DIRS[src]; c,cu=C[tag]
    data,t,sig,_,_=E.load_experimental_data(D,c,cu,1,normalize=True,use_exp_init=True)
    mc=E.load_multichannel_data(data_dir=D/"experiment_data",condition=c,cultivation=cu,days_filter=t.tolist())
    W=np.ones(data.shape); W[:,4]*=5; W[-2:]*=3; W*=np.where(data.mean(0)<0.05,0.1,1)[None,:]
    s=np.broadcast_to(np.asarray(sig,float),data.shape); st=np.broadcast_to(np.asarray(mc["sigma_obs_total"],float),data.shape)
    i1=np.sum(W/s**2); i2=np.sum(W/st**2)
    sv=mc["sigma_obs_viability"] if sv is None else np.asarray(sv)
    i3=L3[tag]*np.sum(np.broadcast_to(1/np.asarray(sv)**2,(len(t),)))
    return i1,i2,i3,t
rows=[]
for tag,(c,cu) in C.items():
    i1d,i2d,i3d,_=info("digitised",tag,None)
    i1,i2,i3,t=info("raw",tag,None)
    s=ld[(ld.condition==c)&(ld.cultivation==cu)&(ld.day.isin(t))].groupby("day").live_pct
    sd=s.std().values/100; n=s.size().values
    _,_,i3b,_=info("raw",tag,sd)                 # 日ごとの画像 SD
    _,_,i3c,_=info("raw",tag,sd/np.sqrt(n))      # 平均の標準誤差（画像が独立なら）
    _,_,i3d3,_=info("raw",tag,sd/np.sqrt(3))     # バイオフィルム 3 個と仮定
    f=lambda a,b,cc: f"{cc/(a+b+cc)*100:.1f}%"
    rows.append([tag, f(i1d,i2d,i3d), f(i1,i2,i3), f(i1,i2,i3b), f(i1,i2,i3d3), f(i1,i2,i3c),
                 round(float(np.mean(sd)),3), round(float(np.mean(sd/np.sqrt(n))),3), int(n.min()), int(n.max())])
df=pd.DataFrame(rows,columns=["cond","share: digitised now","raw now (mean image SD)","raw per-day image SD","raw SD/sqrt(3 biofilms)","raw SE (images indep.)","image SD","SE","n_img min","n_img max"])
print(df.to_markdown(index=False))
