"""CPU-only timing IDM toy. Fixed bent path, command=o,d_move1,d_hold,d_move2.
Rendered video -> intensity moments -> MLP -> command, without true executor latents.
Independent seed streams for commands, executor noise, pixels, and fresh replay.
No decoding or hold-detection heuristic: the command representation is fixed.
"""
import os,time,json,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn

FPS,T,H=60,72,16
FT=np.arange(T)/FPS
P=np.array([[.2,.25],[.75,.25],[.75,.75]],np.float32)
Y,X=np.mgrid[:H,:H].astype(np.float32)
X=(X+.5)/H;Y=(Y+.5)/H

def warp(c,cond,rng=None):
 z=np.array(c,copy=True)
 if cond in ('det','detnoise'):
  d=1.2*c[:,1];h=c[:,2]
  z[:,0]+=.025;z[:,1]=d*d/(d+h);z[:,2]=d+h-z[:,1];z[:,3]*=.8
  if cond=='detnoise' and rng is not None:
   z[:,0]+=rng.normal(0,.005,len(c));z[:,1:]*=1+rng.normal(0,.03,(len(c),3))
 elif cond=='rand' and rng is not None:z[:,0]+=rng.uniform(-.025,.025,len(c))
 elif cond=='collapse':
  z[:,1]=.05;z[:,2]=c[:,1]+c[:,2]-.05
 return z

def inverse(z,cond):
 c=z.copy()
 if cond in ('det','detnoise'):
  s=z[:,1]+z[:,2];d=np.sqrt(z[:,1]*s)
  c[:,0]-=.025;c[:,1]=d/1.2;c[:,2]=s-d;c[:,3]/=.8
 return c

def position(c,t):
 # n commands x q times; positions held at start/end outside contact, masked separately
 t=np.asarray(t)[None,:];o,d,h,e=c.T
 u=t-o[:,None]
 a=np.clip(u/d[:,None],0,1)
 b=np.clip((u-d[:,None]-h[:,None])/e[:,None],0,1)
 xy=P[0]+a[:,:,None]*(P[1]-P[0])+b[:,:,None]*(P[2]-P[1])
 mask=(u>=0)&(u<=(d+h+e)[:,None])
 return xy,mask

def render_features(z,seed):
 xy,mask=position(z,FT)
 # 16x16 gaussian blob plus 6-frame fading trail; 60Hz native image sequence
 v=np.exp(-((X[None,None]-xy[:,:,0,None,None])**2+(Y[None,None]-xy[:,:,1,None,None])**2)/(2*.035**2))
 v*=mask[:,:,None,None]
 raw=v.copy()
 for k in range(1,7):v[:,k:]=np.maximum(v[:,k:],(.6*(1-k/7))*raw[:,:-k])
 rng=np.random.default_rng(seed);v+=rng.normal(0,.005,v.shape);v=np.clip(v,0,1).astype(np.float32)
 # No ground-truth coordinates/times enter feature extraction. High-order intensity
 # moments suppress the lower-amplitude tail. Whole-clip context is intentionally generous.
 w=v**4;mass=w.sum((2,3))+1e-7
 fx=(w*X).sum((2,3))/mass;fy=(w*Y).sum((2,3))/mass
 features=np.stack([fx,fy,mass/4,v.max((2,3))],-1).reshape(len(z),-1)
 return features.astype(np.float32),v

def data(n,seed,cond):
 rng=np.random.default_rng(seed)
 c=np.stack([rng.uniform(.10,.24,n),rng.uniform(.14,.30,n),rng.uniform(.06,.18,n),rng.uniform(.14,.30,n)],1)
 z=warp(c,cond,np.random.default_rng(seed+10000))
 f,v=render_features(z,seed+20000)
 return c,z,f

def fit(x,y,seed,epochs):
 torch.manual_seed(seed)
 xm=x.mean(0);xs=x.std(0).clip(.05);ym=y.mean(0);ys=y.std(0)
 x=torch.tensor((x-xm)/xs);y=torch.tensor((y-ym)/ys,dtype=torch.float32)
 m=nn.Sequential(nn.Linear(x.shape[1],192),nn.SiLU(),nn.Linear(192,96),nn.SiLU(),nn.Linear(96,4))
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
 g=torch.Generator().manual_seed(seed+123)
 for ep in range(epochs):
  for idx in torch.randperm(len(x),generator=g).split(128):
   loss=((m(x[idx])-y[idx])**2).mean();opt.zero_grad();loss.backward();opt.step()
  if ep in (0,epochs-1):print('epoch',ep,'loss',float(loss.detach()),flush=True)
 def predict(f):
  with torch.no_grad():
   p=m(torch.tensor((f-xm)/xs)).numpy()*ys+ym
  p[:,0]=np.clip(p[:,0],0,.5);p[:,1:]=np.clip(p[:,1:],.02,.6)
  return p
 return predict

def score(a,b):
 tm=np.arange(T*4)/(FPS*4)
 ap,am=position(a,tm);bp,bm=position(b,tm)
 dist=np.linalg.norm(ap-bp,axis=-1);both=am&bm;union=am|bm
 good=((dist<=.03)&both).sum(1)/union.sum(1)
 overlap_error=(dist*both).sum(1)/both.sum(1).clip(1)
 bounds_a=np.stack([a[:,0],a.sum(1)],1);bounds_b=np.stack([b[:,0],b.sum(1)],1)
 b_err=np.abs(bounds_a-bounds_b)*1000
 passed=(good>=.95)&(b_err.max(1)<=2/FPS*1000)
 return dict(xy_mae=float(overlap_error.mean()),xy_mae_p95=float(np.quantile(overlap_error,.95)),
             pass_rate=float(passed.mean()),contact_disagree=float(((am^bm).sum(1)/union.sum(1)).mean()),
             boundary_mae_ms=float(b_err.mean()),good_fraction=float(good.mean()))

def run(seed,cond,ntrain,ntest,epochs):
 c,z,x=data(ntrain,seed*100+1,cond);ct,zt,xt=data(ntest,seed*100+2,cond)
 pred=fit(x,c,seed,epochs)(xt)
 nominal=warp(pred,cond);fresh=warp(pred,cond,np.random.default_rng(seed*100+30000))
 truthfresh=warp(ct,cond,np.random.default_rng(seed*100+40000))
 out=dict(seed=seed,condition=cond,command=score(pred,ct),nominal_replay=score(nominal,zt),
          fresh_replay=score(fresh,zt),repeat_same_true_command=score(truthfresh,zt),
          raw_trace_vs_command=score(ct,zt),human_raw_error=score(pred,zt),
          timing_mae_ms=(np.abs(pred-ct).mean(0)*1000).tolist(),timing_rmse_ms=(np.sqrt(((pred-ct)**2).mean(0))*1000).tolist(),
          ideal_nominal_inverse_replay=score(warp(inverse(zt,cond),cond),zt))
 if cond=='collapse':
  # Explicit balanced pair: durations .14+.18 versus .26+.06, same total; identical pixels.
  ca=np.array([[.15,.14,.18,.2]]);cb=np.array([[.15,.26,.06,.2]])
  za,zb=warp(ca,cond),warp(cb,cond)
  _,va=render_features(za,999);_,vb=render_features(zb,999)
  out['collision']=dict(max_pixel_diff=float(np.abs(va-vb).max()),command_duration_difference_ms=120,
                        balanced_pair_min_duration_rmse_ms=60,observed_timing_diff=float(np.max(np.abs(za-zb))))
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ntrain',type=int,default=2400);p.add_argument('--ntest',type=int,default=600)
 p.add_argument('--epochs',type=int,default=70);p.add_argument('--seeds',default='0,1,2');p.add_argument('--conditions',default='clean,det,rand,detnoise,collapse')
 a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.time();rows=[]
 for seed in map(int,a.seeds.split(',')):
  for cond in a.conditions.split(','):
   t=time.time();print('RUN',seed,cond,flush=True)
   row=run(seed,cond,a.ntrain,a.ntest,a.epochs);row['seconds']=time.time()-t;rows.append(row)
   Path(__file__).with_name('results.json').write_text(json.dumps(dict(config=vars(a),results=rows,elapsed_s=time.time()-start),indent=2))
   print('RESULT',json.dumps(row),flush=True)
 print('DONE',time.time()-start,flush=True)
