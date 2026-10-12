import sys,pathlib,json,numpy as np
p=pathlib.Path(__file__).parent;sys.path.insert(0,str(p));import study
out={}
rng=np.random.default_rng(91)
c=np.stack([rng.uniform(.10,.24,20000),rng.uniform(.14,.30,20000),rng.uniform(.06,.18,20000),rng.uniform(.14,.30,20000)],1)
for cond in ('clean','det'):
 z=study.warp(c,cond);err=float(np.max(np.abs(study.inverse(z,cond)-c)));assert err<1e-12
 out[cond+'_inverse_parameter_max_abs']=err
z=study.warp(c,'rand',np.random.default_rng(10091));o=.5*(np.maximum(.10,z[:,0]-.025)+np.minimum(.24,z[:,0]+.025));estimate=z.copy();estimate[:,0]=o
# Score in chunks so the oracle uses bounded memory.
def chunk_score(a,b):
 rows=[study.score(a[i:i+500],b[i:i+500]) for i in range(0,len(a),500)]
 return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
out['random_onset_uniform_prior_bayes']={'n':len(c),'onset_mae_ms':float(np.abs(o-c[:,0]).mean()*1000),'onset_rmse_ms':float(np.sqrt(((o-c[:,0])**2).mean())*1000),'command':chunk_score(estimate,c),'nominal_replay':chunk_score(estimate,z),'fresh_replay':chunk_score(study.warp(estimate,'rand',np.random.default_rng(92)),z),'warning':'Parameter-squared-error Bayes oracle uses true executed timing, not video. It is not necessarily optimal for trajectory score. Chunk-mean p95 is descriptive only.'}
za=np.array([[.125,.14,.12,.16]]);zb=np.array([[.125,.14,.25,.16]])
ca=study.inverse(za,'det');cb=study.inverse(zb,'det')
fa,va=study.render_features(za,19);fb,vb=study.render_features(zb,19)
i=12;pa,ma=study.position(ca,[i/60]);pb,mb=study.position(cb,[i/60])
localdiff=float(np.abs(va[:,i-2:i+2]-vb[:,i-2:i+2]).max());xygap=float(np.linalg.norm(pa-pb));assert localdiff==0;assert xygap>.06
out['local_window_collision']={'command_a':ca.tolist(),'command_b':cb.tolist(),'frame':i,'time_s':i/60,'window':'t-2..t+1','local_pixel_maxdiff':localdiff,'whole_video_maxdiff':float(np.abs(va-vb).max()),'target_xy_distance':xygap,'minimum_max_pair_xy_error':xygap/2,'caveat':'Not proof against CurvedModel because its5D GroupNorm transmits global means/variances.'}
ca=np.array([[.15,.14,.18,.20]]);cb=np.array([[.15,.26,.06,.20]])
_,va=study.render_features(study.warp(ca,'collapse'),99);_,vb=study.render_features(study.warp(cb,'collapse'),99);assert np.array_equal(va,vb)
out['collapse_collision']={'identical_pixels':True,'move_and_hold_duration_delta_ms':120,'balanced_pair_per_duration_RMSE_lower_bound_ms':60}
(p/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
