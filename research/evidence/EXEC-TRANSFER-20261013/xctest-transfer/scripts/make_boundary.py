import json,pathlib
from probes import chain,split
ROOT=pathlib.Path(__file__).resolve().parents[1];P={}
def put(name,pts,durs,kind='boundary'):
 p,cmd=chain(pts,durs);P[name]={'payload':p,'cmd':cmd,'kind':kind}
for d in (15,16,17,20,25,33):
 for n in (1,2,10,20):
  put(f'boundary_d{d}_n{n}',[(round(107+200*k/n),448) for k in range(n+1)],[d]*n)
 for n in (2,10):
  put(f'boundary_zig_d{d}_n{n}',[(107 if k%2==0 else 307,448) for k in range(n+1)],[d]*n)
for n in (17,18,19,20):
 put(f'boundary_total300_n{n}',[(round(107+200*k/n),448) for k in range(n+1)],split(300,n))
for ds in ([15,100],[100,15],[15,100,15],[100,15,100],[16,17]*10,[17,16]*10,[100,16,100],[100,17,100]):
 put('mixed_'+'_'.join(map(str,ds)),[(107 if k%2==0 else 307,448) for k in range(len(ds)+1)],ds,'mixed')
for h in (15,16,17):
 put(f'boundary_hold_mid{h}',[(107,448),(207,448),(207,448),(307,448)],[100,h,100],'mixed')
(ROOT/'boundary-probes.json').write_text(json.dumps(P,indent=2));print(len(P),'commands',len(P)*5,'requests')
