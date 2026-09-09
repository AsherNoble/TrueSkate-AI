"""Bounded, resumable local experiment batches; never evaluates the test partition.

Each batch has an explicit hypothesis. Review saved training failures between
batches; adding new algorithm families is an agent decision, not a blind search.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from classical_model1 import Config, atomic, digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--configs',type=Path,required=True,help='JSON list of {hypothesis, config}')
    ap.add_argument('--seconds',type=float,default=900);ap.add_argument('--limit',type=int,default=256)
    ap.add_argument('--partition',choices=['train','validation'],default='train')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();results=[]
    for candidate in json.loads(a.configs.read_text()):
        remaining=a.seconds-(time.monotonic()-started)
        if remaining<=0:break
        config=asdict(Config(**candidate['config']));p=a.out/'configs'/(digest(config)[:16]+'.json');atomic(p,config)
        cmd=[sys.executable,str(Path(__file__).with_name('classical_model1.py')),'evaluate',
             '--manifest',str(a.manifest),'--out',str(a.out),'--config',str(p),
             '--partition',a.partition,'--limit',str(a.limit),'--seconds',str(remaining),
             '--hypothesis',candidate['hypothesis']]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        r=json.loads((a.out/'latest.json').read_text());results.append(r)
        print(json.dumps({'config':config,'metrics':r['metrics'],'complete':r['complete']}),flush=True)
        atomic(a.out/'batch_progress.json',{'configs':str(a.configs),'completed':results,
               'resume_command':[sys.executable,*sys.argv], 'elapsed_this_invocation_s':time.monotonic()-started})
        if not r['complete']:break
    complete=[r for r in results if r['complete']]
    if complete:
        best=max(complete,key=lambda r:(r['metrics']['recovery'],
                 -(r['metrics']['conditional_error_mean'] or [1,1,1])[2],-r['wall_seconds']))
        atomic(a.out/('best_'+a.partition+'.json'),best)

if __name__=='__main__':main()
