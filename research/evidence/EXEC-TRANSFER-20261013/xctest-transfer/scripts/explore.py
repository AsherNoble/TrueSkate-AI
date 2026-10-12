import json, sys, time, subprocess
sys.path.insert(0, sys.argv[1]); import wda
LOG = sys.argv[2]
def chain(p0, moves):
    acts=[{'type':'pointerMove','duration':0,'x':p0[0],'y':p0[1],'origin':'viewport'},{'type':'pointerDown','button':0}]
    for (x,y,d) in moves: acts.append({'type':'pointerMove','duration':d,'x':x,'y':y,'origin':'viewport'})
    acts.append({'type':'pointerUp','button':0})
    return {'actions':[{'type':'pointer','id':'timed_path','parameters':{'pointerType':'touch'},'actions':acts}]}
def nlines(): return sum(1 for _ in open(LOG))
sid = wda.new_session()
probes = {
 'move1200': chain((107,448),[(307,448,1200)]),
 'move100': chain((107,448),[(307,448,100)]),
 'move50': chain((107,448),[(307,448,50)]),
 'hold_mid100': chain((107,448),[(207,448,200),(207,448,100),(307,448,200)]),
}
for name,p in probes.items():
    n0 = nlines(); wda.actions(sid,p); time.sleep(0.4)
    lines=[json.loads(l) for l in open(LOG)][n0:]
    t0=lines[0]['ts']
    print('==',name)
    for l in lines: print(f"{(l['ts']-t0)*1000:8.2f} {l['ph']} {l['x']:8.3f} {l['y']:8.3f} cb_lag={(l['now']-l['ts'])*1000:5.1f} nco={len(l['co'])}")
