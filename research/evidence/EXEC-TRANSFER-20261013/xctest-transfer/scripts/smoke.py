import json, sys, time
sys.path.insert(0, sys.argv[1])
import wda
sid = wda.new_session(); print('sid', sid)
print(wda.timing(sid, True))
p = {'actions':[{'type':'pointer','id':'timed_path','parameters':{'pointerType':'touch'},'actions':[
  {'type':'pointerMove','duration':0,'x':107,'y':448,'origin':'viewport'},{'type':'pointerDown','button':0},
  {'type':'pointerMove','duration':200,'x':307,'y':448,'origin':'viewport'},{'type':'pointerUp','button':0}]}]}
r, t0, t1 = wda.actions(sid, p); print(r, t0, t1, t1-t0)
time.sleep(0.5)
print(json.dumps(wda.timing(sid), indent=1)[:3000])
