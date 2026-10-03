"""Bounded paired direct-waypoint execution audit; no training admission."""
import hashlib
import json
import math
import random
from pathlib import Path
from trueskate_ai.sim.timed_waypoints import TimedWaypoints
from trueskate_ai.collection.die_five_calibration import die_five_points_pt
from trueskate_ai.data.control_hitboxes import CONTROL_START_MAP_VERSION

SEED=2026100305
IDENTITY='curved-execution-20261003-v1'
FAMILIES=('arcs','s_curves','multiple_bends','loops','reversals')
PROFILES=('constant','accelerating','decelerating','slow_fast_slow','pause')
SLOTS=(5,10,15,20,25,35,40,45,50,54)
DEVICES=('iPhone_XR','iPhone_XR2')
# v2 resets the board 3 s before the middle and end markers (and before recording,
# ahead of the start marker): in v1 the board drifted under the die-five points and
# its motion out-voted the touches (run 2, XR1 segment 1, end marker 2/5 votes).
# The last sample moves 54->53 s and the end marker 57->58 s to keep that 3 s settle.
SCHEDULES={'v1':dict(identity=IDENTITY,slots=SLOTS,markers=(('start',1),('middle',30),('end',57)),resets=()),
           'v2':dict(identity='curved-execution-20261003-v2-reset-before-markers',slots=(5,10,15,20,25,35,40,45,50,53),
                     markers=(('start',1),('middle',30),('end',58)),resets=(27,55))}
PRE_ROLL_SETTLE_S=3
PARKS=('Inbound','Skateboard GB 2024')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def intervals(weights,total):
    n=len(weights)
    if total<n or min(weights)<=0:raise ValueError('positive duration weights required')
    raw=[w/sum(weights)*(total-n) for w in weights]
    result=[1+int(x) for x in raw]
    for i in sorted(range(n),key=lambda i:raw[i]-int(raw[i]),reverse=True)[:total-sum(result)]:result[i]+=1
    return result

def marker_payload():
    return {'actions':[dict(type='pointer',id=f'marker_{i}',parameters={'pointerType':'touch'},actions=[
        dict(type='pointerMove',duration=0,x=x,y=y,origin='viewport'),dict(type='pointerDown',button=0),
        dict(type='pause',duration=50),dict(type='pointerUp',button=0)]) for i,(x,y) in enumerate(die_five_points_pt(71))]}

def reset_payload():
    """Tap True Skate's reset button, as in the linear speed sweep."""
    return {'actions':[dict(type='pointer',id='reset',parameters={'pointerType':'touch'},actions=[
        dict(type='pointerMove',duration=0,x=207,y=49,origin='viewport'),dict(type='pointerDown',button=0),
        dict(type='pause',duration=50),dict(type='pointerUp',button=0)])]}

def manifest(schedule='v1'):
    plan=SCHEDULES[schedule]
    rng=random.Random(SEED);paths=[]
    for family_i,family in enumerate(FAMILIES):
        for d_i,duration in enumerate((150,300,600,900,1200)):
            for repeat in range(2):
                index=len(paths);n=(5,8,11,15)[index%4];profile=PROFILES[(d_i+2*repeat+family_i)%5]
                length=(100,200,300,400,500)[(d_i+repeat+family_i)%5]
                for attempt in range(10000):
                    m=n-1 if profile=='pause' else n
                    u=[i/(m-1) for i in range(m)]
                    phase=rng.uniform(.7,1.4)
                    if family=='arcs':base=[(math.cos(math.pi*t),phase*math.sin(math.pi*t)) for t in u]
                    elif family=='s_curves':base=[(2*t-1,phase*.6*math.sin(2*math.pi*t)) for t in u]
                    elif family=='multiple_bends':base=[(2*t-1,phase*(.6 if i%2 else -.6)) for i,t in enumerate(u)]
                    elif family=='loops':base=[(math.sin(2*math.pi*t),phase*(math.sin(4*math.pi*t) if repeat else math.cos(2*math.pi*t))) for t in u]
                    else:base=[(math.cos(3*math.pi*t),0) for t in u]
                    if profile=='pause':base.insert(m//2,base[m//2-1])
                    total=sum(math.dist(a,b) for a,b in zip(base,base[1:]))
                    angle=rng.uniform(-math.pi,math.pi);cx=rng.uniform(145,300);cy=rng.uniform(280,650)
                    scale=length/total
                    pts=[((cx+scale*(x*math.cos(angle)-y*math.sin(angle)))/414,
                          (cy+scale*(x*math.sin(angle)+y*math.cos(angle)))/896) for x,y in base]
                    distances=[math.dist(a,b) for a,b in zip(base,base[1:])]
                    weights=[]
                    for i,dist in enumerate(distances):
                        t=(i+.5)/(n-1)
                        factor={'constant':1,'accelerating':1.8-1.5*t,'decelerating':.3+1.5*t,
                                'slow_fast_slow':.3+2*abs(t-.5),'pause':1}[profile]
                        weights.append((dist or total*.15)*factor)
                    times=[0]
                    for dt in intervals(weights,duration):times.append(times[-1]+dt)
                    try:
                        path=TimedWaypoints(tuple(pts),tuple(times));payload=path.payload()
                    except ValueError:continue
                    break
                else:raise ValueError('safe sampling exhausted')
                paths.append(dict(path_id=f'p{index:02}',family=family,duration_ms=duration,waypoint_count=n,
                                  timing_profile=profile,target_length_pt=length,points=pts,times_ms=times,
                                  quantized_points=path.quantized(),payload=payload,rejected_candidates=attempt))
    orders=[]
    for i in range(2):
        order=list(range(50));random.Random(SEED+100+i).shuffle(order);orders.append(order)
    recordings=[]
    for segment in range(5):
        for device_i in range(2):
            commands=[dict(kind='control',role=role,slot_s=slot,payload=marker_payload()) for role,slot in plan['markers']]
            commands+=[dict(kind='reset',slot_s=slot,payload=reset_payload()) for slot in plan['resets']]
            for slot,index in zip(plan['slots'],orders[device_i][segment*10:segment*10+10]):
                commands.append(dict(kind='sample',slot_s=slot,path_id=paths[index]['path_id'],payload=paths[index]['payload']))
            recordings.append(dict(device=DEVICES[device_i],park=PARKS[device_i],commands=sorted(commands,key=lambda c:c['slot_s'])))
    value=dict(identity=plan['identity'],seed=SEED,paths=paths,recordings=recordings,stop_s=59,
               control_map=CONTROL_START_MAP_VERSION,training_admission=False)
    if schedule!='v1':  # v1 bytes stay identical to the frozen db5c843c manifest
        value.update(schedule=schedule,pre_roll_reset_settle_s=PRE_ROLL_SETTLE_S)
    value['sha256']=digest(value)
    return value

def verify_manifest(value):
    if digest({k:v for k,v in value.items() if k!='sha256'})!=value['sha256']:raise ValueError('manifest hash mismatch')
    # libm differs by a few ulps across Intel/ARM hosts. Frozen bytes stay
    # authoritative; permit that difference only in abstract floating positions.
    # Exact integer payloads, times, order, conditions and quantization must match.
    expected=manifest(value.get('schedule','v1'))
    normalized=json.loads(json.dumps(value))
    for actual,wanted in zip(normalized['paths'],expected['paths']):
        if len(actual['points'])!=len(wanted['points']) or any(
            abs(a-b)>1e-12 for p,q in zip(actual['points'],wanted['points']) for a,b in zip(p,q)):
            raise ValueError('abstract path differs from seed')
        actual['points']=wanted['points']
    normalized.pop('sha256');expected.pop('sha256')
    if digest(normalized)!=digest(expected):raise ValueError('manifest differs from frozen generation')
