"""Frozen 135-gesture factorial diagnostic with separate blinded review order."""
import random
import hashlib
import math
from trueskate_ai.research.curve_protocol import digest
from trueskate_ai.research.linear_speed_probe import PATH, payload
from trueskate_ai.data.control_hitboxes import segment_is_safe, CONTROL_START_MAP_VERSION

EXPERIMENT = 'LINEAR-LENGTH-20261003'
DURATIONS = tuple(range(50,9,-5))
FRACTIONS = (.2,.4,.6,.8,1.)
SLOTS = (5.,10.,15.,20.,25.,35.,40.,45.,50.)
SEED = 2026100301


def manifest():
    rng = random.Random(SEED)
    recordings = []
    for repetition in range(1,4):
        cells = [(length,duration) for length in range(5) for duration in DURATIONS]
        rng.shuffle(cells)
        for chunk in range(5):
            commands = [dict(kind='control', role=role, slot_s=slot)
                        for role,slot in (('start',1.),('middle',30.),('end',57.))]
            for slot,(length,duration) in zip(SLOTS,cells[chunk*9:(chunk+1)*9]):
                fraction=FRACTIONS[length]
                end=tuple(a+fraction*(b-a) for a,b in zip(*PATH))
                path=(PATH[0],end)
                if not segment_is_safe(*path):raise ValueError('unsafe full path')
                commands.extend([dict(kind='reset',slot_s=slot-3.),
                    dict(kind='diagnostic',slot_s=slot,duration_ms=duration,path=path,
                         length_index=length,length_fraction=fraction,repetition=repetition,
                         logical_length_points=math.hypot((end[0]-PATH[0][0])*414,(end[1]-PATH[0][1])*896),
                         command_id=f'p{repetition}-l{length}-d{duration}')])
            commands.sort(key=lambda c:c['slot_s'])
            for c in commands:c['payload']=payload(c)
            recordings.append(commands)
    result=dict(experiment=EXPERIMENT,device='iPhone_XR',park='Inbound',
                durations_ms=DURATIONS,length_fractions=FRACTIONS,repetitions=3,
                recordings=recordings,execution_seed=SEED,whole_path_safe=True,
                control_map_version=CONTROL_START_MAP_VERSION,preparation_lead_s=3.,
                late_tolerance_s=.1,stop_s=59.,settle_threshold=2.,settle_consecutive=2,
                gameplay_review='operator visual review',training_admission=False)
    result['sha256']=digest(result)
    return result


def blind_key(frozen):
    """Private mapping: never put this or the manifest inside the viewer root."""
    items=[]
    for recording,commands in enumerate(frozen['recordings'],1):
        for c in commands:
            if c['kind']=='diagnostic':
                token=hashlib.sha256(f'{SEED+1}:{c["command_id"]}'.encode()).hexdigest()[:20]
                items.append(dict(token=token,recording=recording,command_id=c['command_id'],
                                  duration_ms=c['duration_ms'],length_fraction=c['length_fraction'],
                                  repetition=c['repetition']))
    random.Random(SEED+2).shuffle(items)
    return dict(experiment=EXPERIMENT,manifest_sha256=frozen['sha256'],items=items)
