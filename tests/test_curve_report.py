import copy
from trueskate_ai.research.curve_report import stage_decision,rule_results,audit_selection,gesture_pass
from trueskate_ai.research.curve_protocol import command_manifest


def test_incomplete_pilot_never_authorizes_main():
    result=stage_decision(command_manifest(),[],stage='pilot')
    assert result['status']=='inconclusive' and not result['allow_execution']


def row(device,family,rule,duration=.6):
    return dict(device=device,family=family,rule_ms=rule,duration_s=duration,offline_eligible=True,
                metrics=dict(evaluable=True,shape_error=.002,endpoint_error=.002,duration_error_s=.08,
                             progress_errors_s=[.02,.03,.04],frame_time_uncertainty_s=1/30,collapse=False,interruption=False))


def test_select_rules_requires_every_device_family_and_uncertainty():
    rows=[row(d,f,s) for d in ('XR1','XR2') for f in ('straight','arc','s_curve','reversal') for s in (10,20,40)]
    assert all(r['pass_criteria'] for r in rule_results(rows,budget=.01,uncertainty=.003))
    bad=copy.deepcopy(rows);bad[0]['metrics']['interruption']=None
    assert not rule_results(bad,budget=.01,uncertainty=.003)[0]['pass_criteria']
    assert not any(r['pass_criteria'] for r in rule_results(rows,budget=.01,uncertainty=.009))
    bad=copy.deepcopy(rows);bad[0]['offline_eligible']=False
    assert not rule_results(bad,budget=.01,uncertainty=0)[0]['pass_criteria']


def test_short_timing_progress_excluded_but_duration_and_collapse_checked():
    r=row('XR1','arc',10,.3);r['metrics']['progress_errors_s']=[]
    assert gesture_pass(r,.01,.001)
    r['metrics']['duration_error_s']=.11;assert not gesture_pass(r,.01,.001)
    r['metrics']['duration_error_s']=.08;r['metrics']['collapse']=True
    assert not gesture_pass(r,.01,.001)


def test_blind_sample_is_deterministic_and_indeterminate_cases_are_extra():
    rows=[row(d,f,s,t) for d in ('XR1','XR2') for f in ('straight','arc','s_curve','reversal') for s in (10,20,40) for t in (.3,.6,1.2)]
    a=audit_selection(rows);assert a==audit_selection(rows)
    assert len(a['audit'])==11 and len(a['repeat'])==8
    rows[0]['metrics']['evaluable']=False
    assert 0 in audit_selection(rows)['review']


def test_human_resolution_retains_primary_automated_observation():
    import numpy as np
    from trueskate_ai.research.curve_report import reviewed_rows
    from trueskate_ai.research.curve_measurement import score_observation
    from trueskate_ai.sim.cubic_curve import CubicInTime,compile_curve
    from trueskate_ai.data.touch_labels import cubic_command_label
    manifest=command_manifest();spec=manifest['devices']['iPhone_XR']['pilot'][0]
    c=CubicInTime.from_dict(spec['curve']);compiled=compile_curve(c,n=16)
    points=c.evaluate(np.linspace(0,1,101)).tolist()
    observation=dict(centreline=points,contacts=[],shape_evaluable=False,duration_evaluable=False,
                     visible_motion_interval_s=None,collapse_suspected=False,interruption=None,indeterminate_reasons=['branch'])
    metrics=score_observation(observation,compiled,c,onset_s=10,frame_s=1/30)
    r=dict(command_id=spec['command_id'],device='iPhone_XR',family='straight',duration_s=.3,rule_ms=None,
           onset_s=10.,observation=observation,metrics=metrics)
    contacts=[]
    for i,t in enumerate(np.linspace(0,.3,10)):
        label=cubic_command_label(compiled,float(t));contacts.append(dict(xy=[label.x,label.y],pts_s=10+float(t),source_frame=i))
    mark=dict(complete_trail=True,centreline=points,contacts=contacts,endpoint=points[-1],
              motion_start_s=10.,motion_end_s=10.3,interruption=False,collapse=False)
    resolved=reviewed_rows([r],manifest,{0:mark})[0]
    assert not r['metrics']['evaluable'] and not r['observation']['shape_evaluable']
    assert resolved['metrics']['evaluable']
    assert resolved['automated_observation']==r['observation']


def test_pilot_floor_fallback_is_preregistered_and_dense_collapse_is_diagnostic():
    from trueskate_ai.research.curve_report import pilot_floor
    import numpy as np
    manifest=command_manifest();rows=[]
    for name,device in manifest['devices'].items():
        for spec in device['pilot']:
            dense=spec['stage']=='pilot_dense'
            rows.append(dict(device=name,family=spec['family'],duration_s=spec['curve']['duration_s'],rule_ms=None,
                             stage=spec['stage'],curve_id=spec['curve_id'],n=spec['compiled']['n'],
                             observation=dict(centreline=[[.4,.5],[.6,.5]],shape_evaluable=True),
                             metrics=dict(evaluable=not dense,collapse=dense,interruption=False)))
    selected=audit_selection(rows)
    def annotate(offset):return {i:dict(complete_trail=True,centreline=[[.4,.5+offset],[.6,.5+offset]]) for i in selected['audit']}
    for offset,budget,status in [(0.,.01,'ready'),(.006,.03,'ready'),(.026,.03,'inconclusive')]:
        annotations=annotate(offset);repeat={i:annotations[i] for i in selected['repeat']}
        result=pilot_floor(rows,annotations,repeat)
        assert result['budget']==budget and result['status']==status
