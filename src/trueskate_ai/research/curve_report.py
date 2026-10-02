"""Blinded audit selection and fail-closed spacing-rule decisions."""
from __future__ import annotations
from collections import defaultdict
import copy
import math
import numpy as np
from trueskate_ai.research.curve_protocol import PROTOCOL,RULES,digest
from trueskate_ai.research.curve_measurement import symmetric_distance


def reviewed_rows(rows,manifest,annotations):
    """Resolve only indeterminate cases, retaining immutable automated evidence.

    Human observations are still blind pixel annotations; they are never inferred
    from the intended curve. Audited automated cases retain primary measurements.
    """
    from trueskate_ai.sim.cubic_curve import CubicInTime,compile_curve
    from trueskate_ai.research.curve_measurement import score_observation
    specs={}
    for device in manifest['devices'].values():
        for stage in ('pilot','main'):
            specs.update({r['command_id']:r for r in device[stage]})
        for commands in device['confirmation'].values():
            specs.update({r['command_id']:r for r in commands})
    result=copy.deepcopy(rows)
    for i in audit_selection(rows)['review']:
        mark=annotations.get(i)
        if not mark or not mark.get('complete_trail') or mark.get('interruption') is None or mark.get('collapse') is None:
            continue
        if not mark.get('centreline') or len(mark.get('contacts',[]))<3 or mark.get('endpoint') is None:
            continue
        start,end=mark.get('motion_start_s'),mark.get('motion_end_s')
        if start is None or end is None or not (math.isfinite(start) and math.isfinite(end)
                and (end>start or (end==start and mark['collapse']))):continue
        row=result[i];spec=specs[row['command_id']]
        observation=copy.deepcopy(row['observation'])
        row['automated_observation']=row['observation'];row['automated_metrics']=row['metrics']
        if not observation['shape_evaluable']:observation['centreline']=mark['centreline']
        observation.update(shape_evaluable=True,contacts=sorted(mark['contacts'],key=lambda c:c['pts_s']),
                           duration_evaluable=True,visible_motion_interval_s=[start,end],
                           collapse_suspected=mark['collapse'],interruption=mark['interruption'],indeterminate_reasons=[])
        curve=CubicInTime.from_dict(spec['curve'])
        kwargs={'n':spec['requested_n']} if spec['requested_n'] is not None else {'max_segment_duration_ms':spec['rule_ms']}
        compiled=compile_curve(curve,**kwargs)
        metrics=score_observation(observation,compiled,curve,onset_s=row['onset_s'],frame_s=row['metrics']['frame_time_uncertainty_s'])
        metrics['endpoint_error']=float(np.linalg.norm(np.asarray(mark['endpoint'])-curve.evaluate(1)))
        metrics['evaluable']=(metrics['shape_error'] is not None and metrics['duration_error_s'] is not None
                              and all(v is not None for v in metrics['progress_errors_s']))
        row.update(observation=observation,metrics=metrics,review_source='blinded human resolution')
    return result


def audit_selection(rows):
    """Seeded random 15% audit with marginal stratification, plus 10% repeats.

    Indeterminate observations are additional review cases, not audit substitutes.
    """
    if not rows:return dict(audit=[],repeat=[],review=[])
    rng=np.random.default_rng(PROTOCOL['audit_seed'])
    count=max(1,math.ceil(.15*len(rows)))
    fields=('device','family','duration_s','rule_ms')
    levels={f:{r[f] for r in rows} for f in fields}
    selected=None
    for _ in range(10000):
        candidate=rng.choice(len(rows),size=count,replace=False).tolist()
        if all({rows[i][f] for i in candidate}==levels[f] for f in fields):
            selected=candidate;break
    if selected is None:
        # Small early-aborted pilots cannot cover absent/unrepresented cells.
        selected=rng.choice(len(rows),size=count,replace=False).tolist()
    repeat=rng.choice(selected,size=min(len(selected),max(1,math.ceil(.10*len(rows)))),replace=False).tolist()
    review=[i for i,r in enumerate(rows) if not r['metrics']['evaluable'] or r['metrics'].get('interruption') is None]
    return dict(audit=sorted(selected),repeat=sorted(repeat),review=review)


def pilot_floor(rows,annotations,repeat_annotations):
    selection=audit_selection(rows)
    missing=[i for i in selection['audit'] if i not in annotations]
    missing_repeat=[i for i in selection['repeat'] if i not in repeat_annotations]
    if missing or missing_repeat:return dict(status='inconclusive',reason='blinded audit/reannotation incomplete',missing=missing,missing_repeat=missing_repeat)
    discrepancies=[];repeat_errors=[]
    try:
        for i in selection['audit']:
            if not annotations[i].get('complete_trail'):raise ValueError('audit trail incomplete')
            original=rows[i].get('automated_observation',rows[i]['observation'])
            discrepancies.append(symmetric_distance(original['centreline'],annotations[i]['centreline']))
        for i in selection['repeat']:
            repeat_errors.append(symmetric_distance(annotations[i]['centreline'],repeat_annotations[i]['centreline']))
        uncertainty=max(discrepancies,default=0)+max(repeat_errors,default=0)
        groups=defaultdict(list)
        for row in rows:
            if row['stage']=='pilot':groups[(row['device'],row['curve_id'],row['n'])].append(row)
        pairs=[]
        for pair in groups.values():
            if len(pair)!=2 or not all(r['observation']['shape_evaluable'] for r in pair):
                raise ValueError('identical-command repeat not spatially evaluable')
            pairs.append(symmetric_distance(pair[0]['observation']['centreline'],pair[1]['observation']['centreline']))
        if len(pairs)!=16:raise ValueError('pilot identical-command pairs incomplete')
    except ValueError as exc:return dict(status='inconclusive',reason=str(exc))
    floor=max(uncertainty,max(pairs,default=0)/2)
    budget=.01 if floor<=.005 else .03
    result=dict(status='ready' if floor<=.025 else 'inconclusive',floor=floor,
                extractor_human_max=max(discrepancies,default=0),within_human_max=max(repeat_errors,default=0),
                paired_repeatability_max=max(pairs,default=0),uncertainty_bound=uncertainty,budget=budget,
                primary_assessment='eligible' if floor<=.005 else 'inconclusive')
    # Dense pilot commands are deliberate collapse probes. An observable
    # collapse is a diagnostic result, not proof the whole method is inadequate.
    if any(not r['metrics']['evaluable'] and not (r['stage']=='pilot_dense' and r['metrics']['collapse']) for r in rows):
        result.update(status='inconclusive',reason='pilot measurement indeterminate; human review required')
    return result


def gesture_pass(row,budget,uncertainty):
    m=row['metrics']
    if not m['evaluable']:return False
    shape=m.get('shape_error');end=m.get('endpoint_error');duration=m.get('duration_error_s')
    if any(v is None or not math.isfinite(v) or v<0 for v in (shape,end,duration)):return False
    progress=m.get('progress_errors_s',[])
    if row['duration_s']>=.6 and (len(progress)!=3 or any(v is None or not math.isfinite(v) or v>2*m['frame_time_uncertainty_s'] for v in progress)):
        return False
    return (shape+uncertainty<=budget and end+uncertainty<=budget and duration<=.10
            and m.get('interruption') is False and m.get('collapse') is False and not row.get('contamination',False))


def rule_results(rows,*,budget,uncertainty):
    results=[]
    for rule in RULES:
        subset=[r for r in rows if r['rule_ms']==rule]
        groups=defaultdict(list)
        for r in subset:groups[(r['device'],r['family'])].append(r)
        evaluated=[r for r in subset if r['metrics']['evaluable']]
        coverage=len(evaluated)/len(subset) if subset else 0
        group_coverage={f'{d}/{f}':sum(r['metrics']['evaluable'] for r in g)/len(g) for (d,f),g in groups.items()}
        passed=(bool(subset) and all(r['offline_eligible'] for r in subset) and coverage>=.90
                and len(groups)==8 and all(c>=.80 for c in group_coverage.values())
                and all(gesture_pass(r,budget,uncertainty) for r in evaluated)
                and all(r['metrics'].get('collapse') is False and r['metrics'].get('interruption') is False and not r.get('contamination',False) for r in subset))
        results.append(dict(rule_ms=rule,pass_criteria=passed,attempts=len(subset),evaluable=len(evaluated),
                            coverage=coverage,device_family_coverage=group_coverage))
    return results


def stage_decision(manifest,rows,*,stage,pilot_gate=None,annotations=None,repeat_annotations=None):
    """Only complete, audited, frozen cohorts can authorize a subsequent stage."""
    result=dict(manifest_sha256=manifest['sha256'],stage=stage,status='inconclusive',allow_execution=False,
                primary_budget=.01,secondary_budget=.03)
    annotations=annotations or {};repeat_annotations=repeat_annotations or {}
    if stage=='pilot':
        expected={r['command_id'] for d in manifest['devices'].values() for r in d['pilot']}
        if {r['command_id'] for r in rows}!=expected or len(rows)!=36:
            return dict(**result,reason='pilot incomplete or duplicate attempts; preserve failures, do not replace')
        floor=pilot_floor(rows,annotations,repeat_annotations);result['measurement_floor']=floor
        if floor['status']=='ready':
            result.update(status='pilot_pass',allow_execution=True,next_stage='main',budget=floor['budget'],
                          uncertainty_bound=floor['uncertainty_bound'],extractor_parameters=PROTOCOL['orange_hsv'])
        return result
    if not pilot_gate or pilot_gate.get('manifest_sha256')!=manifest['sha256'] or not pilot_gate.get('allow_execution'):
        return dict(**result,reason='missing frozen passing pilot gate')
    audit=audit_selection(rows)
    if any(i not in annotations for i in audit['audit']) or any(i not in repeat_annotations for i in audit['repeat']):
        return dict(**result,reason='blinded audit/reannotation incomplete')
    # Main audit must not exceed the pilot-frozen measurement uncertainty.
    floor_uncertainty=pilot_gate['uncertainty_bound']
    try:
        audit_error=max(symmetric_distance(rows[i].get('automated_observation',rows[i]['observation'])['centreline'],annotations[i]['centreline']) for i in audit['audit'])
        repeat_error=max(symmetric_distance(annotations[i]['centreline'],repeat_annotations[i]['centreline']) for i in audit['repeat'])
    except (ValueError,KeyError):return dict(**result,reason='audit indeterminate')
    if audit_error+repeat_error>floor_uncertainty:
        return dict(**result,reason='audit exceeds frozen uncertainty bound')
    if stage=='main':
        expected={r['command_id'] for d in manifest['devices'].values() for r in d['main']}
        if {r['command_id'] for r in rows}!=expected or len(rows)!=144:
            return dict(**result,reason='main cohort incomplete or duplicated')
        results=rule_results(rows,budget=pilot_gate['budget'],uncertainty=floor_uncertainty)
        result['rule_results']=results
        passing=[r['rule_ms'] for r in results if r['pass_criteria']]
        if passing:result.update(status='main_pass',allow_execution=True,next_stage='confirmation',selected_rule_ms=max(passing),
                                 budget=pilot_gate['budget'],uncertainty_bound=floor_uncertainty)
        return result
    selected=pilot_gate['selected_rule_ms']
    expected={r['command_id'] for d in manifest['devices'].values() for r in d['confirmation'][str(selected)]}
    if {r['command_id'] for r in rows}!=expected or len(rows)!=24:
        return dict(**result,reason='confirmation cohort incomplete or duplicated')
    results=rule_results(rows,budget=pilot_gate['budget'],uncertainty=floor_uncertainty)
    passing=next(r for r in results if r['rule_ms']==selected)['pass_criteria']
    result.update(status='confirmed' if passing else 'failed_confirmation',selected_rule_ms=selected,
                  budget=pilot_gate['budget'],rule_results=results)
    return result


def grouped_summary(rows):
    groups=defaultdict(list)
    for r in rows:
        key=(r['device'],r['family'],r['duration_s'],r['n'],r['min_segment_duration_ms'],r['max_segment_duration_ms'])
        groups[key].append(r)
    summary=[]
    for key,group in groups.items():
        summary.append(dict(device=key[0],family=key[1],duration_s=key[2],n=key[3],min_segment_duration_ms=key[4],max_segment_duration_ms=key[5],
                            attempts=len(group),evaluable=sum(r['metrics']['evaluable'] for r in group),
                            indeterminate=sum(not r['metrics']['evaluable'] for r in group),
                            collapse_candidates=sum(r['metrics']['collapse'] for r in group),
                            max_shape_error=max((r['metrics']['shape_error'] for r in group if r['metrics']['shape_error'] is not None),default=None)))
    return summary
