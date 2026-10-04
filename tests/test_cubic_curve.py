import numpy as np
import pytest
from trueskate_ai.sim.cubic_curve import CubicInTime,compile_curve,fit_eight_positions,curve_pointer
from trueskate_ai.data.touch_labels import cubic_command_label,segment_durations_s
from trueskate_ai.sim.touch_actions import make_touch_pointer,build_curved_drag
from trueskate_ai.research.curve_protocol import seeded_curves,command_manifest,verify_manifest,offline_report


def cubic(duration=.3006):
    return CubicInTime(((.35,.4),(.4,.7),(.7,.3),(.85,.6)),duration)


def test_fit_exact_and_endpoints():
    c=cubic();positions=c.evaluate(np.linspace(0,1,8));fit=fit_eight_positions(positions,c.duration_s)
    np.testing.assert_allclose(fit.coefficients,c.coefficients,atol=1e-14)
    noisy=positions.copy();noisy[1:-1]+=.004
    f=fit_eight_positions(noisy,.6)
    np.testing.assert_array_equal(f.evaluate(0),positions[0]);np.testing.assert_array_equal(f.evaluate(1),positions[-1])
    assert CubicInTime.from_dict(c.to_dict())==c


@pytest.mark.parametrize('points',[[(.5,.5)]*4,[(.3+i*.1,.4+i*.05) for i in range(4)]])
def test_degeneracies(points):
    c=CubicInTime(points,.6);f=fit_eight_positions(c.evaluate(np.linspace(0,1,8)),.6)
    np.testing.assert_allclose(f.evaluate(np.linspace(0,1,101)),c.evaluate(np.linspace(0,1,101)),atol=1e-14)
    c.validate_safe()


def test_derivatives():
    c=cubic();u=.42;epsilon=1e-5
    np.testing.assert_allclose(c.derivative(u),(c.evaluate(u+epsilon)-c.evaluate(u-epsilon))/(2*epsilon),atol=1e-8)
    np.testing.assert_allclose(c.derivative(u,2),(c.derivative(u+epsilon)-c.derivative(u-epsilon))/(2*epsilon),atol=1e-8)
    np.testing.assert_allclose(c.derivative(u,seconds=True),c.derivative(u)/c.duration_s)


@pytest.mark.parametrize('points',[
    ((.5,.5),(.05,.5),(.8,.5),(.8,.6)), # unsafe coefficient despite safe endpoints
    ((.14,.3),(.14,.4),(.4,.5),(.5,.5)), # left control
    ((.2,.2),(.9,.13),(.9,.2),(.4,.5)), # top control hull
])
def test_unsafe_hull_rejected_without_repair(points):
    c=CubicInTime(points,.6)
    with pytest.raises(ValueError):compile_curve(c,n=16)
    assert c.coefficients==points


@pytest.mark.parametrize('duration',[.3006,.6,1.2])
@pytest.mark.parametrize('n',[2,4,8,16,32])
def test_payload_and_labels_agree_at_every_actual_boundary(duration,n):
    c=cubic(duration);compiled=compile_curve(c,n=n)
    actions=curve_pointer(compiled).encode()['actions']
    assert [a['type'] for a in actions]==['pointerMove','pointerDown']+['pointerMove']*n+['pointerUp']
    assert sum(a.get('duration',0) for a in actions)==round(duration*1000)
    elapsed=0;observed=[]
    for action in actions:
        elapsed+=action.get('duration',0)
        if action['type']=='pointerMove':
            label=cubic_command_label(compiled,elapsed/1000)
            assert (label.x,label.y)==pytest.approx((action['x']/414,action['y']/896),abs=1e-12)
            observed.append(elapsed)
    assert observed==list(compiled.boundaries_ms)
    np.testing.assert_allclose(np.asarray(compiled.points_device),np.rint(c.evaluate(np.asarray(observed)/observed[-1])*[414,896]))
    midpoint=(compiled.boundaries_ms[0]+compiled.boundaries_ms[1])/2000
    label=cubic_command_label(compiled,midpoint)
    assert (label.x,label.y)==pytest.approx(np.mean(compiled.points_normalized[:2],axis=0))
    assert not cubic_command_label(compiled,-.01).active
    assert not cubic_command_label(compiled,duration+.01).active


def test_spacing_constraints_and_floor():
    c=cubic(.3)
    for spacing in (10,16,20,37.5,40):
        compiled=compile_curve(c,max_segment_duration_ms=spacing)
        assert max(compiled.segment_durations_ms)<=spacing
        assert sum(compiled.segment_durations_ms)==300
    with pytest.raises(ValueError,match='minimum'):compile_curve(c,n=32,min_segment_duration_ms=16)
    with pytest.raises(ValueError,match='minimum'):compile_curve(c,max_segment_duration_ms=10,min_segment_duration_ms=16)
    for n in (0,301,True):
        with pytest.raises(ValueError):compile_curve(c,n=n)
    with pytest.raises(ValueError):compile_curve(c,n=16,max_segment_duration_ms=20)


def test_closed_endpoint_snaps_before_range_rejection():
    compiled=compile_curve(cubic(.301),n=8)
    end=compiled.boundaries_ms[-1]/1000
    label=cubic_command_label(compiled,np.nextafter(end,np.inf))
    assert label.active
    assert (label.x,label.y)==pytest.approx(compiled.points_normalized[-1])
    assert cubic_command_label(compiled,-1e-13).active
    assert not cubic_command_label(compiled,end+1e-6).active
    assert not cubic_command_label(compiled,-1e-6).active


def test_bound_covers_dense_time_aligned_error_and_device_scaling():
    c=cubic();compiled=compile_curve(c,n=8,device_size=(375,812))
    errors=[]
    for t in np.linspace(0,compiled.boundaries_ms[-1]/1000,3001):
        label=cubic_command_label(compiled,t)
        errors.append(np.linalg.norm(np.array([label.x,label.y])-c.evaluate(t/(compiled.boundaries_ms[-1]/1000))))
    assert max(errors)<=compiled.approximation_bound+1e-12


def test_explicit_duration_rejects_bad_inputs_and_legacy_is_unchanged():
    points=[(.3,.4),(.4,.5),(.5,.6),(.6,.7)]
    finger=make_touch_pointer();build_curved_drag(finger,points,total_duration=.301)
    assert [a['duration'] for a in finger.encode()['actions'] if a['type']=='pointerMove'][1:]==[100]*3
    assert segment_durations_s(4,.301,1)==[.1]*3
    for durations in ([1,2],[1,0,2],[1,True,2],[1,2.,3]):
        with pytest.raises(ValueError):build_curved_drag(make_touch_pointer(),points,segment_durations_ms=durations)
    with pytest.raises(ValueError):build_curved_drag(make_touch_pointer(),points,segment_durations_ms=[1,2,3],easing=lambda t:t)


def test_deterministic_manifest_and_offline_limits():
    assert seeded_curves()==seeded_curves()
    assert seeded_curves()!=seeded_curves(20261003)
    manifest=command_manifest();verify_manifest(manifest)
    assert sum(len(d['pilot']) for d in manifest['devices'].values())==36
    assert sum(len(d['main']) for d in manifest['devices'].values())==144
    assert all(min(r['compiled']['segment_durations_ms'])<16 for d in manifest['devices'].values() for r in d['pilot'][-2:])
    report=offline_report()
    assert len(report['rows'])==96
    assert all(r['max_time_aligned_residual']>.005 for r in report['representation_residuals'])
    manifest['protocol']['primary_budget']=.5
    with pytest.raises(ValueError):verify_manifest(manifest)


def test_frozen_integrity_and_cross_numpy_ulp_tolerance():
    from trueskate_ai.research.curve_protocol import digest,equivalent_commands
    m=command_manifest();m['devices']['iPhone_XR']['pilot'][0]['curve']['coefficients'][0][0]+=1e-16
    with pytest.raises(ValueError):verify_manifest(m) # checksum still exact
    m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'});verify_manifest(m)
    m['devices']['iPhone_XR']['pilot'][0]['compiled']['points_device'][0][0]+=1
    m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'})
    with pytest.raises(ValueError):verify_manifest(m) # never permit even one logical point/ms
    assert not equivalent_commands([1],[1.])
