"""Synthetic XR control fixtures. Never contacts Appium or a physical device."""
import io
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from PIL import Image
import pytest
import requests

from trueskate_ai.control.service import (
    CONFIGS, ControlError, ControlServer, Device, FrameRelay, gesture_actions,
)

ROOT = Path(__file__).resolve().parents[1]


def jpeg(size=(414, 896)):
    output = io.BytesIO()
    Image.new('RGB', size, '#345678').save(output, format='JPEG')
    return output.getvalue()


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.active = []
        self.fail = None
        self.size = {'width':414, 'height':896}

    def request(self, method, url, json=None, timeout=None):
        path = urlsplit(url).path
        self.calls.append((method, url, json, timeout))
        if self.fail and (self.fail == path or (self.fail != '/session' and self.fail in path)):
            raise requests.Timeout('synthetic timeout')
        if path == '/appium/sessions':
            value = [{'id':sid} for sid in self.active]
        elif path == '/status':
            value = {'ready':True}
        elif path == '/session':
            self.active = ['our-session']
            value = {'sessionId':'our-session'}
        elif path.endswith('/window/rect'):
            value = self.size
        elif method == 'DELETE':
            self.active = []
            value = None
        else:
            value = None
        response = requests.Response()
        response.status_code = 200
        response._content = __import__('json').dumps({'value':value}).encode()
        return response


@pytest.fixture
def device():
    relay = FrameRelay(0)
    relay.publish(jpeg())
    return Device(CONFIGS[0], 'synthetic-udid', relay, FakeTransport(), lambda:False)


def body(device, **kwargs):
    return {'epoch':device.epoch, 'sequence':device.sequence+1,
            'frame':device.relay.sequence, **kwargs}


PATH = [{'x':0.1,'y':0.2,'t':0}, {'x':0.7,'y':0.6,'t':200}, {'x':0.3,'y':0.9,'t':500}]


def test_curve_mapping_and_timing():
    steps = gesture_actions(PATH)['actions'][0]['actions']
    assert [(s['x'],s['y']) for s in steps if s['type']=='pointerMove'] == [(41,179),(290,538),(124,806)]
    assert [s['duration'] for s in steps if 'duration' in s] == [0,200,300]
    assert [s['type'] for s in steps] == ['pointerMove','pointerDown','pointerMove','pointerMove','pointerUp']


@pytest.mark.parametrize('duration',[30,50,900,5000])
def test_tap_and_hold(duration):
    steps = gesture_actions([{'x':0,'y':1,'t':0},{'x':0,'y':1,'t':duration}])['actions'][0]['actions']
    assert steps[0]['x']==0 and steps[0]['y']==896
    assert steps[2] == {'type':'pause','duration':duration}


@pytest.mark.parametrize('points', [None, [], PATH[:1], PATH*200,
    [{'x':1.1,'y':0,'t':0},PATH[-1]], [{'x':float('nan'),'y':0,'t':0},PATH[-1]],
    [{'x':True,'y':0,'t':0},PATH[-1]], [PATH[0],{'x':0,'y':0,'t':5001}],
    [PATH[0],{'x':0,'y':0,'t':0}], [PATH[0],{'x':0,'y':0,'t':10}],
    [{'x':0,'y':0,'t':1},PATH[-1]], [PATH[0], {'x':0,'y':0,'t':'20'}]])
def test_invalid_paths(points):
    with pytest.raises(ControlError): gesture_actions(points)


def test_connect_is_noninvasive(device):
    result = device.command('connect', {})
    assert result['connected']
    creation = [c for c in device.transport.calls if c[0]=='POST']
    assert len(creation)==1
    caps=creation[0][2]['capabilities']['alwaysMatch']
    for name in ['autoLaunch','fullReset','forceAppLaunch','shouldTerminateApp','useNewWDA']:
        assert caps['appium:'+name] is False
    assert caps['appium:noReset'] is True
    assert caps['appium:webDriverAgentUrl']=='http://127.0.0.1:8100'
    assert 'synthetic-udid' not in json.dumps(result)


def test_busy_and_collector_refusal(device):
    device.transport.active=['someone-else']
    with pytest.raises(ControlError,match='busy'): device.command('connect',{})
    device.transport.active=[]
    device.ownership=lambda:True
    with pytest.raises(ControlError,match='Collector'): device.command('connect',{})
    assert not any(c[0]=='POST' for c in device.transport.calls)


def test_serialization(device):
    with device.lock:
        with pytest.raises(ControlError,match='already executing'): device.command('connect',{})
    assert not device.transport.calls


def test_duplicate_stale_and_reconnect(device):
    device.command('connect',{})
    first=body(device,points=PATH)
    device.command('gesture',first)
    with pytest.raises(ControlError,match='Duplicate'): device.command('gesture',first)
    with device.relay.condition:
        device.relay.history.clear()
    with pytest.raises(ControlError,match='stale'): device.command('gesture',body(device,points=PATH))
    old_epoch=device.epoch
    device.command('disconnect',body(device))
    device.command('connect',{})
    assert old_epoch!=device.epoch
    with pytest.raises(ControlError,match='Connection changed'): device.command('gesture',first)
    assert sum(c[1].endswith('/actions') for c in device.transport.calls)==1


def test_timeout_quarantines_and_never_replays(device):
    device.command('connect',{})
    device.transport.fail='/actions'
    with pytest.raises(ControlError,match='unknown'): device.command('gesture',body(device,points=PATH))
    assert device.status()['uncertain']
    count=len(device.transport.calls)
    for kind in ['connect','disconnect','gesture','activate','home','app_switcher','control_center']:
        with pytest.raises(ControlError,match='unknown'): device.command(kind,body(device,points=PATH))
    assert len(device.transport.calls)==count


def test_connect_timeout_quarantines(device):
    device.transport.fail='/session'
    with pytest.raises(ControlError,match='unknown'): device.command('connect',{})
    assert device.uncertain


def test_unavailable_does_not_mutate(device):
    device.transport.fail='/appium/sessions'
    with pytest.raises(ControlError,match='unavailable'): device.command('connect',{})
    assert not device.uncertain
    assert not any(c[0]=='POST' for c in device.transport.calls)


def test_wrong_geometry_closed_without_activation(device):
    device.transport.size={'width':375,'height':812}
    with pytest.raises(ControlError,match='414'): device.command('connect',{})
    assert device.session is None
    assert device.transport.active==[]


def test_ownership_changed_before_gesture(device):
    device.command('connect',{})
    device.transport.active=['foreign-session']
    with pytest.raises(ControlError,match='ownership'): device.command('gesture',body(device,points=PATH))
    assert not any(c[1].endswith('/actions') for c in device.transport.calls)


def test_frame_validation():
    relay=FrameRelay(0)
    with pytest.raises(ValueError): relay.publish(jpeg((896,414)))
    relay.publish(jpeg())
    assert relay.fresh(1)
    assert not relay.fresh(True)
    assert not relay.fresh(2)
    relay.history[0]=(1,time.monotonic()-3)
    assert not relay.fresh(1)


@pytest.fixture
def server(device):
    xr2=Device(CONFIGS[1], 'other-udid', transport=FakeTransport(), ownership=lambda:False)
    xr2.relay.publish(jpeg())
    server=ControlServer([device,xr2], ROOT/'scripts/control/web',port=0, token='secret')
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    yield server
    server.shutdown(); server.server_close(); thread.join()


def test_http_auth_origin_size_and_routing(server):
    client=requests.Session(); client.trust_env=False
    origin=server.origin
    headers={'X-Control-Token':'secret','Origin':origin}
    def post(path, **kwargs): return client.post(origin+path, timeout=3, **kwargs)
    assert post('/api/XR1/connect',json={}).status_code==403
    assert post('/api/XR1/connect',json={},headers={'X-Control-Token':'secret','Origin':'http://evil.test'}).status_code==403
    assert post('/api/XR1/connect',json={},headers={**headers,'Host':'evil.test'}).status_code==403
    assert post('/api/UDID/connect',json={},headers=headers).status_code==404
    assert post('/api/XR1/connect',data='x'*48001,headers={**headers,'Content-Type':'application/json'}).status_code==413
    assert post('/api/XR1/connect',data='[]',headers={**headers,'Content-Type':'application/json'}).status_code==400
    response=post('/api/XR2/connect',json={},headers=headers)
    assert response.status_code==200
    assert not server.devices['XR1'].transport.calls
    device=server.devices['XR2']
    assert post('/api/XR2/activate',json=body(device),headers=headers).status_code==200
    assert device.transport.calls[-1][1].startswith('http://127.0.0.1:4726/')
    assert device.transport.calls[-1][2]['args']==[{'bundleId':'com.trueaxis.skate'}]
    assert client.get(origin+'/api/XR2/frame',headers=headers,timeout=3).headers['Content-Type']=='image/jpeg'
    assert client.get(origin+'/api/XR2/status',timeout=3).status_code==403
    html=client.get(origin+'/',timeout=3)
    assert 'secret' not in html.text and 'frame-ancestors' in html.headers['Content-Security-Policy']


def test_expired_session_can_reconnect(device):
    device.command('connect',{})
    old_epoch=device.epoch
    device.transport.active=[]
    with pytest.raises(ControlError,match='ownership'):
        device.command('gesture',body(device,points=PATH))
    assert not device.status()['connected']
    device.command('connect',{})
    assert device.epoch!=old_epoch


def test_discovery_flag_denied_fails_closed(device):
    def denied(*args, **kwargs):
        response=requests.Response(); response.status_code=500
        raise requests.HTTPError(response=response)
    device.transport.request=denied
    with pytest.raises(ControlError,match='session_discovery'):
        device.command('connect',{})
    assert not device.session and not device.uncertain


def test_mjpeg_reader_handles_split_frames_and_one_upstream(monkeypatch):
    import trueskate_ai.control.service as service
    relay=FrameRelay(9100)
    frame=jpeg()
    chunks=iter([b'--boundary\r\nContent-Type: image/jpeg\r\n\r\n'+frame[:1],frame[1:99],frame[99:]+b'\r\n',b''])
    opened=[]
    class Stream:
        def __enter__(self): return self
        def __exit__(self,*_): relay.stop.set()
        def read1(self,_): return next(chunks)
    def open_stream(url,timeout): opened.append(url); return Stream()
    monkeypatch.setattr(service,'urlopen',open_stream)
    relay.run()
    assert opened==['http://127.0.0.1:9100/']
    assert relay.snapshot()[0]==frame and relay.sequence==1
    # Browser consumers only read the relay; they cannot create another upstream.
    for _ in range(5): assert relay.snapshot()[0]==frame
    assert len(opened)==1


def test_collector_probe_fails_closed(device):
    def cannot_probe(): raise OSError('ps unavailable')
    device.ownership=cannot_probe
    with pytest.raises(ControlError,match='unavailable'): device.command('connect',{})
    assert not device.transport.calls


def test_geometry_change_and_old_frame_do_not_execute(device):
    device.command('connect',{})
    device.transport.size={'width':896,'height':414}
    with pytest.raises(ControlError,match='geometry'): device.command('gesture',body(device,points=PATH))
    assert not any(c[1].endswith('/actions') for c in device.transport.calls)


def test_status_detects_outage_and_session_expiry(device):
    device.command('connect',{})
    device.transport.fail='/appium/sessions'
    device.poll_status()
    assert not device.status()['available']
    device.transport.fail=None
    device.transport.active=[]
    device.last_probe=0
    device.poll_status()
    assert not device.status()['connected']


def test_home_presses_native_button(device):
    device.command('connect',{})
    device.command('home',body(device))
    assert device.transport.calls[-1][2]=={'script':'mobile: pressButton','args':[{'name':'home'}]}
    assert device.message=='Connected — command complete'


def test_system_swipes_use_screen_edges(device):
    device.command('connect',{})
    device.command('control_center',body(device))
    steps=device.transport.calls[-1][2]['actions'][0]['actions']
    assert device.transport.calls[-1][1].endswith('/actions')
    assert (steps[0]['x'],steps[0]['y'])==(373,0) and steps[-2]['y']==448
    device.command('app_switcher',body(device))
    steps=device.transport.calls[-1][2]['actions'][0]['actions']
    assert (steps[0]['x'],steps[0]['y'])==(207,892)
    assert steps[-2]=={'type':'pause','duration':800}


def test_system_commands_require_fresh_frame_and_ownership(device):
    device.command('connect',{})
    count=len(device.transport.calls)
    for kind in ['home','app_switcher','control_center']:
        with pytest.raises(ControlError,match='stale'): device.command(kind,body(device,frame=999))
    device.transport.active=['foreign-session']
    with pytest.raises(ControlError,match='ownership'): device.command('home',body(device))
    assert not any(c[0]=='POST' for c in device.transport.calls[count:])
