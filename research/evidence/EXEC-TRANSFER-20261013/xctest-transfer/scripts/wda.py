"""Tiny WDA HTTP client for the local simulator only (127.0.0.1:8199)."""
import json, time, urllib.request
BASE = 'http://127.0.0.1:8199'
assert BASE.startswith('http://127.0.0.1:'), 'local simulator only'

def call(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def new_session(bundle='com.l0.touchlogger'):
    r = call('POST', '/session', {'capabilities': {'alwaysMatch': {'bundleId': bundle, 'shouldWaitForQuiescence': False}}})
    return r['sessionId']

def actions(sid, payload):
    t0 = time.monotonic(); r = call('POST', f'/session/{sid}/actions', payload); t1 = time.monotonic()
    return r, t0, t1

def timing(sid, enabled=None):
    if enabled is None:
        return call('GET', f'/session/{sid}/wda/actionTiming')
    return call('POST', f'/session/{sid}/wda/actionTiming', {'enabled': enabled})
