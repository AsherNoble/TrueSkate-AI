import test from 'node:test';
import assert from 'node:assert/strict';
import {openDeveloperProxy, validateProxyReply, validateServices, serveRegistry} from '../scripts/ops/ios_ipv4_tunnel.mjs';

test('identity mismatch never starts developer service', async () => {
  let started = false;
  const auth = {client: {getValue: async () => 'other', startService: async () => {started = true;}}};
  await assert.rejects(openDeveloperProxy(auth, 'xr2', '127.0.0.1'), /identity mismatch/);
  assert.equal(started, false);
});
test('missing service and invalid TLS reply fail closed', () => {
  for (const reply of [{Error: 'InvalidService'}, {Port: 10}, {Port: 0, EnableServiceSSL: true}]) {
    assert.throws(() => validateProxyReply(reply));
  }
  assert.throws(() => validateServices({'com.apple.instruments.dtservicehub': {port: 1}}), /Required developer service missing/);
});
test('transport loss propagates without retrying service start', async () => {
  let starts = 0;
  const auth = {pair: {}, client: {getValue: async () => 'xr2', startService: async () => {
    starts++; return {Port: 1234, EnableServiceSSL: true};
  }}};
  await assert.rejects(openDeveloperProxy(auth, 'xr2', '127.0.0.1', async () => {throw new Error('transport lost');}), /transport lost/);
  assert.equal(starts, 1);
});
test('diagnostic registry binds loopback and closes owned listener', async () => {
  const server = await serveRegistry({handleRequest: (_req, res) => res.end('fixture')}, 0);
  assert.equal(server.address().address, '127.0.0.1');
  const response = await fetch(`http://127.0.0.1:${server.address().port}/`);
  assert.equal(await response.text(), 'fixture');
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  assert.equal(server.listening, false);
});
