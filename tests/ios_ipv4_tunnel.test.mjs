import test from 'node:test';
import assert from 'node:assert/strict';
import {openDeveloperProxy, validateProxyReply, validateServices, pemText, serveRegistry, parseWifiReady, wifiTunnel} from '../scripts/ops/ios_ipv4_tunnel.mjs';

test('plist Uint8Array credentials become PEM text, not comma-separated numbers', () => {
  const fixture = '-----BEGIN CERTIFICATE-----\nsynthetic fixture\n-----END CERTIFICATE-----';
  assert.equal(pemText(new Uint8Array(Buffer.from(fixture))), fixture);
  assert.equal(pemText(Buffer.from(fixture)), fixture);
  assert.throws(() => pemText(new Uint8Array([1, 2])), /not PEM/);
});

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

import {EventEmitter} from 'node:events';
import {handshakeRequest, parseHandshake, IPv6Packets, forwardPackets} from '../scripts/ops/ios_ipv4_node_tunnel.mjs';

test('CDTunnel handshake handles fragmentation, tail bytes and invalid parameters', () => {
  const info = {serverAddress:'fd00::1', serverRSDPort:58783, clientParameters:{address:'fd00::2',mtu:1280}};
  const body = Buffer.from(JSON.stringify(info));
  const head = Buffer.alloc(10); head.write('CDTunnel'); head.writeUInt16BE(body.length,8);
  const response = Buffer.concat([head,body,Buffer.from([1,2])]);
  for (let n=0;n<10+body.length;n++) assert.equal(parseHandshake(response.subarray(0,n)), null);
  assert.deepEqual(parseHandshake(response).info, info);
  assert.deepEqual(parseHandshake(response).tail, Buffer.from([1,2]));
  const request = handshakeRequest();
  assert.equal(JSON.parse(request.subarray(10)).mtu,1280);
  const corrupt = Buffer.from(response); corrupt[0]=0;
  assert.throws(() => parseHandshake(corrupt), /magic/);
  const invalid = Buffer.from(JSON.stringify({...info,serverRSDPort:0}));
  head.writeUInt16BE(invalid.length,8);
  assert.throws(() => parseHandshake(Buffer.concat([head,invalid])), /parameters/);
});

function packet(size=44) {
  const p=Buffer.alloc(size);p[0]=0x60;p.writeUInt16BE(size-40,4);return p;
}
test('TLS stream splitter preserves fragmented and coalesced IPv6 packets', () => {
  const p=packet(), q=packet(48), parser=new IPv6Packets();
  assert.deepEqual(parser.push(p.subarray(0,20)),[]);
  assert.deepEqual(parser.push(Buffer.concat([p.subarray(20),q])),[p,q]);
  assert.equal(parser.pending.length,0);
  const corrupt=Buffer.from(p);corrupt[0]=0x40;
  assert.throws(() => new IPv6Packets().push(corrupt), /Non-IPv6/);
});
test('packet forwarding preserves data and applies output backpressure', () => {
  const secure=new EventEmitter();
  let paused=0,resumed=0,callback,failed;
  const output=[],ingress=[];
  Object.assign(secure,{writableLength:0,write:b=>{output.push(b);return false;},pause:()=>{},resume:()=>{},destroy:()=>{}});
  const tun={startPolling:cb=>{callback=cb;},pausePolling:()=>paused++,resumePolling:()=>resumed++,write:b=>{ingress.push(b);return b.length;}};
  const stop=forwardPackets(secure,tun,Buffer.alloc(0),e=>{failed=e;});
  callback(packet());assert.equal(paused,1);secure.emit('drain');assert.equal(resumed,1);
  secure.emit('data',Buffer.concat([packet(),packet(48)]));assert.equal(ingress.length,2);
  assert.deepEqual(output,[packet()]);
  secure.emit('close');assert.equal(failed,'Paired TLS transport closed');
  stop();assert.equal(secure.listenerCount('data'),0);
});

function fakeChild() {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stdin = {end: () => {}};
  child.exitCode = null; child.signalCode = null; child.signals = [];
  child.kill = (signal) => { child.signals.push(signal); child.signalCode = signal; setImmediate(() => child.emit('exit', null, signal)); };
  return child;
}
const wifiOpts = {'tunnel-python': '/py', 'pair-dir': '/pairs', udid: 'xr2', host: 'Test-XR-2.local', lifetime: 900};

test('Wi-Fi ready line is validated and errors fail closed', () => {
  assert.equal(parseWifiReady('not json'), null);
  assert.equal(parseWifiReady('{"event":"other"}'), null);
  assert.deepEqual(parseWifiReady('{"event":"pmd3-tunnel-ready","address":"fd2c::1","rsd_port":61250,"interface":"utun9"}'),
    {Address: 'fd2c::1', RsdPort: 61250, interface: 'utun9'});
  assert.throws(() => parseWifiReady('{"event":"pmd3-tunnel-ready","address":"x; rm","rsd_port":1}'), /invalid parameters/);
  assert.throws(() => parseWifiReady('{"event":"pmd3-tunnel-error","error":"ConnectionError"}'), /Wi-Fi tunnel: ConnectionError/);
});
test('Wi-Fi tunnel reports unexpected exit as loss', async () => {
  const child = fakeChild(); let dead = 0;
  const pending = wifiTunnel(wifiOpts, () => dead++, () => child);
  child.stdout.emit('data', '{"event":"pmd3-tunnel-ready","address":"fd2c::1","rsd_port":61250}\n');
  await pending;
  child.emit('exit', 1, null);
  assert.equal(dead, 1);
});
test('Wi-Fi tunnel resolves on ready and stops its child without reporting loss', async () => {
  const child = fakeChild(); let argv; let dead = 0;
  const pending = wifiTunnel(wifiOpts, () => dead++, (cmd, args) => { argv = [cmd, ...args]; return child; });
  child.stdout.emit('data', '{"event":"pmd3-tunnel-ready","address":"fd2c::1",');
  child.stdout.emit('data', '"rsd_port":61250,"interface":"utun9"}\n');
  const tunnel = await pending;
  assert.equal(argv[0], '/py');
  assert.deepEqual(argv.slice(2), ['--udid', 'xr2', '--host', 'Test-XR-2.local', '--pair-dir', '/pairs', '--lifetime', '900']);
  assert.equal(tunnel.RsdPort, 61250);
  await tunnel.closer();
  assert.deepEqual(child.signals, ['SIGTERM']);
  assert.equal(dead, 0);
});
test('Wi-Fi tunnel exit before ready rejects without reporting loss', async () => {
  const child = fakeChild(); let dead = 0;
  const pending = wifiTunnel(wifiOpts, () => dead++, () => child);
  child.stdout.emit('data', '{"event":"pmd3-tunnel-error","error":"RuntimeError: RemotePairing record missing"}\n');
  await assert.rejects(pending, /record missing/);
  assert.equal(dead, 0);
});
