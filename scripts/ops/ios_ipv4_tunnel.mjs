#!/usr/bin/env node
/** Bounded paired-IPv4 diagnostic. Never changes the installed Appium modules. */
import net from 'node:net';
import http from 'node:http';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import dns from 'node:dns/promises';
import fs from 'node:fs';
import tls from 'node:tls';
import {NodeTunnelError, nodeHandshake, nodeTunnel} from './ios_ipv4_node_tunnel.mjs';

const require = createRequire(import.meta.url);
const emit = (event, details = {}) => process.stdout.write(JSON.stringify({event, ...details}) + '\n');
export class DiagnosticError extends Error {}
const safeFailure = error => error instanceof DiagnosticError || error instanceof NodeTunnelError || /^(SSL_connect|Failed to (load TLS|send CDTunnel|read CDTunnel|parse handshake)|Tunnel handshake timeout|Invalid CDTunnel|Handshake response|Malformed clientParameters|TLS session)/.test(error.message || '')
  ? error.message : error.code || error.name;

export function pemText(value) {
  const text = typeof value === 'string' ? value : Buffer.from(value).toString('utf8');
  if (!text.startsWith('-----BEGIN ')) throw new DiagnosticError('Pairing credential is not PEM');
  return text;
}

export function validateIdentity(actual, expected) {
  if (!expected || actual !== expected) throw new DiagnosticError('Paired device identity mismatch');
}

export function validateProxyReply(reply) {
  if (reply.Error) throw new DiagnosticError(`Developer proxy rejected request: ${reply.Error}`);
  if (!Number.isInteger(reply.Port) || reply.Port < 1 || reply.Port > 65535 || !reply.EnableServiceSSL) {
    throw new DiagnosticError('Developer proxy did not return a valid TLS service port');
  }
  return reply.Port;
}

export function validateServices(services) {
  for (const name of ['com.apple.instruments.dtservicehub', 'com.apple.dt.testmanagerd.remote',
    'com.apple.mobile.installation_proxy.shim.remote']) {
    if (!services[name]) throw new DiagnosticError(`Required developer service missing: ${name}`);
  }
}

export async function openDeveloperProxy(auth, udid, host, connect = connectTcp) {
  validateIdentity(await auth.client.getValue({Key: 'UniqueDeviceID'}, 3000), udid);
  // The installed older plist encoder cannot serialize a null EscrowBag.
  const reply = await auth.client.startService('com.apple.internal.devicecompute.CoreDeviceProxy', 3000);
  const socket = await connect(host, validateProxyReply(reply));
  return {socket, cert: auth.pair.HostCertificate, key: auth.pair.HostPrivateKey};
}

export function connectTcp(host, port) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({host, port});
    socket.on('error', () => {});
    const timer = setTimeout(() => socket.destroy(new Error('TCP connection timeout')), 5000);
    socket.once('error', (error) => { clearTimeout(timer); reject(error); });
    socket.once('connect', () => { clearTimeout(timer); resolve(socket); });
  });
}

export async function bounded(promise, ms, label) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new DiagnosticError(`${label} timed out`)), ms);
    })]);
  } finally { clearTimeout(timer); }
}

function parse(argv) {
  const opts = {mode: argv[0], host: 'Test-XR-2.local', port: 42315, lifetime: 900};
  for (let i = 1; i < argv.length; i += 2) {
    if (!argv[i].startsWith('--') || argv[i + 1] === undefined) throw new DiagnosticError('Expected --key value');
    opts[argv[i].slice(2)] = argv[i + 1];
  }
  opts.port = Number(opts.port); opts.lifetime = Number(opts.lifetime);
  if (opts.mtu && ![1280, 16000].includes(Number(opts.mtu))) throw new DiagnosticError('Probe MTU must be 1280 or 16000');
  if (!['probe', 'tls-probe', 'native-tls-probe', 'tunnel', 'wda', 'registry-info'].includes(opts.mode)) throw new DiagnosticError('Expected probe, tls-probe, native-tls-probe, tunnel, wda or registry-info mode');
  if (!opts['modules-root'] || !path.isAbsolute(opts['modules-root'])) throw new DiagnosticError('An absolute --modules-root is required');
  if (opts.mode !== 'registry-info' && !opts.udid) throw new DiagnosticError('--udid is required');
  if (opts.port !== 42315 || !Number.isInteger(opts.lifetime) || opts.lifetime < 1 || opts.lifetime > 900) {
    throw new DiagnosticError('Diagnostic port is fixed at 42315; lifetime must be 1..900 seconds');
  }
  return opts;
}

async function pairedLockdown(opts) {
  const base = path.join(opts['modules-root'], 'appium-ios-device/build/lib');
  const {Usbmux, getDefaultSocket} = require(path.join(base, 'usbmux'));
  const {Lockdown} = require(path.join(base, 'lockdown'));
  const {PlistService} = require(path.join(base, 'plist-service'));
  const {address: host} = await dns.lookup(opts.host, {family: 4});
  const mux = new Usbmux(await getDefaultSocket({timeout: 3000}));
  let pair;
  try { pair = await mux.readPairRecord(opts.udid); }
  finally { mux.close(); mux._socketClient.destroy(); }
  if (!pair?.HostPrivateKey || !pair?.HostCertificate) throw new DiagnosticError('Existing pairing credentials unavailable');
  const socket = await connectTcp(host, 62078);
  const ps = new PlistService(socket);
  const client = new Lockdown(ps);
  let session;
  const close = async () => {
    try {
      if (session) await ps.sendPlistAndReceive({Label: 'appium-internal', Request: 'StopSession', SessionID: session.sessionID}, 1000);
    } catch { /* Socket destruction still closes the session. */ }
    ps.close(); socket.destroy();
  };
  let stage = 'query-type';
  try {
    await client.queryType(3000);
    stage = 'start-session';
    session = await client.startSession(pair.HostID, pair.SystemBUID, 3000);
    if (session.enableSessionSSL) {
      // Fix duplicate response piping only on this temporary client instance.
      ps._splitter.unpipe(ps._decoder);
      client.enableSessionSSL(pair.HostPrivateKey, pair.HostCertificate);
    }
    stage = 'identity';
    validateIdentity(await client.getValue({Key: 'UniqueDeviceID'}, 3000), opts.udid);
    stage = 'version';
    const version = await client.getValue({Key: 'ProductVersion'}, 3000);
    return {client, pair, host, version, close};
  } catch (error) {
    await close();
    const message = String(error.message || error.name);
    throw new DiagnosticError(`Lockdown ${stage}: ${/BEGIN|HostPrivateKey|HostCertificate|PairRecordData/.test(message) ? 'redacted credential error' : message.slice(0,160)}`);
  }
}

async function loadSdk(opts) {
  return await import(pathToFileURL(path.join(opts['modules-root'], 'appium-ios-remotexpc/build/src/index.js')).href);
}

export async function serveRegistry(handler, port = 42315) {
  // SDK start() binds all interfaces; wrap its pinned JS handler to bind loopback.
  const server = http.createServer((req, res) => {
    Promise.resolve(handler.handleRequest(req, res)).catch(() => {
      if (!res.headersSent) res.writeHead(500);
      res.end();
    });
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  return server;
}

async function runTunnel(opts) {
  if (process.getuid?.() !== 0) throw new DiagnosticError('Native tunnel creation requires administrator authentication');
  const checkLease = () => {
    if (!opts['lease-file'] || !path.isAbsolute(opts['lease-file']) || !opts['lease-token']) throw new DiagnosticError('Coordinator lease required');
    const lease = JSON.parse(fs.readFileSync(opts['lease-file'], 'utf8'));
    if (lease.token !== opts['lease-token'] || Date.now()/1000 - lease.updated > 15 || !Number.isInteger(lease.pid)) throw new DiagnosticError('Coordinator lease expired');
    process.kill(lease.pid, 0);
  };
  checkLease();
  const sdk = await loadSdk(opts);
  let auth, proxy, tunnel, server, handler, exiting = false;
  const cleanup = async (reason, code = 0) => {
    if (exiting) return;
    exiting = true;
    const force = setTimeout(() => process.exit(1), 8000);
    try {
      if (handler) handler.removeTunnelEntry(opts.udid);
      if (server) { server.closeAllConnections(); await new Promise(r => server.close(r)); }
      if (tunnel) await tunnel.closer();
      proxy?.socket.destroy();
      if (auth) await auth.close();
      emit('helper-stopped', {reason});
    } finally { clearTimeout(force); process.exit(code); }
  };
  process.once('SIGTERM', () => void cleanup('SIGTERM'));
  process.once('SIGINT', () => void cleanup('SIGINT'));
  setTimeout(() => void cleanup('lifetime-limit', 1), opts.lifetime * 1000);
  setInterval(() => { try { checkLease(); } catch { void cleanup('coordinator-lost', 1); } }, 2000);
  try {
    auth = await pairedLockdown(opts);
    emit('paired', {host: auth.host, ios: auth.version});
    proxy = await openDeveloperProxy(auth, opts.udid, auth.host);
    const onDead = () => {
      handler?.removeTunnelEntry(opts.udid);
      emit('tunnel-lost');
      void cleanup('transport-lost', 1);
    };
    tunnel = await bounded(nodeTunnel(proxy.socket, proxy, opts['modules-root'], onDead), 25000, 'Node TLS tunnel establishment');
    await auth.close(); auth = null;
    const services = sdk.servicesToCatalog(await bounded(
      sdk.discoverServices(opts.udid, tunnel.Address, tunnel.RsdPort), 20000, 'RSD discovery'));
    validateServices(services);
    const registry = {tunnels: {}, metadata: {}};
    handler = new sdk.TunnelRegistryServer(registry, opts.port, {
      refreshServices: async (udid, entry) => ({...entry, services: sdk.servicesToCatalog(
        await bounded(sdk.discoverServices(udid, entry.address, entry.rsdPort), 20000, 'RSD refresh'))}),
    });
    // loadRegistry must first select the same mutable object as the HTTP handler.
    handler.registry = registry;
    handler.upsertReadyEntry(opts.udid, {
      udid: opts.udid, address: tunnel.Address, rsdPort: tunnel.RsdPort, services,
    });
    server = await serveRegistry(handler, opts.port);
    emit('helper-ready', {pid: process.pid, host: tunnel.Address, rsd_port: tunnel.RsdPort,
      registry_port: opts.port, service_names: Object.keys(services)});
  } catch (error) {
    emit('helper-error', {error: safeFailure(error)});
    await cleanup('startup-failed', 1);
  }
}

async function runWda(opts) {
  const sdk = await loadSdk(opts);
  const runnerId = 'com.asher.WebDriverAgentRunner.xctrunner';
  const dvt = await sdk.Services.startDVTService(opts.udid);
  try {
    const pid = await dvt.processControl.getPidForBundleIdentifier(runnerId);
    if (pid) throw new DiagnosticError('An existing WDA runner is present; refusing to launch or terminate it');
  } finally { await dvt.dvtService.close(); }
  const runner = new sdk.XCTestRunner({udid: opts.udid,
    testRunnerBundleId: runnerId, xctestBundleId: 'com.asher.WebDriverAgentRunner',
    appUnderTestBundleId: 'com.trueaxis.skate', killExisting: false,
    timeoutMs: 600000, launchEnvironment: {USE_PORT: '8100'}});
  let exiting = false;
  const cleanup = async () => {
    if (exiting) return;
    exiting = true;
    const force = setTimeout(() => process.exit(1), 8000);
    try { await runner.close(); emit('wda-stopped'); }
    finally { clearTimeout(force); process.exit(0); }
  };
  process.once('SIGTERM', () => void cleanup());
  process.once('SIGINT', () => void cleanup());
  runner.on('step', (step) => emit('wda-step', {step}));
  const result = await runner.run();
  if (!exiting) {
    emit('wda-ended', {status: result.status});
    process.exitCode = result.status === 'passed' ? 0 : 1;
  }
}

async function main(argv) {
  if (argv[0] === '--help') {
    console.log('ios_ipv4_tunnel.mjs probe|tls-probe|native-tls-probe|tunnel|wda|registry-info --modules-root ABS --udid UDID [--host DNS_OR_IPV4] [--lifetime 1..900]');
    return;
  }
  const opts = parse(argv);
  if (opts.mode === 'registry-info') {
    const {strongbox, BaseItem} = require(path.join(opts['modules-root'], '@appium/strongbox'));
    const item = new BaseItem('tunnelRegistryPort', strongbox('appium-xcuitest-driver'));
    emit('registry-info', {file: item.id, value: await item.read()});
  } else if (opts.mode === 'probe') {
    const auth = await pairedLockdown(opts);
    emit('probe-stage', {stage: 'paired'});
    try { emit('paired', {host: auth.host, ios: auth.version}); }
    finally { await auth.close(); }
  } else if (opts.mode === 'native-tls-probe') {
    if (opts['load-sdk'] === 'yes') await loadSdk(opts);
    const {TunnelForwarder} = await import(pathToFileURL(path.join(opts['modules-root'], 'appium-ios-tuntap/lib/tunnel/forwarder.js')).href);
    const auth = await pairedLockdown(opts);
    const forwarder = new TunnelForwarder();
    let proxy, destroy;
    try {
      proxy = await openDeveloperProxy(auth, opts.udid, auth.host);
      emit('probe-stage', {stage: 'proxy-open'});
      if (['yes', 'nodelay'].includes(opts['socket-options'])) proxy.socket.setNoDelay(true);
      if (['yes', 'keepalive'].includes(opts['socket-options'])) proxy.socket.setKeepAlive(true, 1000);
      if (opts['close-before-tls'] === 'yes') await auth.close();
      if (opts['read-stop'] === 'yes') {
        proxy.socket.pause();
        proxy.socket._handle.readStop();
        await new Promise(resolve => setImmediate(resolve));
      }
      if (opts['retain-socket'] === 'yes') {
        destroy = proxy.socket.destroy.bind(proxy.socket);
        proxy.socket.destroy = () => proxy.socket;
      }
      forwarder.connect(proxy.socket, {cert: pemText(proxy.cert), key: pemText(proxy.key)});
      emit('native-tls-ready', {host: auth.host, read_stop: opts['read-stop'] === 'yes'});
      if (opts.handshake === 'yes') {
        const info = forwarder.handshake(16000);
        emit('native-handshake-ready', {server_address: info.serverAddress, rsd_port: info.serverRSDPort});
      }
    } finally { forwarder.stop(); if (destroy) destroy(); else proxy?.socket.destroy(); await auth.close(); }
  } else if (opts.mode === 'tls-probe') {
    const auth = await pairedLockdown(opts);
    emit('probe-stage', {stage: 'paired'});
    let proxy, secure;
    try {
      proxy = await openDeveloperProxy(auth, opts.udid, auth.host);
      emit('probe-stage', {stage: 'proxy-open'});
      if (opts['close-before-tls'] === 'yes') await auth.close();
      if (opts.handshake === 'yes') {
        const result = await nodeHandshake(proxy.socket, proxy, opts.mtu ? Number(opts.mtu) : 16000);
        secure = result.secure;
        emit('node-handshake-ready', {server_address: result.info.serverAddress, rsd_port: result.info.serverRSDPort,
          client_address: result.info.clientParameters.address, mtu: result.info.clientParameters.mtu});
        return;
      }
      secure = tls.connect({socket: proxy.socket, cert: Buffer.from(proxy.cert), key: Buffer.from(proxy.key),
        rejectUnauthorized: false, minVersion: 'TLSv1.2', maxVersion: 'TLSv1.2'});
      await bounded(new Promise((resolve, reject) => {secure.once('secureConnect', resolve); secure.once('error', reject);}), 8000, 'Paired TLS probe');
      emit('tls-ready', {host: auth.host, close_before_tls: opts['close-before-tls'] === 'yes'});
    } catch (error) {
      const message = String(error.message || error.name);
      emit('probe-failure', {error: /BEGIN|HostPrivateKey|HostCertificate/.test(message) ? 'redacted credential error' : message.slice(0,200)});
      throw error;
    } finally { secure?.destroy(); proxy?.socket.destroy(); await auth.close(); }
  } else if (opts.mode === 'tunnel') await runTunnel(opts);
  else await runWda(opts);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch(error => {
    emit('helper-error', {error: safeFailure(error)});
    if (['probe', 'tls-probe', 'native-tls-probe'].includes(process.argv[2])) {
      const message = String(error.message || error.name);
      emit('probe-detail', {error: /BEGIN|PrivateKey|Certificate|PairRecord|Uint8Array|Buffer</.test(message) ? 'redacted credential error' : message.slice(0,200)});
    }
    process.exitCode = 1;
  });
}
