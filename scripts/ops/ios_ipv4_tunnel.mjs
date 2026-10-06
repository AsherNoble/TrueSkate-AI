#!/usr/bin/env node
/** Bounded paired-IPv4 diagnostic. Never changes the installed Appium modules. */
import net from 'node:net';
import http from 'node:http';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import dns from 'node:dns/promises';
import fs from 'node:fs';

const require = createRequire(import.meta.url);
const emit = (event, details = {}) => process.stdout.write(JSON.stringify({event, ...details}) + '\n');
export class DiagnosticError extends Error {}

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
  if (!['probe', 'tunnel', 'wda', 'registry-info'].includes(opts.mode)) throw new DiagnosticError('Expected probe, tunnel, wda or registry-info mode');
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
  try {
    await client.queryType(3000);
    session = await client.startSession(pair.HostID, pair.SystemBUID, 3000);
    if (session.enableSessionSSL) {
      // Fix duplicate response piping only on this temporary client instance.
      ps._splitter.unpipe(ps._decoder);
      client.enableSessionSSL(pair.HostPrivateKey, pair.HostCertificate);
    }
    validateIdentity(await client.getValue({Key: 'UniqueDeviceID'}, 3000), opts.udid);
    const version = await client.getValue({Key: 'ProductVersion'}, 3000);
    return {client, pair, host, version, close};
  } catch (error) { await close(); throw error; }
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
      if (tunnel) await sdk.TunnelManager.closeTunnelByAddress(tunnel.Address);
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
    await auth.close(); auth = null;
    tunnel = await bounded(sdk.TunnelManager.getTunnel(proxy.socket, {
      cert: proxy.cert.toString(), key: proxy.key.toString(),
    }, {onDead: () => {
      handler?.removeTunnelEntry(opts.udid);
      emit('tunnel-lost');
    }}), 20000, 'Native tunnel establishment');
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
    emit('helper-error', {error: error instanceof DiagnosticError ? error.message : error.code || error.name});
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
    console.log('ios_ipv4_tunnel.mjs probe|tunnel|wda|registry-info --modules-root ABS --udid UDID [--host DNS_OR_IPV4] [--lifetime 1..900]');
    return;
  }
  const opts = parse(argv);
  if (opts.mode === 'registry-info') {
    const {strongbox, BaseItem} = require(path.join(opts['modules-root'], '@appium/strongbox'));
    const item = new BaseItem('tunnelRegistryPort', strongbox('appium-xcuitest-driver'));
    emit('registry-info', {file: item.id, value: await item.read()});
  } else if (opts.mode === 'probe') {
    const auth = await pairedLockdown(opts);
    try { emit('paired', {host: auth.host, ios: auth.version}); }
    finally { await auth.close(); }
  } else if (opts.mode === 'tunnel') await runTunnel(opts);
  else await runWda(opts);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch(error => {
    emit('helper-error', {error: error instanceof DiagnosticError ? error.message : error.code || error.name});
    process.exitCode = 1;
  });
}
