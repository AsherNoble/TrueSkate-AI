/** Node TLS + installed native TUN, isolated from the native OpenSSL forwarder. */
import tls from 'node:tls';
import net from 'node:net';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

export class NodeTunnelError extends Error {}

export function handshakeRequest(mtu = 1280) {
  const body = Buffer.from(JSON.stringify({type: 'clientHandshakeRequest', mtu}));
  const head = Buffer.alloc(10);
  head.write('CDTunnel'); head.writeUInt16BE(body.length, 8);
  return Buffer.concat([head, body]);
}

export function parseHandshake(buffer) {
  if (buffer.length < 10) return null;
  if (buffer.subarray(0, 8).toString() !== 'CDTunnel') throw new NodeTunnelError('Invalid CDTunnel response magic');
  const length = buffer.readUInt16BE(8);
  if (!length || length > 32768) throw new NodeTunnelError('Invalid CDTunnel response size');
  if (buffer.length < 10 + length) return null;
  const info = JSON.parse(buffer.subarray(10, 10 + length).toString());
  if (!net.isIPv6(info.serverAddress) || !net.isIPv6(info.clientParameters?.address) ||
      !Number.isInteger(info.serverRSDPort) || info.serverRSDPort < 1 || info.serverRSDPort > 65535 ||
      !Number.isInteger(info.clientParameters.mtu) || info.clientParameters.mtu < 1280 || info.clientParameters.mtu > 65535) {
    throw new NodeTunnelError('Invalid CDTunnel response parameters');
  }
  return {info, tail: buffer.subarray(10 + length)};
}

export class IPv6Packets {
  pending = Buffer.alloc(0);
  push(chunk) {
    this.pending = Buffer.concat([this.pending, chunk]);
    if (this.pending.length > 2 * 1024 * 1024) throw new NodeTunnelError('IPv6 ingress buffer exceeded');
    const packets = [];
    let offset = 0;
    while (this.pending.length-offset >= 40) {
      if ((this.pending[offset] >> 4) !== 6) throw new NodeTunnelError('Non-IPv6 tunnel packet');
      const size = 40 + this.pending.readUInt16BE(offset + 4);
      if (size > 65535) throw new NodeTunnelError('IPv6 packet exceeds native TUN limit');
      if (this.pending.length-offset < size) break;
      packets.push(this.pending.subarray(offset, offset + size));
      offset += size;
    }
    this.pending = this.pending.subarray(offset);
    return packets;
  }
}

export async function nodeHandshake(socket, credentials, mtu = 16000) {
  const secure = tls.connect({socket, cert: Buffer.from(credentials.cert), key: Buffer.from(credentials.key),
    rejectUnauthorized: false, minVersion: 'TLSv1.2', maxVersion: 'TLSv1.2'});
  // Pairing identity and authenticated lockdown were checked before this service.
  // Keep an error listener through both handshakes and ownership transfer.
  secure.on('error', () => {});
  try {
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new NodeTunnelError('Paired Node TLS timeout')), 8000);
      const done = (error) => {
        clearTimeout(timer); secure.off('error', failed); secure.off('secureConnect', ready);
        error ? reject(error) : resolve();
      };
      const ready = () => done();
      const failed = error => done(error);
      secure.once('secureConnect', ready); secure.once('error', failed);
    });
    const result = await new Promise((resolve, reject) => {
      let buffer = Buffer.alloc(0);
      const timer = setTimeout(() => done(new NodeTunnelError('CDTunnel handshake timeout')), 8000);
      const done = (error, parsed) => {
        clearTimeout(timer); secure.off('data', data); secure.off('error', failed); secure.off('close', closed);
        secure.pause(); error ? reject(error) : resolve(parsed);
      };
      const failed = error => done(error);
      const closed = () => done(new NodeTunnelError('CDTunnel closed during handshake'));
      const data = chunk => {
        try {
          buffer = Buffer.concat([buffer, chunk]);
          if (buffer.length > 65536) throw new NodeTunnelError('CDTunnel handshake buffer exceeded');
          const parsed = parseHandshake(buffer);
          if (parsed) done(null, parsed);
        } catch (error) { done(error); }
      };
      secure.on('data', data); secure.once('error', failed); secure.once('close', closed);
      secure.write(handshakeRequest(mtu));
    });
    return {secure, ...result};
  } catch (error) { secure.destroy(); throw error; }
}

export function forwardPackets(secure, tun, tail, onDead) {
  const parser = new IPv6Packets();
  const queue = [];
  let bytes = 0, timer, closing = false, pumping = false, stalledAt = null;
  const die = error => { if (!closing) { closing = true; onDead(error.message); } };
  const pump = () => {
    if (closing) return;
    pumping = true;
    let count = 0;
    while (queue.length && count++ < 64) {
      const packet = queue[0];
      try {
        if (tun.write(packet) !== packet.length) throw new NodeTunnelError('Partial native TUN write');
        queue.shift(); bytes -= packet.length; stalledAt = null;
      } catch (error) {
        if (/temporarily unavailable|EAGAIN|EWOULDBLOCK/.test(error.message)) {
          stalledAt ??= Date.now();
          if (Date.now() - stalledAt > 2000) { die(new NodeTunnelError('TUN write stall')); return; }
          timer = setTimeout(pump, 1); return;
        }
        die(error); return;
      }
    }
    if (queue.length) timer = setImmediate(pump);
    else { pumping = false; secure.resume(); }
  };
  const incoming = chunk => {
    if (closing) return;
    try {
      for (const packet of parser.push(chunk)) { queue.push(packet); bytes += packet.length; }
      if (bytes > 2 * 1024 * 1024) throw new NodeTunnelError('TUN ingress queue exceeded');
      if (queue.length > 64) secure.pause();
      if (!pumping) pump();
    } catch (error) { die(error); }
  };
  const failed = error => die(error);
  const closed = () => die(new NodeTunnelError('Paired TLS transport closed'));
  const drained = () => { if (!closing) tun.resumePolling(); };
  secure.on('data', incoming); secure.on('error', failed); secure.once('close', closed); secure.on('drain', drained);
  tun.startPolling(packet => {
    if (closing) return;
    try {
      // Poll callbacks are packet based; cap queued TLS output and propagate backpressure.
      if (secure.writableLength > 2 * 1024 * 1024) throw new NodeTunnelError('TLS egress queue exceeded');
      if (!secure.write(packet)) tun.pausePolling();
    } catch (error) { die(error); }
  }, 65535, 8);
  if (tail.length) incoming(tail);
  secure.resume();
  return () => {
    closing = true; clearTimeout(timer); clearImmediate(timer);
    secure.off('data', incoming); secure.off('error', failed); secure.off('close', closed); secure.off('drain', drained);
    tun.pausePolling(); secure.destroy();
  };
}

export async function nodeTunnel(socket, credentials, modulesRoot, onDead) {
  const {TunTap} = await import(pathToFileURL(path.join(modulesRoot, 'appium-ios-tuntap/lib/TunTap.js')).href);
  const {secure, info, tail} = await nodeHandshake(socket, credentials);
  const tun = new TunTap();
  let stop, routed = false, closed = false;
  const closer = async () => {
    if (closed) return;
    closed = true;
    if (stop) stop(); else secure.destroy();
    try { if (routed && tun.isOpen) await tun.removeRoute(info.serverAddress+'/128'); }
    finally { tun.close(); }
  };
  try {
    tun.open();
    await tun.configure(info.clientParameters.address, info.clientParameters.mtu);
    await tun.addRoute(info.serverAddress+'/128'); routed = true;
    stop = forwardPackets(secure, tun, tail, onDead);
    return {Address: info.serverAddress, RsdPort: info.serverRSDPort, closer, interface: tun.name};
  } catch (error) { await closer(); throw error; }
}
