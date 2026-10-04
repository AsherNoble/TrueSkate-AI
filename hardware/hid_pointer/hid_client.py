"""Bounded protocol-v2 client; the persistent bridge owns the serial port."""
import socket
import time

PROTOCOL_VERSION = 2
MAX_COMMAND_BYTES = 63
MAX_REPLY_BYTES = 1024


class PointerClient:
    def __init__(self, port=8765):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.pending = b''

    def send(self, line):
        lines = line.split('\n')
        if any(not part or len(part.encode('ascii')) > MAX_COMMAND_BYTES or '\r' in part for part in lines):
            raise ValueError('malformed or overlong pointer command')
        self.sock.settimeout(2)
        self.sock.sendall((line + '\n').encode('ascii'))

    def read_lines(self, wait=0.3):
        if wait < 0:
            raise ValueError('negative wait')
        end = time.monotonic() + wait
        lines = []
        while time.monotonic() < end:
            self.sock.settimeout(min(0.05, max(0.001, end - time.monotonic())))
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError('pointer control owner disconnected')
            self.pending += chunk
            while b'\n' in self.pending:
                raw, self.pending = self.pending.split(b'\n', 1)
                if len(raw) > MAX_REPLY_BYTES:
                    raise ValueError('overlong pointer reply')
                line = raw.decode('ascii').strip()
                if line:
                    lines.append(line)
            if len(self.pending) > MAX_REPLY_BYTES:
                raise ValueError('overlong pointer reply')
        return lines

    def ask(self, line, wait=0.3):
        self.send(line)
        lines = self.read_lines(wait)
        self.check_errors(lines)
        return lines

    @staticmethod
    def check_errors(lines):
        if any(line.startswith(('ERR', 'ABORT')) for line in lines):
            raise RuntimeError(f'pointer aborted or rejected command: {lines}')

    def wait_for(self, prefix, timeout=10.0):
        end, seen = time.monotonic() + timeout, []
        while time.monotonic() < end:
            batch = self.read_lines(min(0.1, end - time.monotonic()))
            seen.extend(batch)
            self.check_errors(batch)
            if any(line.startswith(prefix) for line in batch):
                return seen
        raise TimeoutError(f'no {prefix!r} within {timeout}s; saw {seen[-5:]}')

    def cancel(self):
        self.send('CANCEL')

    def close(self):
        self.sock.close()
