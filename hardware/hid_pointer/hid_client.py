"""Talk to the pointer through hid_bridge.py (never opens the serial port itself)."""
import socket
import time


class PointerClient:
    def __init__(self, port=8765):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=5)
        # Small command lines would otherwise wait on Nagle + delayed ACK (~100 ms each).
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(0.05)
        self.pending = b''

    def send(self, line):
        self.sock.sendall((line + '\n').encode())

    def read_lines(self, wait=0.3):
        end = time.time() + wait
        while time.time() < end:
            try:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                self.pending += chunk
            except socket.timeout:
                pass
        *lines, self.pending = self.pending.split(b'\n')
        return [l.decode(errors='replace').strip() for l in lines if l.strip()]

    def ask(self, line, wait=0.3):
        self.send(line)
        return self.read_lines(wait)

    def wait_for(self, prefix, timeout=10.0):
        end, seen = time.time() + timeout, []
        while time.time() < end:
            for l in self.read_lines(0.1):
                seen.append(l)
                if l.startswith(prefix):
                    return seen
        raise TimeoutError(f'no {prefix!r} within {timeout}s; saw {seen[-5:]}')

    def close(self):
        self.sock.close()
