"""Persistent serial bridge with one exclusive TCP control owner.

Usage: hid_bridge.py <serial_port> <logfile> [tcp_port=8765]
Owner disconnect, malformed input or output failure sends CANCEL. Opening the
serial port resets the board, so it remains open for the life of the bridge.
"""
import socket
import sys
import threading
import time

MAX_COMMAND_BYTES = 63


class Bridge:
    def __init__(self, serial_port, log):
        self.serial = serial_port
        self.log = log
        self.owner = None
        self.lock = threading.RLock()

    def claim(self, conn):
        with self.lock:
            if self.owner is not None:
                conn.sendall(b'ERR busy\n')
                conn.close()
                return False
            conn.settimeout(1)
            self.owner = conn
            return True

    def release(self, conn):
        with self.lock:
            if self.owner is conn:
                try:
                    self.serial.write(b'CANCEL\n')
                finally:
                    self.owner = None
            conn.close()

    def relay(self, line):
        with self.lock:
            if self.owner is not None:
                conn = self.owner
                try:
                    conn.sendall(line)
                except OSError:
                    self.release(conn)

    def pump_serial(self):
        while True:
            line = self.serial.readline(4096)
            if line:
                self.log.write(f'{time.time():.3f} {line.decode(errors="replace")}')
                self.relay(line)

    def serve(self, conn):
        buffer = b''
        try:
            while True:
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer += data
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if not line or len(line) > MAX_COMMAND_BYTES or any(c < 32 or c > 126 for c in line):
                        raise ValueError('malformed or overlong command')
                    with self.lock:
                        if self.owner is not conn:
                            return
                        self.serial.write(line + b'\n')
                if len(buffer) > MAX_COMMAND_BYTES:
                    raise ValueError('overlong command')
        except (OSError, ValueError):
            pass
        finally:
            self.release(conn)


def main():
    import serial
    serial_port, log_path = sys.argv[1:3]
    tcp_port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
    port = serial.Serial()
    port.port, port.baudrate, port.timeout, port.write_timeout = serial_port, 115200, 0.1, 2
    port.dtr = port.rts = False
    port.open()
    with open(log_path, 'a', buffering=1) as log, socket.socket() as server:
        bridge = Bridge(port, log)
        log.write(f'--- bridge start {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n')
        threading.Thread(target=bridge.pump_serial, daemon=True).start()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', tcp_port))
        server.listen()
        while True:
            conn, _ = server.accept()
            conn.settimeout(1)
            if bridge.claim(conn):
                threading.Thread(target=bridge.serve, args=(conn,), daemon=True).start()


if __name__ == '__main__':
    main()
