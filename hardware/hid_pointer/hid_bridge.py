"""Own the pointer's serial port for good: log its output and relay commands over TCP.

Opening or closing the CH340 port pulses the ESP32's auto-reset line, which
drops the Bluetooth connection. This bridge opens the port once and keeps it.
Clients connect to 127.0.0.1:<tcp_port>, send command lines, and receive every
line the board prints until they disconnect.
Usage: hid_bridge.py <serial_port> <logfile> [tcp_port=8765]
"""
import socket, sys, threading, time
import serial

serial_port, log_path = sys.argv[1], sys.argv[2]
tcp_port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
s = serial.Serial()
s.port, s.baudrate, s.timeout = serial_port, 115200, 0.1
s.dtr = False
s.rts = False
s.open()
clients, lock = [], threading.Lock()
log = open(log_path, 'a', buffering=1)
log.write(f'--- bridge start {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n')


def pump_serial():
    while True:
        line = s.readline()
        if not line:
            continue
        log.write(f'{time.time():.3f} {line.decode(errors="replace")}')
        with lock:
            for c in list(clients):
                try:
                    c.sendall(line)
                except OSError:
                    clients.remove(c)


def serve(conn):
    with lock:
        clients.append(conn)
    buffer = b''
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                s.write(line + b'\n')
    finally:
        with lock:
            if conn in clients:
                clients.remove(conn)
        conn.close()


threading.Thread(target=pump_serial, daemon=True).start()
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', tcp_port))
server.listen()
while True:
    conn, _ = server.accept()
    threading.Thread(target=serve, args=(conn,), daemon=True).start()
