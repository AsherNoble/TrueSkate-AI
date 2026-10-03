"""Hold the pointer's serial port open and append everything it prints to a log.

Opening a CH340 port normally pulses DTR/RTS, which resets the ESP32 and loses
early handshake messages, so both lines are deasserted before opening.
Usage: hid_log.py <port> <logfile>   (stop with Ctrl-C or SIGTERM)
"""
import signal, sys, time
import serial

port, path = sys.argv[1], sys.argv[2]
s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0.2
s.dtr = False
s.rts = False
s.open()
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
with open(path, 'a', buffering=1) as log:
    log.write(f'--- logger start {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n')
    while True:
        line = s.readline().decode(errors='replace')
        if line:
            log.write(f'{time.time():.3f} {line}')
