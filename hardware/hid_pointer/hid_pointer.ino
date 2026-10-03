// Bluetooth HID mouse that plays a timed event schedule from its own clock.
//
// iOS AssistiveTouch turns a paired pointer into one touch: button down = touch
// down, movement while down = drag. Timing lives on this board so the host link's
// latency only shifts the start of a schedule, never the gaps inside it.
//
// Serial protocol (115200 baud, one command per line):
//   STATUS                 -> "STATUS connected=<0|1> events=<n>"
//   CLEAR                  -> "OK"
//   E <t_us> <dx> <dy> <b> -> append event: at t_us after GO send a report with
//                             relative move (dx, dy in -127..127) and button state b (0/1)
//   GO                     -> play; then "SENT <i> <actual_us>" per event and "DONE <n>"
//   NOW <dx> <dy> <b>      -> send one report immediately (calibration, homing)
//
// Pilot only: Bluetooth reports leave on the connection interval (~7.5-15 ms), so
// the phone sees them with that much jitter even though this clock is exact.

#include <BLEDevice.h>
#include <BLEHIDDevice.h>
#include <BLEServer.h>
#include <HIDTypes.h>

static const uint8_t kReportMap[] = {
  USAGE_PAGE(1), 0x01, USAGE(1), 0x02,          // Generic Desktop, Mouse
  COLLECTION(1), 0x01,                          // Application
  REPORT_ID(1), 0x01,
  USAGE(1), 0x01, COLLECTION(1), 0x00,          // Pointer, Physical
  USAGE_PAGE(1), 0x09, USAGE_MINIMUM(1), 0x01, USAGE_MAXIMUM(1), 0x03,
  LOGICAL_MINIMUM(1), 0x00, LOGICAL_MAXIMUM(1), 0x01,
  REPORT_COUNT(1), 0x03, REPORT_SIZE(1), 0x01, HIDINPUT(1), 0x02,   // 3 buttons
  REPORT_COUNT(1), 0x01, REPORT_SIZE(1), 0x05, HIDINPUT(1), 0x03,   // padding
  USAGE_PAGE(1), 0x01, USAGE(1), 0x30, USAGE(1), 0x31,              // X, Y
  LOGICAL_MINIMUM(1), 0x81, LOGICAL_MAXIMUM(1), 0x7F,
  REPORT_SIZE(1), 0x08, REPORT_COUNT(1), 0x02, HIDINPUT(1), 0x06,   // relative
  END_COLLECTION(0), END_COLLECTION(0)
};

struct Event { uint32_t t_us; int8_t dx; int8_t dy; uint8_t buttons; };
static const size_t kMaxEvents = 2048;
static Event events[kMaxEvents];
static uint32_t sentAt[kMaxEvents];
static size_t eventCount = 0;

static BLEHIDDevice *hid = nullptr;
static BLECharacteristic *input = nullptr;
static volatile bool connected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override { connected = true; }
  void onDisconnect(BLEServer *server) override {
    connected = false;
    server->getAdvertising()->start();  // stay pairable after a drop
  }
};

static void sendReport(int8_t dx, int8_t dy, uint8_t buttons) {
  if (!connected) return;
  uint8_t report[3] = {static_cast<uint8_t>(buttons & 0x07), static_cast<uint8_t>(dx), static_cast<uint8_t>(dy)};
  input->setValue(report, sizeof(report));
  input->notify();
}

static int8_t clampByte(long v) { return static_cast<int8_t>(v < -127 ? -127 : (v > 127 ? 127 : v)); }

static void handleLine(char *line) {
  if (strcmp(line, "STATUS") == 0) {
    Serial.printf("STATUS connected=%d events=%u\n", connected ? 1 : 0, static_cast<unsigned>(eventCount));
  } else if (strcmp(line, "CLEAR") == 0) {
    eventCount = 0;
    Serial.println("OK");
  } else if (line[0] == 'E' && line[1] == ' ') {
    unsigned long t; long dx, dy, b;
    if (sscanf(line + 2, "%lu %ld %ld %ld", &t, &dx, &dy, &b) != 4 || eventCount >= kMaxEvents) {
      Serial.println("ERR event");
      return;
    }
    if (eventCount > 0 && t < events[eventCount - 1].t_us) {
      Serial.println("ERR order");
      return;
    }
    events[eventCount++] = {static_cast<uint32_t>(t), clampByte(dx), clampByte(dy), static_cast<uint8_t>(b ? 1 : 0)};
    Serial.println("OK");
  } else if (strncmp(line, "NOW ", 4) == 0) {
    long dx, dy, b;
    if (sscanf(line + 4, "%ld %ld %ld", &dx, &dy, &b) != 3) { Serial.println("ERR now"); return; }
    sendReport(clampByte(dx), clampByte(dy), b ? 1 : 0);
    Serial.println("OK");
  } else if (strcmp(line, "GO") == 0) {
    if (!connected) { Serial.println("ERR not connected"); return; }
    const uint32_t start = micros();
    for (size_t i = 0; i < eventCount; i++) {
      while (static_cast<uint32_t>(micros() - start) < events[i].t_us) { }
      sentAt[i] = micros() - start;
      sendReport(events[i].dx, events[i].dy, events[i].buttons);
    }
    for (size_t i = 0; i < eventCount; i++) Serial.printf("SENT %u %lu\n", static_cast<unsigned>(i), static_cast<unsigned long>(sentAt[i]));
    Serial.printf("DONE %u\n", static_cast<unsigned>(eventCount));
  } else if (line[0] != '\0') {
    Serial.println("ERR unknown");
  }
}

void setup() {
  Serial.begin(115200);
  BLEDevice::init("TrueSkate Pointer");
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  hid = new BLEHIDDevice(server);
  input = hid->inputReport(1);
  hid->manufacturer()->setValue("TrueSkate-AI");
  hid->pnp(0x02, 0xe502, 0xa111, 0x0210);
  hid->hidInfo(0x00, 0x02);
  BLESecurity *security = new BLESecurity();
  security->setAuthenticationMode(ESP_LE_AUTH_BOND);
  hid->reportMap(const_cast<uint8_t *>(kReportMap), sizeof(kReportMap));
  hid->startServices();
  BLEAdvertising *advertising = server->getAdvertising();
  advertising->setAppearance(HID_MOUSE);
  advertising->addServiceUUID(hid->hidService()->getUUID());
  advertising->start();
  Serial.println("READY");
}

void loop() {
  static char buffer[64];
  static size_t length = 0;
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') continue;
    if (c == '\n' || length == sizeof(buffer) - 1) {
      buffer[length] = '\0';
      handleLine(buffer);
      length = 0;
    } else {
      buffer[length++] = c;
    }
  }
}
