// Bluetooth HID mouse that plays a timed event schedule from its own clock.
//
// iOS AssistiveTouch turns a paired pointer into one touch: button down = touch
// down, movement while down = drag. Timing lives on this board so the host link's
// latency only shifts the start of a schedule, never the gaps inside it.
//
// Serial protocol (115200 baud, one command per line):
//   STATUS                 -> "STATUS connected=<0|1> auth=<-1|0|1> subscribed=<0|1>
//                             interval_us=<connection interval> events=<n>"
//   CLEAR                  -> "OK"
//   E <t_us> <dx> <dy> <b> -> append event: at t_us after GO send a report with
//                             relative move (dx, dy in -127..127) and button state b (0/1)
//   GO                     -> play; then "SENT <i> <actual_us>" per event and "DONE <n>"
//   NOW <dx> <dy> <b>      -> send one report immediately (calibration, homing)
//   PARAMS <min> <max> <latency> <timeout>
//                          -> ask the phone for new connection parameters (interval in
//                             1.25 ms units, timeout in 10 ms units); the result is logged
//                             as "CONN update status=..." when the phone answers
//
// Reports leave on the next connection event, so the phone sees them quantised to
// the connection interval even though this clock is exact. Every interval change
// is logged as "CONN ...".

#include <BLEDevice.h>
#include <BLEHIDDevice.h>
#include <BLE2902.h>
#include <BLESecurity.h>
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
static volatile int authStatus = -1;  // -1 none yet, 1 bonded, 0 failed
static volatile uint16_t connInterval = 0;  // 1.25 ms units, 0 when unknown
static esp_bd_addr_t peer = {0};

// iOS reads HID reports only over an encrypted, bonded link (the report
// characteristic is READ_ENCRYPTED), so pairing must actually complete.
class SecurityCallbacks : public BLESecurityCallbacks {
  uint32_t onPassKeyRequest() override { return 0; }
  void onPassKeyNotify(uint32_t) override {}
  bool onSecurityRequest() override { return true; }
  bool onConfirmPIN(uint32_t) override { return true; }
  void onAuthenticationComplete(esp_ble_auth_cmpl_t desc) override {
    authStatus = desc.success ? 1 : 0;
    Serial.printf("AUTH success=%d reason=%d\n", desc.success ? 1 : 0, desc.fail_reason);
  }
};

static bool subscribed() {
  BLEDescriptor *cccd = input ? input->getDescriptorByUUID(BLEUUID(static_cast<uint16_t>(0x2902))) : nullptr;
  return cccd != nullptr && static_cast<BLE2902 *>(cccd)->getNotifications();
}

// Handshake log: every GATT access iOS makes, so a stalled enumeration shows
// the last step it completed.
class LogCharacteristic : public BLECharacteristicCallbacks {
 public:
  explicit LogCharacteristic(const char *name) : name_(name) {}
  void onRead(BLECharacteristic *) override { Serial.printf("GATT %lu read %s\n", millis(), name_); }
  void onWrite(BLECharacteristic *c) override {
    Serial.printf("GATT %lu write %s len=%u\n", millis(), name_, static_cast<unsigned>(c->getLength()));
  }
 private:
  const char *name_;
};

class LogDescriptor : public BLEDescriptorCallbacks {
 public:
  explicit LogDescriptor(const char *name) : name_(name) {}
  void onRead(BLEDescriptor *) override { Serial.printf("GATT %lu read %s\n", millis(), name_); }
  void onWrite(BLEDescriptor *d) override {
    uint8_t *v = d->getValue();
    Serial.printf("GATT %lu write %s len=%u first=%u\n", millis(), name_, static_cast<unsigned>(d->getLength()),
                  d->getLength() ? v[0] : 0);
  }
 private:
  const char *name_;
};

static void logCharacteristic(BLEService *service, uint16_t uuid, const char *name) {
  BLECharacteristic *c = service ? service->getCharacteristic(BLEUUID(uuid)) : nullptr;
  if (c) c->setCallbacks(new LogCharacteristic(name));
  else Serial.printf("GATT missing %s\n", name);
}

static void logDescriptor(BLECharacteristic *c, uint16_t uuid, const char *name) {
  BLEDescriptor *d = c ? c->getDescriptorByUUID(BLEUUID(uuid)) : nullptr;
  if (d) d->setCallbacks(new LogDescriptor(name));
  else Serial.printf("GATT missing %s\n", name);
}

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *, esp_ble_gatts_cb_param_t *param) override {
    connected = true;
    memcpy(peer, param->connect.remote_bda, sizeof(peer));
    connInterval = param->connect.conn_params.interval;
    Serial.printf("GATT %lu connect conn_id=%u\n", millis(), param->connect.conn_id);
    Serial.printf("CONN %lu connect interval_us=%u latency=%u timeout_ms=%u\n", millis(),
                  param->connect.conn_params.interval * 1250u, param->connect.conn_params.latency,
                  param->connect.conn_params.timeout * 10u);
  }
  void onDisconnect(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
    connected = false;
    Serial.printf("GATT %lu disconnect reason=0x%02x\n", millis(), param->disconnect.reason);
    server->getAdvertising()->start();  // stay pairable after a drop
  }
  void onMtuChanged(BLEServer *, esp_ble_gatts_cb_param_t *param) override {
    Serial.printf("GATT %lu mtu=%u\n", millis(), param->mtu.mtu);
  }
};

static void sendReport(int8_t dx, int8_t dy, uint8_t buttons) {
  if (!connected) return;
  uint8_t report[3] = {static_cast<uint8_t>(buttons & 0x07), static_cast<uint8_t>(dx), static_cast<uint8_t>(dy)};
  input->setValue(report, sizeof(report));
  input->notify();
}

// Connection parameter updates arrive as GAP events, whichever side asked for them.
static void onGapEvent(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
  if (event != ESP_GAP_BLE_UPDATE_CONN_PARAMS_EVT) return;
  const auto &u = param->update_conn_params;
  if (u.status == ESP_BT_STATUS_SUCCESS) connInterval = u.conn_int;
  Serial.printf("CONN %lu update status=%d interval_us=%u latency=%u timeout_ms=%u\n", millis(), u.status,
                u.conn_int * 1250u, u.latency, u.timeout * 10u);
}

static int8_t clampByte(long v) { return static_cast<int8_t>(v < -127 ? -127 : (v > 127 ? 127 : v)); }

static void handleLine(char *line) {
  if (strcmp(line, "STATUS") == 0) {
    Serial.printf("STATUS connected=%d auth=%d subscribed=%d interval_us=%u events=%u\n", connected ? 1 : 0,
                  authStatus, subscribed() ? 1 : 0, connInterval * 1250u, static_cast<unsigned>(eventCount));
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
  } else if (strncmp(line, "PARAMS ", 7) == 0) {
    unsigned minInt, maxInt, latency, timeout;
    if (sscanf(line + 7, "%u %u %u %u", &minInt, &maxInt, &latency, &timeout) != 4 || !connected) {
      Serial.println("ERR params");
      return;
    }
    esp_ble_conn_update_params_t params = {};
    memcpy(params.bda, peer, sizeof(peer));
    params.min_int = minInt;
    params.max_int = maxInt;
    params.latency = latency;
    params.timeout = timeout;
    Serial.println(esp_ble_gap_update_conn_params(&params) == ESP_OK ? "OK" : "ERR params");
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
  BLEDevice::setSecurityCallbacks(new SecurityCallbacks());
  BLEDevice::setCustomGapHandler(onGapEvent);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  hid = new BLEHIDDevice(server);
  input = hid->inputReport(1);
  hid->manufacturer()->setValue("TrueSkate-AI");
  hid->pnp(0x02, 0xe502, 0xa111, 0x0210);
  hid->hidInfo(0x00, 0x02);
  // Use the (bonding, mitm, sc) overload: on esp32 core 3.3.x the uint8_t overload
  // stores the flags but leaves security disabled, so the board never requests
  // encryption on connect and iOS stalls the HID device on an unencrypted link.
  BLESecurity::setAuthenticationMode(true, false, true);
  BLESecurity::setForceAuthentication(true);
  BLESecurity::setCapability(ESP_IO_CAP_NONE);  // "Just Works" pairing, no PIN
  BLESecurity::setInitEncryptionKey(ESP_BLE_ENC_KEY_MASK | ESP_BLE_ID_KEY_MASK);
  BLESecurity::setRespEncryptionKey(ESP_BLE_ENC_KEY_MASK | ESP_BLE_ID_KEY_MASK);
  hid->reportMap(const_cast<uint8_t *>(kReportMap), sizeof(kReportMap));
  hid->startServices();
  logCharacteristic(hid->hidService(), 0x2a4a, "hid_info");
  logCharacteristic(hid->hidService(), 0x2a4b, "report_map");
  logCharacteristic(hid->hidService(), 0x2a4c, "control_point");
  logCharacteristic(hid->hidService(), 0x2a4e, "protocol_mode");
  logCharacteristic(hid->deviceInfo(), 0x2a50, "pnp_id");
  logCharacteristic(hid->deviceInfo(), 0x2a29, "manufacturer");
  logCharacteristic(hid->batteryService(), 0x2a19, "battery_level");
  input->setCallbacks(new LogCharacteristic("input_report"));
  logDescriptor(input, 0x2902, "input_cccd");
  logDescriptor(input, 0x2908, "input_report_ref");
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
