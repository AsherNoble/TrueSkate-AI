// Bluetooth HID mouse that plays a timed event schedule from its own clock.
//
// iOS AssistiveTouch turns a paired pointer into one touch: button down = touch
// down, movement while down = drag. Timing lives on this board so the host link's
// latency only shifts the start of a schedule, never the gaps inside it.
//
// Serial protocol (115200 baud, one command per line):
//   HELLO 2                -> "HELLO 2"; older hosts/firmware are rejected
//   CANCEL                 -> "ABORT <attempt-count> <reason>"; cancels playback
//   NEUTRAL                -> attempt released buttons; required before another run
//   STATUS                 -> "STATUS connected=<0|1> auth=<-1|0|1> subscribed=<0|1>
//                             interval_us=<connection interval> events=<n>"
//   CLEAR                  -> "OK"
//   E <t_us> <dx> <dy> <b> -> append event: at t_us after GO send a report with
//                             relative move (dx, dy in -127..127) and button state b (0/1)
//   GO                     -> cancellable play; SENT records are notification attempts
//                             ordered "SENT <i> <actual_us>" then "DONE <n>" or ABORT
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
#include "schedule_protocol.h"

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

using namespace pointer_protocol;
static Schedule schedule;
static uint8_t manualButtons = 0;
static volatile bool linkFault = false;
static volatile bool protocolReady = false;

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
    if (!desc.success) linkFault = true;
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
    if (strcmp(name_, "input_cccd") == 0 && (!d->getLength() || !(v[0] & 1))) linkFault = true;
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
    authStatus = -1;
    memcpy(peer, param->connect.remote_bda, sizeof(peer));
    connInterval = param->connect.conn_params.interval;
    Serial.printf("GATT %lu connect conn_id=%u\n", millis(), param->connect.conn_id);
    Serial.printf("CONN %lu connect interval_us=%u latency=%u timeout_ms=%u\n", millis(),
                  param->connect.conn_params.interval * 1250u, param->connect.conn_params.latency,
                  param->connect.conn_params.timeout * 10u);
  }
  void onDisconnect(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
    connected = false;
    authStatus = -1;
    linkFault = true;
    protocolReady = false;
    Serial.printf("GATT %lu disconnect reason=0x%02x\n", millis(), param->disconnect.reason);
    server->getAdvertising()->start();  // stay pairable after a drop
  }
  void onMtuChanged(BLEServer *, esp_ble_gatts_cb_param_t *param) override {
    Serial.printf("GATT %lu mtu=%u\n", millis(), param->mtu.mtu);
  }
};

static bool ready() { return connected && authStatus == 1 && subscribed(); }

// An attempt to notify is not proof that iOS received or acted on a report.
static bool sendReport(int8_t dx, int8_t dy, uint8_t buttons) {
  if (!ready() || linkFault) return false;
  uint8_t report[3] = {buttons, static_cast<uint8_t>(dx), static_cast<uint8_t>(dy)};
  input->setValue(report, sizeof(report));
  input->notify();
  return ready() && !linkFault;
}

static void abortPlayback(const char *reason) {
  schedule.abort();
  sendReport(0, 0, 0); // best effort only; NEUTRAL remains mandatory after recovery
  Serial.printf("ABORT %u %s\n", static_cast<unsigned>(schedule.next), reason);
}

// Connection parameter updates arrive as GAP events, whichever side asked for them.
static void onGapEvent(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
  if (event != ESP_GAP_BLE_UPDATE_CONN_PARAMS_EVT) return;
  const auto &u = param->update_conn_params;
  if (u.status == ESP_BT_STATUS_SUCCESS) connInterval = u.conn_int;
  Serial.printf("CONN %lu update status=%d interval_us=%u latency=%u timeout_ms=%u\n", millis(), u.status,
                u.conn_int * 1250u, u.latency, u.timeout * 10u);
}

static void handleLine(char *line) {
  if (strcmp(line, "HELLO 2") == 0) {
    if (schedule.playing) { Serial.println("ERR busy"); return; }
    protocolReady = true;
    Serial.printf("HELLO %u\n", kVersion);
    return;
  }
  if (strcmp(line, "CANCEL") == 0) { abortPlayback("cancel"); return; }
  if (!protocolReady) { Serial.println("ERR protocol requires HELLO 2"); return; }
  if (strcmp(line, "STATUS") == 0) {
    Serial.printf("STATUS protocol=%u connected=%d auth=%d subscribed=%d interval_us=%u events=%u neutral_required=%d\n",
                  kVersion, connected ? 1 : 0, authStatus, subscribed() ? 1 : 0,
                  connInterval * 1250u, static_cast<unsigned>(schedule.count), schedule.needsNeutral ? 1 : 0);
    return;
  }
  if (schedule.playing) { Serial.println("ERR busy"); return; }
  if (strcmp(line, "NEUTRAL") == 0) {
    schedule.neutral(sendReport(0, 0, 0));
    if (!schedule.needsNeutral) manualButtons = 0;
    Serial.println(schedule.needsNeutral ? "ERR neutral unavailable" : "OK");
  } else if (strcmp(line, "CLEAR") == 0) {
    schedule.count = schedule.next = 0;
    Serial.println("OK");
  } else if (strncmp(line, "E ", 2) == 0) {
    int64_t values[4];
    Serial.println(numbers(line + 2, values, 4) && schedule.add(values) ? "OK" : "ERR event");
  } else if (strncmp(line, "NOW ", 4) == 0) {
    int64_t values[3];
    if (!numbers(line + 4, values, 3) || !reportValues(values[0], values[1], values[2]) ||
        schedule.needsNeutral || !sendReport(values[0], values[1], values[2])) {
      Serial.println("ERR now"); return;
    }
    manualButtons = values[2];
    Serial.println("OK");
  } else if (strncmp(line, "PARAMS ", 7) == 0) {
    int64_t v[4];
    if (!numbers(line + 7, v, 4) || !connected || v[0] < 6 || v[1] < v[0] || v[1] > 3200 ||
        v[2] < 0 || v[2] > 499 || v[3] < 10 || v[3] > 3200 || v[3] * 4 <= (1 + v[2]) * v[1]) {
      Serial.println("ERR params"); return;
    }
    esp_ble_conn_update_params_t params = {};
    memcpy(params.bda, peer, sizeof(peer));
    params.min_int = v[0]; params.max_int = v[1]; params.latency = v[2]; params.timeout = v[3];
    Serial.println(esp_ble_gap_update_conn_params(&params) == ESP_OK ? "OK" : "ERR params");
  } else if (strcmp(line, "GO") == 0) {
    if (linkFault || manualButtons != 0 || !schedule.begin(micros(), ready())) Serial.println("ERR unsafe schedule or link; NEUTRAL required");
  } else {
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
  static LineBuffer line;
  // Bounded serial work allows cancellation and link checks even under a flood.
  for (unsigned budget = 0; Serial.available() && budget < 1024; ++budget) {
    LineResult result = line.push(static_cast<unsigned char>(Serial.read()));
    if (result == LineResult::invalid) {
      abortPlayback("malformed-line");
      Serial.println("ERR malformed or overlong line");
    } else if (result == LineResult::complete) {
      handleLine(line.text);
    }
  }
  bool fault = linkFault;
  linkFault = false;
  Tick result = schedule.tick(micros(), ready(), fault, [](const Event &e) {
    return sendReport(e.dx, e.dy, e.buttons);
  });
  if (result == Tick::aborted) abortPlayback("link-or-deadline");
  if (result == Tick::done && (!ready() || linkFault)) { abortPlayback("link-lost-at-completion"); return; }
  if (result == Tick::done) {
    for (size_t i = 0; i < schedule.next; ++i)
      Serial.printf("SENT %u %lu\n", static_cast<unsigned>(i), static_cast<unsigned long>(schedule.attemptedAt[i]));
    Serial.printf("DONE %u\n", static_cast<unsigned>(schedule.next));
  }
}
