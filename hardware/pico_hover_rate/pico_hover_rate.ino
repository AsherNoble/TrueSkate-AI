// USB pointer rate/jitter probe: plays schedule.h once per flash, then records timing.
// RP2040 Pico, Arduino-Pico, Adafruit TinyUSB. Mouse only: no CDC, keyboard or storage.
#include <Adafruit_TinyUSB.h>
#include <EEPROM.h>
#include "schedule.h"

// Same boot-compatible three-button relative mouse as pico_mouse_smoke, 1 ms interval.
static const uint8_t reportDescriptor[] = {
  0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x09, 0x01, 0xa1, 0x00,
  0x05, 0x09, 0x19, 0x01, 0x29, 0x03, 0x15, 0x00, 0x25, 0x01,
  0x95, 0x03, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01, 0x75, 0x05,
  0x81, 0x03, 0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x15, 0x81,
  0x25, 0x7f, 0x75, 0x08, 0x95, 0x02, 0x81, 0x06, 0xc0, 0xc0
};
static Adafruit_USBD_HID mouse(reportDescriptor, sizeof(reportDescriptor),
                               HID_ITF_PROTOCOL_MOUSE, 1, false);

// Stats sector (EEPROM emulation, last flash sector), read back with picotool in BOOTSEL.
enum Status : uint16_t { kArmed = 1, kDone = 2, kNoMount = 3, kHostLost = 4, kSendTimeout = 5 };
struct __attribute__((packed)) Header {
  char magic[4];          // "PHR1"
  uint32_t buildId;       // this compile; a matching latch means the run already happened
  uint32_t scheduleHash;
  uint16_t status;
  uint16_t eventCount;
  uint16_t eventsSent;
  uint16_t clipped;       // events later than the field can hold (~1 s)
  uint32_t mountMs;       // since boot
  uint32_t notReadyWaits; // sends that had to wait for the endpoint
  uint32_t maxWaitUs;
};
static_assert(sizeof(Header) == 32, "header layout");
static_assert(sizeof(Header) + 2u * kEventCount <= 4096u, "stats must fit one sector");
static constexpr uint32_t kMountTimeoutMs = 60000;
static constexpr uint32_t kSendTimeoutUs = 1000000;   // a host that stops polling this long is gone

static constexpr uint32_t fnv1a(const char *s, uint32_t h = 2166136261u) {
  return *s ? fnv1a(s + 1, (h ^ static_cast<uint8_t>(*s)) * 16777619u) : h;
}
static constexpr uint32_t kBuildId = fnv1a(__DATE__ " " __TIME__) ^ kScheduleHash;

enum class State { Locked, Waiting, Lead, Done, Failed };
static State state = State::Waiting;
static Header header;
static uint32_t bootMs, mountMs;

static void saveStats() {
  EEPROM.put(0, header);
  EEPROM.commit();
}

// Lateness per event in 16 us units (up to ~1.05 s); 0xFFFF marks an unsent event.
static constexpr uint32_t kLateUnitUs = 16;
static constexpr uint16_t kUnsent = 0xFFFF;
static void storeSlot(uint16_t index, uint16_t value) { EEPROM.put(sizeof(Header) + 2u * index, value); }
static void recordLateness(uint16_t index, uint32_t lateUs) {
  uint32_t units = (lateUs + kLateUnitUs / 2) / kLateUnitUs;
  if (units >= kUnsent) {
    units = kUnsent - 1;
    ++header.clipped;
  }
  storeSlot(index, static_cast<uint16_t>(units));
}

static bool hostPresent() { return TinyUSBDevice.mounted() && !TinyUSBDevice.suspended(); }

// Same GET_REPORT answer as pico_mouse_smoke, the build iOS accepted: a neutral report.
static uint16_t getReport(uint8_t reportId, hid_report_type_t reportType,
                          uint8_t *buffer, uint16_t requestedLength) {
  if (reportId != 0 || reportType != HID_REPORT_TYPE_INPUT || requestedLength < 3) return 0;
  buffer[0] = buffer[1] = buffer[2] = 0;
  return 3;
}

// Best effort to never leave a press held: wait (bounded) for the endpoint, then lift.
static void releaseButtons() {
  const uint8_t neutral[] = {0, 0, 0};
  const uint32_t start = micros();
  while (hostPresent() && micros() - start < 200000u) {
    if (mouse.ready() && mouse.sendReport(0, neutral, sizeof(neutral))) return;
  }
}

// Plays every event at start + t_us, busy-waiting on micros(). A late event is sent as soon
// as the endpoint frees, so after a stall the overdue events go out back to back at the host's
// poll rate. Every event's lateness is recorded; analysis drops passes sent more than 1 ms late.
static Status play() {
  const uint32_t start = micros();
  for (uint16_t i = 0; i < kEventCount; ++i) {
    const Event &e = kEvents[i];
    const uint32_t due = start + e.t_us;
    while (static_cast<int32_t>(micros() - due) < 0) {
      if (!hostPresent()) return kHostLost;
    }
    const uint32_t waitStart = micros();
    const uint8_t report[] = {e.buttons, static_cast<uint8_t>(e.dx), static_cast<uint8_t>(e.dy)};
    bool waited = false;
    while (!(mouse.ready() && mouse.sendReport(0, report, sizeof(report)))) {
      waited = true;
      if (!hostPresent()) return kHostLost;
      if (micros() - waitStart > kSendTimeoutUs) return kSendTimeout;
    }
    const uint32_t sent = micros();
    if (waited) {
      ++header.notReadyWaits;
      if (sent - waitStart > header.maxWaitUs) header.maxWaitUs = sent - waitStart;
    }
    recordLateness(i, sent - due);
    header.eventsSent = i + 1;
  }
  return kDone;
}

void setup() {
  // Drop the core's default CDC device first: nothing enumerates while the latch is
  // written (flash writes stall interrupts), and a locked board never re-attaches.
  TinyUSBDevice.detach();
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  EEPROM.begin(4096);
  EEPROM.get(0, header);
  if (memcmp(header.magic, "PHR1", 4) == 0 && header.buildId == kBuildId) {
    state = State::Locked;  // This build already ran: never attach, never move or click.
    return;
  }
  // Latch before USB exists, so a crash or replug can never replay the schedule.
  memset(&header, 0, sizeof(header));
  memcpy(header.magic, "PHR1", 4);
  header.buildId = kBuildId;
  header.scheduleHash = kScheduleHash;
  header.status = kArmed;
  header.eventCount = kEventCount;
  for (uint16_t i = 0; i < kEventCount; ++i) storeSlot(i, kUnsent);
  saveStats();

  TinyUSBDevice.clearConfiguration();
  TinyUSBDevice.setManufacturerDescriptor("TrueSkate-AI");
  TinyUSBDevice.setProductDescriptor("TrueSkate Pico Rate Probe");
  TinyUSBDevice.setConfigurationAttribute(0x80);  // Bus powered, no wakeup.
  TinyUSBDevice.setConfigurationMaxPower(100);
  mouse.setReportCallback(getReport, nullptr);
  if (!mouse.begin()) {
    state = State::Failed;
    return;
  }
  TinyUSBDevice.attach();
  bootMs = millis();
}

void loop() {
  const uint32_t now = millis();
  if (state == State::Waiting) {
    if (TinyUSBDevice.mounted() && mouse.ready()) {
      mountMs = now;
      header.mountMs = now;
      state = State::Lead;
    } else if (now - bootMs >= kMountTimeoutMs) {
      header.status = kNoMount;
      saveStats();
      state = State::Failed;
    }
  } else if (state == State::Lead) {
    if (!hostPresent()) {
      header.status = kHostLost;
      saveStats();
      state = State::Failed;
    } else if (now - mountMs >= kLeadMs) {
      digitalWrite(LED_BUILTIN, HIGH);  // Solid while playing.
      header.status = play();
      if (header.status != kDone) releaseButtons();
      saveStats();
      state = header.status == kDone ? State::Done : State::Failed;
    }
  }

  // Waiting/lead: 1 s blink. Done: 100 ms flash every 2 s. Failed: double flash. Locked: 10 Hz.
  bool led;
  if (state == State::Done) led = now % 2000 < 100;
  else if (state == State::Failed) led = (now % 1600 < 100) || (now % 1600 >= 250 && now % 1600 < 350);
  else if (state == State::Locked) led = now % 100 < 50;
  else led = now % 1000 < 500;
  digitalWrite(LED_BUILTIN, led ? HIGH : LOW);
  delay(1);
}
