// Adapter diagnostic only: one bounded hover test per power cycle, no clicks.
// RP2040 Pico, Arduino-Pico, Adafruit TinyUSB stack. No ESP32/UART needed.
#include <Adafruit_TinyUSB.h>

// Standard boot-compatible three-button relative mouse, no report ID or wheel.
static const uint8_t reportDescriptor[] = {
  0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x09, 0x01, 0xa1, 0x00,
  0x05, 0x09, 0x19, 0x01, 0x29, 0x03, 0x15, 0x00, 0x25, 0x01,
  0x95, 0x03, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01, 0x75, 0x05,
  0x81, 0x03, 0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x15, 0x81,
  0x25, 0x7f, 0x75, 0x08, 0x95, 0x02, 0x81, 0x06, 0xc0, 0xc0
};
static Adafruit_USBD_HID mouse(reportDescriptor, sizeof(reportDescriptor),
                               HID_ITF_PROTOCOL_MOUSE, 1, false);

enum class State { Waiting, Lead, Moving, Neutral, Done, Failed };
static State state = State::Waiting;
static uint32_t bootMs, mountMs, nextMs, deadlineMs;
static uint8_t moves = 0;
static const int8_t dx[] = {8, 0, -8, 0};
static const int8_t dy[] = {0, 8, 0, -8};
static constexpr uint8_t kMoves = 80;  // Two squares, 10 reports per side.
static constexpr uint32_t kSpacingMs = 40;
static constexpr uint32_t kLeadMs = 8000;
static constexpr uint32_t kRunBudgetMs = 5000;

static uint16_t getReport(uint8_t reportId, hid_report_type_t reportType,
                          uint8_t *buffer, uint16_t requestedLength) {
  if (reportId != 0 || reportType != HID_REPORT_TYPE_INPUT || requestedLength < 3) return 0;
  buffer[0] = buffer[1] = buffer[2] = 0;
  return 3;
}

static bool hover(int8_t x, int8_t y) {
  // Button byte is always zero: the diagnostic cannot press or click.
  const uint8_t report[] = {0, static_cast<uint8_t>(x), static_cast<uint8_t>(y)};
  return mouse.sendReport(0, report, sizeof(report));
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  // Arduino's TinyUSB initialization adds CDC. Rebuild as mouse-only, then
  // re-enumerate; leave no serial, keyboard, storage or other interface.
  TinyUSBDevice.detach();
  TinyUSBDevice.clearConfiguration();
  TinyUSBDevice.setManufacturerDescriptor("TrueSkate-AI");
  TinyUSBDevice.setProductDescriptor("TrueSkate Pico Mouse Test");
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
      state = State::Lead;
    } else if (now - bootMs >= 30000) {
      state = State::Failed;
    }
  } else if (state == State::Lead || state == State::Moving || state == State::Neutral) {
    // A disconnected/suspended host terminates the test; no replay on reconnect.
    if (!TinyUSBDevice.mounted() || TinyUSBDevice.suspended()) {
      state = State::Failed;
    } else if (state == State::Lead && now - mountMs >= kLeadMs) {
      deadlineMs = now + kRunBudgetMs;
      nextMs = now;
      state = State::Moving;
    } else if (state == State::Moving || state == State::Neutral) {
      if (static_cast<int32_t>(now - deadlineMs) >= 0) {
        state = State::Failed;
      } else if (static_cast<int32_t>(now - nextMs) >= 0 && mouse.ready()) {
        if (state == State::Neutral) {
          if (hover(0, 0)) state = State::Done;
        } else if (hover(dx[(moves / 10) % 4], dy[(moves / 10) % 4])) {
          ++moves;
          nextMs = now + kSpacingMs;  // Never burst to catch up after a stall.
          if (moves == kMoves) state = State::Neutral;
        }
      }
    }
  }

  // Waiting: slow blink. Moving: fast blink. Done: solid. Failed: double blink.
  bool led;
  if (state == State::Done) led = true;
  else if (state == State::Failed) led = (now % 1600 < 100) || (now % 1600 >= 250 && now % 1600 < 350);
  else if (state == State::Moving || state == State::Neutral) led = now % 100 < 50;
  else led = now % 1000 < 500;
  digitalWrite(LED_BUILTIN, led ? HIGH : LOW);
  delay(1);
}
