#pragma once
// Portable protocol core shared by the sketch and native synthetic regressions.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <errno.h>

namespace pointer_protocol {
constexpr unsigned kVersion = 2;
constexpr size_t kMaxEvents = 2048;
constexpr uint32_t kMaxTimeUs = 60000000;

enum class LineResult { pending, complete, invalid };
struct LineBuffer {
  char text[64];
  size_t length = 0;
  bool discard = false;
  LineResult push(unsigned char c) {
    if (c == '\n') {
      if (discard) { length = 0; discard = false; return LineResult::pending; }
      text[length] = '\0';
      bool empty = length == 0;
      length = 0;
      return empty ? LineResult::invalid : LineResult::complete;
    }
    if (discard) return LineResult::pending;
    if (c < 32 || c > 126 || length == sizeof(text) - 1) {
      discard = true;
      return LineResult::invalid;
    }
    text[length++] = static_cast<char>(c);
    return LineResult::pending;
  }
};

inline bool numbers(const char *text, int64_t *values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    while (*text == ' ') ++text;
    const char *digits = *text == '-' ? text + 1 : text;
    if (*digits < '0' || *digits > '9') return false;
    errno = 0;
    char *end;
    long long value = strtoll(text, &end, 10);
    if (errno == ERANGE || (*end != ' ' && *end != '\0')) return false;
    values[i] = value;
    text = end;
  }
  while (*text == ' ') ++text;
  return *text == '\0';
}

inline bool reportValues(int64_t dx, int64_t dy, int64_t buttons) {
  return dx >= -127 && dx <= 127 && dy >= -127 && dy <= 127 && (buttons == 0 || buttons == 1);
}

struct Event { uint32_t t_us; int8_t dx, dy; uint8_t buttons; };
enum class Tick { idle, waiting, sent, done, aborted };

struct Schedule {
  Event events[kMaxEvents];
  uint32_t attemptedAt[kMaxEvents];
  size_t count = 0, next = 0;
  uint32_t start = 0;
  bool playing = false, needsNeutral = true;

  bool add(const int64_t *v) {
    if (playing || count >= kMaxEvents || v[0] < 0 || v[0] > kMaxTimeUs ||
        !reportValues(v[1], v[2], v[3]) || (count && v[0] <= events[count - 1].t_us)) return false;
    events[count++] = {static_cast<uint32_t>(v[0]), static_cast<int8_t>(v[1]),
                      static_cast<int8_t>(v[2]), static_cast<uint8_t>(v[3])};
    return true;
  }
  void neutral(bool attempted) { needsNeutral = !attempted; }
  void abort() { playing = false; needsNeutral = true; }
  bool begin(uint32_t now, bool ready) {
    if (playing || needsNeutral || !ready || !count || events[count - 1].buttons != 0) return false;
    start = now; next = 0; playing = true;
    return true;
  }
  template <class Notify>
  Tick tick(uint32_t now, bool ready, bool fault, Notify notify) {
    if (fault || !ready) {
      bool active = playing;
      abort();
      return active ? Tick::aborted : Tick::idle;
    }
    if (!playing) return Tick::idle;
    uint32_t elapsed = now - start; // unsigned subtraction also handles micros() wrap
    if (elapsed > kMaxTimeUs) { abort(); return Tick::aborted; }
    if (elapsed < events[next].t_us) return Tick::waiting;
    if (!notify(events[next])) { abort(); return Tick::aborted; }
    attemptedAt[next++] = elapsed;
    if (next == count) { playing = false; return Tick::done; }
    return Tick::sent;
  }
};
} // namespace pointer_protocol
