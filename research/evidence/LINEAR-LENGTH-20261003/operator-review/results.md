# Blinded review results

All 135 labels validated against bundle `dc954a013a05531482095bdff5f1e6b5e232051d41d5342b900cfecc62ff1e1e`.

| Duration (ms) | Trace | Hold | Flicker | Board moved |
|---:|---:|---:|---:|---:|
| 50 | 15/15 | 0/15 | 0/15 | 15/15 |
| 45 | 15/15 | 0/15 | 0/15 | 15/15 |
| 40 | 15/15 | 0/15 | 0/15 | 15/15 |
| 35 | 15/15 | 0/15 | 0/15 | 15/15 |
| 30 | 15/15 | 0/15 | 0/15 | 15/15 |
| 25 | 12/15 | 0/15 | 3/15 | 15/15 |
| 20 | 7/15 | 0/15 | 8/15 | 15/15 |
| 15 | 7/15 | 0/15 | 8/15 | 15/15 |
| 10 | 0/15 | 0/15 | 15/15 | 0/15 |

## Trace counts by gesture length

Lengths are fractions of the original diagonal. Each cell is trace labels out of three. Board movement totals are reported in the duration table above.

| Duration (ms) | 20% | 40% | 60% | 80% | 100% |
|---:|---:|---:|---:|---:|---:|
| 50 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 45 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 40 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 35 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 30 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 25 | 3/3 | 2/3 | 1/3 | 3/3 | 3/3 |
| 20 | 2/3 | 2/3 | 1/3 | 2/3 | 0/3 |
| 15 | 0/3 | 1/3 | 2/3 | 2/3 | 2/3 |
| 10 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

Adjacent tested durations separating no board movement from movement in every clip: [[10, 15]]. All-trace durations: [30, 35, 40, 45, 50] ms; mixed trace durations: [15, 20, 25] ms. Of 34 flicker labels, 19 still have board movement. Hold labels: 0.

Three repeats per cell describe this session; they do not establish an exact universal threshold or input path collapse. Trace and board defaults were used by the UI, so these are operator-saved assessments, not independent instrument readings.

## Comments

- Clip 84: 45 ms, 20% length — Very short trace. Definitely not a flicker but remarkably short.
- Clip 126: 45 ms, 20% length — Another very short trace (there have been about 5 – might just be very short gesture length – just noting it).
