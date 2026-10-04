# Bounded XR2 spin/pointer coexistence evidence

Three diagnostic runs: pointer only, WDA spin hold only, and the identical pointer scoop during a four-second WDA hold. Every recording was shorter than 10 seconds. Park provenance is an unverified indoor gameplay scene. No training admission, production collection or XR1 action occurred.

- `03-combined/coexistence.png` and `coexistence-frame342.png`: separate held spin-button glow and moving pointer trail in one source frame.
- Per-condition `inspection-summary.json`: exact source counts, PTS statistics and MCU scheduling observations.
- `01-pointer/decoder-diagnosis.json` and `source-frame-validation.json`: rig OpenCV AVFoundation's 447 count versus 446 exact source frames. The first runner stopped; its inspection retains `completed: false`. Independent FFmpeg/probe validation subsequently established the movie's source-frame count. This diagnostic revalidation is not corpus admission.
- `final-state-summary.json`: released button, healthy WDA/foreground, idle recorder, running tunnel and zero attachments. Full original guard output remains in the local artifact; device identifiers and network addresses are omitted here.
- `flick-a.json`: the identical corrected demo scoop.
- `run_spin_probe.py`, `run_spin_probe_v2.py`, `inspect_probe.py`: historical experiment harnesses. Copy to isolated `tmp/` and review/adapt paths before any use; these are not current operating instructions or authorization to run a test. The revised scratch harness validates exact FFmpeg decoder/probe equality, with all original caps retained.

External raw records and recordings are not backed up by Git:

- Local: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/spin/`, including each condition's `run.json`, `ffprobe.json`, `.mov`, original decoder failure and full final state.
- Rig: `/Users/training-server/trueskate-ai-runtime/tmp/hid-pointer/agent-spin-codex-20261004/`.

The observed coexistence concerns XCTest and AssistiveTouch on XR2 iOS 18.7.10. A physical finger or pad, intended spin-trick fidelity and end-to-end latency remain untested. MCU lateness of 0–1 microseconds does not establish iOS touch latency; source timing has approximately 16.7 ms sampling resolution.
