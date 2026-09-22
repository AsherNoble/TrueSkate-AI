# XCTEST-ATTACHMENT-20260922 — Stale XCTest recording recovery

Status: resolved on XR2; collection remained off.

## Incident

After the bounded whole-path smoke test, Appium's official `cleanup-videos`
command repeatedly reported that it deleted XR2 attachment
`221AA31E-EB0D-4DB9-BE8D-D33001095B5A`, but an immediate dry-run still listed
the same UUID. WDA's `/wda/video` endpoint returned `null`; later recorder starts
alternated between `Already recording` and `XCTDaemon.ScreenRecordingError
Code=7` (`Failed to write file`). The required root RemoteXPC tunnel was running,
no collector was active, and the phone had approximately 44 GB free.

## Identification and preservation

The UUID existed only in testmanagerd's `tmp/Attachments` app-data directory.
It was copied to durable rig storage before mutation:

`/Users/training-server/trueskate-ai-runtime/tmp/`
`recovered-xctest-attachments-20260922/`
`221AA31E-EB0D-4DB9-BE8D-D33001095B5A.mov`

The preserved file is 382,106,978 bytes, lasts 304.13 seconds, carries creation
time `2026-06-25T08:18:43Z`, and has SHA-256
`5a6ccf0b98b80a4ce0e35d801e046ece69c17684515e8ca93531b2b05742caae`.
It is an old five-minute recording from the known oversized-segment failure era,
not output from the September smoke test.

## Experiments

1. A direct call to appium-ios-remotexpc's `XCTestAttachment.delete` returned
   `null` without error, but the UUID remained. The installed Appium cleanup
   script does not relist after deletion, so its `Deleted 1 attachment` message
   was a false success.
2. CoreDevice empty-directory replacement removed the stale UUID and the entire
   `tmp/Attachments` directory. A recorder start then failed with Code 7.
3. Recreating the directory through `devicectl` with a placeholder made it
   listable. Restarting only the existing prebuilt WDA runner still produced
   Code 7. A failed start could also leave an XCTest-side recorder active while
   WDA continued to report `/wda/video = null`, causing the next start to say
   `Already recording`.
4. The placeholder directory was removed, XR2's WDA runner was stopped, and the
   phone's `testmanagerd` process was terminated through `devicectl`. After iOS
   relaunched testmanagerd, the same prebuilt WDA runner was started again.
5. Two consecutive five-second, gesture-free recordings then completed normally.
   They reported UUIDs `860EF5CD-E6B3-40ED-B947-1F32DE240ED7` and
   `3E0C6DA4-D69A-43C0-885E-C343EDC81A79`; the first retrieved file was
   2,706,175 bytes and 5.133333 seconds. After stopping, `/wda/video` returned
   `null` and the official cleanup dry-run found zero UUID-shaped attachments.

## Conclusion

The old attachment was a genuine orphan, and the installed cleanup command can
falsely claim it deleted such an orphan. Removing the directory is insufficient
while the existing testmanagerd process retains its state; manually recreating
the directory is also unsafe. The proven recovery is: preserve the file, remove
the stale directory, restart testmanagerd while it is absent, relaunch the same
prebuilt WDA, and validate two short start/stop cycles. The exact internal cause
of the stale daemon/filesystem state is not observable through supported APIs,
but the recovery boundary is experimentally isolated.

The repository cleanup wrapper now verifies deletion by relisting and refuses
to claim success when a UUID survives. The guarded deep-recovery sequence is in
`DEPLOYMENT.md`.
