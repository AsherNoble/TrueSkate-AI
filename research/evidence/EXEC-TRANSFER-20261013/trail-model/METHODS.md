# Trail-model evidence and limitations

This resumes Claude task `a96804ae920b3eb1e`, recovered from session `e0f1c48e-973a-4ff3-a26a-f5f2350d9a43`. Original instructions specified all 100 local round 6 clips, a measured trail model and visual validation. Original scripts and 42 sampled data files were retained. The remaining 58 profiles and all stationary-hold profiles were completed with the repository `.venv`; no source/rig changes were made.

## Inputs and runs

Scratch root: `/private/tmp/claude-503/-Users-ashernoble-Projects-Robotics---hardware-TrueSkate-AI/e0f1c48e-973a-4ff3-a26a-f5f2350d9a43/scratchpad/`.

- Inputs: `audit6-mirror/audit-6/manifest.json`,100 corresponding native 60 fps 828×1792 clips; `design/EXEC-TRANSFER-DESIGN.md`; read-only `lag/` source and output.
- Preserved sampler `sample_all.py` samples each command path every 4 native px, perpendicular offsets−80..80 by 4 px, with RGB background from offsets≥72 px. It stores native elapsed times, raw warm excess, background, noise and deck-texture mask.
- Completed sampling log: `data/resume-sampling.log`; preserved earlier log: `data/run.log`.
- `holds.py` samples polar radii 0..100 by 2 px,72 angles, subtracting an outer-annulus median. `data/holds.json` includes 13 true zero-length holds plus quasi-stationary clips. For true first-hold cap estimates,55 is the only long clean plateau;99 supplies an independent short clean hold. Other holds are deck, horizon, edge or overlap contaminated.
- `resume_summary.py` reads all 100 profiles, selects clean fixed spots, aligns each spot to its **observed local intensity peak**, and reports clip-balanced medians. It does not pretend the peak equals actual touch arrival. `data/resume-fade-spots.json` stores all 728 selected spots, from 23 clips.
- `resume_cap_audit.py` records two clean first holds, polar cap radii and manual midpoint classifications. Frames are repeated observations, not independent subjects. Directional/background asymmetry is larger than within-plateau variation.
- `resume_images.py` generates all 100 midpoint command-centred crops and six hold strips from source video. Each midpoint is selected mechanically as the frame nearest half the commanded duration. Cyan dots indicate command positions or held points, not detected fingertips.

## Selection and uncertainty

Fade selection requires speed 0.5–4 native px/ms, more than 70 px from a vertex/end, more than 160 px from distant path stretches, a full sampling corridor in bounds, pre-onset warm ridge<12, observed peak≥40 above baseline, peak ridge within 8 px of the commanded path, peak FWHM 20–65 px, noise≤8, no texture-mask hit during the following 340 ms, and a low final background. These filters deliberately choose measurable cases; they cannot estimate how often the trail is measurable. There are 728 correlated spatial spots;23 clip medians are equally weighted for headline statistics.

The half-peak and excess 10 fade times use linear interpolation between native frames. Subframe decimal output is numerical interpolation, not subframe validation. The small selection/post-background conditions can shorten estimated tails;250–300 ms retrace exclusion is a conservative operational proposal. Command-age expiry additionally includes per-clip command/visible-motion offset and cannot isolate rendering lifetime. `resume-command-age.json` records that distinction.

The cross-section uses a local perpendicular background, so broad illumination gradients, changing camera position, curved/repeated paths and texture can bias it. Peak-age widths at late times become unreliable when signal approaches background. The FWHM 200 ms number is descriptive, not a robust head-localization width. Search for the ridge is limited near the command path; this analysis does not establish independent spatial fidelity or a bound on executor deviations.

Cap radii are threshold dependent. In long held clip 55 the radial excess 25 radius is 27.53 px, forward-sector 30.69 px, and side-sector 27.58 px; short hold 99 gives 27.31/27.96/26.58 px. The faint halo radii are≈35–43 px. First-contact buildup changes the threshold edge from zero to≈28 px over several frames; stationary values should not simply be substituted for moving-cap geometry. The fresh moving-trail FWHM≈45 px is broader than the stationary cap's≈34–41 px diameter at half intensity.

## Visibility audit

All 100 midpoint crops were visually inspected. Nine centres intersect deck/silhouette:62,73,77,122,125,128,134,137,155. Twelve have definite hard scenery/HUD edges within about 30 native px of centre:52,54,56,57,58,63,64,65,72,78,85,119. Eight more are marked possible:60,89,95,111,120,136,140,157. Those are visual risk flags, not ground-truth labels of fully invisible trail heads. They describe these 100 fixed snapshots only; no population confidence interval or per-clip any-frame rate is claimed.

The inherited deck-texture detector flags 26 midpoint crops, including all 9 true deck intersections, and many rails/text/background edges. Its all-active-frame counts (1358/4272 frames across 71 clips) are **proxy flags**, not 1358 confirmed occlusions. Similarly 1433/4272 frames have a preexisting warm ridge and 1201/4272 have geometric-overlap risk; none is a certified failure rate. These raw proxies are in `resume-visibility-proxies.json` so they remain inspectable without being overstated.

The input premise that the trail is always drawn under the board is too strong. In `resume-hold-137.jpg`, a held glow visibly brightens over deck texture; board pixels may contain bloom/compositing rather than complete darkness. Geometry and intensity must be treated as contaminated there even when a glow remains visible.

## Why clip 67 does not establish a hold timing warp

The last motion is 528,850→358,1124 over 173 ms. The previous motion is 190,1398→528,850 over 161 ms. Their unit-direction dot product is−0.99996844285; the later endpoint is 2.56 px from the previous line and was crossed around 680 ms. The final 322 px retrace is therefore already luminous before it begins. The tracker uses 200 ms of **candidate-command-time** history: a false advanced candidate near 921 ms excludes the earlier 680 ms visit, although that visit is only≈30 ms old in observed frame 77. This makes overlap validation depend on the erroneous timing it should validate.

`lag/vlag-all.json` assigns frame 77 tau 921 ms (nominal 710.82 ms), frame 78 tau 927 ms, frame 79 tau 933 ms and frame 80 tau 930 ms, all evidence 1, visibility 1, ambiguousFalse and reliableTrue. The chosen edge can extend past the finite path; `past_end` counts that nonexistent future path as dark and fully visible. The score can thus favour the final endpoint without observing the moving fingertip. `full-67-77.jpg` shows the deck crossing this region throughout the final approach. Neither the plateau-time alias nor this false endpoint score is a physical executor measurement.

## Recommended follow-up, without device execution

1. Recompute a **sensitivity envelope**, keeping the baseline, for cap radii 24/28/32, fade/retrace windows 200/250/300 ms and endpoint-darkness disabled. Reject cases whose direction changes across reasonable settings.
2. Make ambiguity conservative over the candidate-time interval, or geometrically censor all retracing terminal legs before fitting lag. Build synthetic rendered examples that reproduce the 67 failure.
3. Use a hold-specific change detector for contact arrival/departure; report intervals during which stationary time is unobservable. Continuous persistence cannot distinguish a long real hold from scenery.
4. Obtain independent touch timestamps if one-frame **absolute** timing is required. No amount of re-analysis can separate app render latency, calibrated-marker bias and XCTest delivery from these videos alone.

Confidence is high for the observed age-dependent width/fade and the existence of the 67 measurement confound; moderate for using the two held caps to bracket moving-cap bias; low for actual event timing, rendering latency, small corner cutting or a precise physical occlusion frequency.
