# Rig reconciliation — 2026-09-08

- Original BC: `15f7873f797e1db625f90ffe2040168da036ad49`.
- Rig committed head: `463316d34b81129986a171920369dd6067e91f7b`.
- Preserved live source: `567d0929155494a4905e27f741b2972daf983ec7` (`archive/rig-working-20260908`).
- The live checkout was copied, not reset or switched. `.env`, credentials, datasets and runtime logs were excluded.
- Snapshot launchd files were read in full; copied source was scanned for credential literals and URLs before publication. URLs were local service endpoints; no credential literals found.
- Snapshot includes historical .orig/.bak/.rig_wip source for recovery only; those files are not active entrypoints.

## Commit disposition

Patch equality ignores commit IDs. `git cherry` identifies exact equivalent patches; the remaining commits were reviewed against later BC implementations. Merge commits carry ancestry, not another feature to replay.

| Rig commit | Disposition | BC reference |
|---|---|---|
| `cb25e8b34faeefe1ce4dfa03546cb6347f7ca71f` | Dashboard fallback already incorporated; current preview source also matches live rig. | `5552098300a573fb3d88871b8ac0c32dd67b3db5` |
| `62544e9757eb56b43c7752a7e0a5e2f0c9eba2c3` | Staleness badge incorporated; newer BC monitoring retained. | `3132a6c8d4e87f64f70db4b65c7e4db5499ed3e6` |
| `ed96a43cca82ad7aae9f31ceeb93558e26cf9f13` | Mixed checkpoint: calibration, samplers, guards, offload and collector already present. Preserve missing executable bits and port rig_collect.sh with recorder failure cap respected. | `08aa87f4e45628ef0d73771c0cdca864c0ad6e77` |
| `da35b42c5cd7d0030808016d4d1d44be0a7a8e0d` | Hold variant retained; BC adds later calibration/training changes. | `29978704ba10f2f2854c96697e4e0b98ceeeb2fb` |
| `ccd0476cb81022149b297e67ea664e8ebb36f554` | Equivalent patch already in BC; no replay. | BC history |
| `3e3f4e5b246ba4694376f2150b019b705c90e586` | Equivalent patch already in BC; no replay. | BC history |
| `e83648d79df921956c6aeb321b328d64592daa6d` | Equivalent patch already in BC; no replay. | BC history |
| `726def53dba4b066f98c6c088d13d061c9a9fb9c` | Equivalent patch already in BC; no replay. | BC history |
| `937a0ea672d0aa258a8d86491a174848d51068d1` | Equivalent patch already in BC; no replay. | BC history |
| `8598f5339bc6471961a8942e0bab59382da38baf` | Hold spatial evidence retained in BC. | `ab5af3d5e25a9b6619ef30e640a2bec063597c7d` |
| `f9b87fc518c74923f015a702f518565923405327` | Equivalent patch already in BC; no replay. | BC history |
| `6ccc28127f370eb61ed6d1e0c80ffcba9fc84941` | Equivalent patch already in BC; no replay. | BC history |
| `aff1eb9111188268d34819b40ba44d8a64e0d0f9` | Equivalent patch already in BC; no replay. | BC history |
| `ef48079c670a5633d4d2a236099b8c034c7fe9b3` | Equivalent patch already in BC; no replay. | BC history |
| `b44eb4830af3b183f8348e0e1b1c52e55f53e9e5` | Equivalent patch already in BC; no replay. | BC history |
| `8c8cc3125b7c3bca3162687e0721f29787b79488` | Equivalent patch already in BC; no replay. | BC history |
| `aed4e61f37482444f78efbb804d8d97c72fa0eb9` | Equivalent patch already in BC; no replay. | BC history |
| `0a68d8267764a066263cab84e383114ab7d6755d` | Equivalent patch already in BC; no replay. | BC history |
| `2572f974c52e9d15f6be35f24b5ec0aee9922be3` | Equivalent patch already in BC; no replay. | BC history |
| `989a013c3908b8c2954875410934ba8392337c47` | Equivalent patch already in BC; no replay. | BC history |
| `a948803d504f07d6d90a2b01a10a0f1e8493222f` | Equivalent patch already in BC; no replay. | BC history |
| `bf1257bf2c5f95456080d079dd2f77a421f71b6b` | Equivalent patch already in BC; no replay. | BC history |
| `ce02051f0c78f7bd0c42bfb8405ed183d3865130` | Equivalent patch already in BC; no replay. | BC history |
| `35238439fad1279cc8e2d6d4ee46066009ae7cef` | Linear MVP retained and subsequently extended in BC. | `54d4932f65212e0fc26272558ec0df1b6fa23e09` |
| `fe3e09e319ffa233965f4a000d030d9d77bd8908` | Port executable permission for mvp_collect_linear.sh. | Reconciliation commit |
| `27e4bc2478dd510d6550252cbbd2a5d29b046f20` | Timing controls already incorporated; newer BC controls retained. | `a0634df77efe6afe197777c3a369d74c50db49cd` |
| `312ab44b00d4325e64d849d0255cb3cafebb895f` | Equivalent patch already in BC; no replay. | BC history |
| `6b42b5d07cf1e722afc637398762f04423102d1f` | Equivalent patch already in BC; no replay. | BC history |
| `0a46f2dc01d8f5801a80074d8a6ae015686778a0` | Equivalent patch already in BC; no replay. | BC history |
| `d2e80044e28cce342003c235e182d0e129589580` | Equivalent patch already in BC; no replay. | BC history |
| `a876aff1a7d39a724c87a95384b24ff8d339dad9` | Equivalent patch already in BC; no replay. | BC history |
| `ed679910e79b5bafc6af992994dc604dce4b7efc` | Port executable permission for finalize_basic_linear_run.sh. | Reconciliation commit |
| `23634863ddd089a64aee5e0e6aba770f0cfc6528` | Equivalent patch already in BC; no replay. | BC history |
| `9f274197dc3607b3f1b585588e574f14b05bb761` | Port executable permission for stop_basic_linear_fleet_at_target.sh. | Reconciliation commit |
| `6fa7bea96d2821013bca5c561fb2170c4ff1bcd8` | Equivalent patch already in BC; no replay. | BC history |
| `00ece9a3f94cc9005e06b126bb48419ae60fab2b` | Equivalent patch already in BC; no replay. | BC history |
| `77bd3767b894864a36827666393a45902f1e6d92` | Direct video encoding incorporated; retain BC short-extract rejection and frame-count verification. | `fcce4b838822fbe34f29be7f0d9be7e96fb7fad8` |
| `b49d18fe364f762eecda01de4e6b52e56b303522` | Equivalent patch already in BC; no replay. | BC history |
| `dfcc659d72a41ec3edf373c7facc664164eee820` | Equivalent patch already in BC; no replay. | BC history |
| `48e920570ac9d467d88475bbc0d085b0bd46d5b5` | Equivalent patch already in BC; no replay. | BC history |
| `0b97aa12cd61f372dc0c2d778d865dbf9d6f4bc9` | Equivalent patch already in BC; no replay. | BC history |
| `92227e12e30bfc0acb390ec9c0ca87679f8ca6e7` | Equivalent patch already in BC; no replay. | BC history |
| `2c23d14136c94425cd2ac3da8f7928478fdb474d` | Equivalent patch already in BC; no replay. | BC history |
| `bb66b12d8b19a22d1abd6f0c75fe53a60ea6179c` | Equivalent patch already in BC; no replay. | BC history |
| `e72ebb44cc3412734ab6a470e29462dd804c0b56` | Equivalent patch already in BC; no replay. | BC history |
| `852e36117ce9a28fb007ec403c614c16d3e23259` | Equivalent patch already in BC; no replay. | BC history |
| `36ac42521aeae5d2600bdbaf128d1e7a25f3be1b` | Equivalent patch already in BC; no replay. | BC history |
| `e252e67cc51856037e5e75bdaa33db1cf481ab46` | Equivalent patch already in BC; no replay. | BC history |
| `0c928ac36caf072887ec4a908df3ac2a5064ef16` | Equivalent patch already in BC; no replay. | BC history |
| `7d69b2c7ece3eac2d35f266be56eefe3cdc876c8` | Equivalent patch already in BC; no replay. | BC history |
| `852495e77c29e4e2972def7c29dcc1c9e45db1d9` | Equivalent patch already in BC; no replay. | BC history |
| `49678587dedce980e49876efb32ec7a819997c6c` | Equivalent patch already in BC; no replay. | BC history |
| `92da5f02ce2d5b8c7663b5f2d30379918f88f9c2` | Equivalent patch already in BC; no replay. | BC history |
| `0b7a775879c4e2fc884c98f20f359c1718885628` | Equivalent patch already in BC; no replay. | BC history |
| `02fdcf41a41aa3b2de3de39efd9bd54ed77b2362` | Equivalent patch already in BC; no replay. | BC history |
| `38cedcdeda3797216ad0742b37c76221404c1bfd` | Equivalent patch already in BC; no replay. | BC history |
| `492629e05e5947cc14a5a77e1733133e5184f131` | Equivalent patch already in BC; no replay. | BC history |
| `85e3176d19521bcc9d807908e694d2087ae39263` | Equivalent patch already in BC; no replay. | BC history |
| `e4f92f9d2ec96216330f8747fe941ac72a01cd5e` | Equivalent patch already in BC; no replay. | BC history |
| `7bd110e67d45d7ce2ba35995a0fdc53bed2658e9` | Equivalent patch already in BC; no replay. | BC history |
| `83ce42983a51f34604b1c98bd177346402cefe53` | Equivalent patch already in BC; no replay. | BC history |
| `35aa98db57985e20f607b1aecc11cfb1ce0dc001` | Equivalent patch already in BC; no replay. | BC history |
| `3435070fcd628a3ffda166fb8705365e4e77dcec` | Equivalent patch already in BC; no replay. | BC history |
| `dd49d0c79ea4363fdf8b35f36cff7f391d1fc0e9` | Equivalent patch already in BC; no replay. | BC history |
| `b8594530c59ca85cda41952743f38b3b3e26765c` | Equivalent patch already in BC; no replay. | BC history |
| `bf49fbfb34adfe424331586ec65e65e135247240` | Equivalent patch already in BC; no replay. | BC history |
| `e5c44ed6614542ed1085bcf55a65033fa2e552c3` | Equivalent patch already in BC; no replay. | BC history |
| `c1f7552f49a62e1510aa71bcdb538813d589b0f5` | Equivalent patch already in BC; no replay. | BC history |
| `311565b0cd00518c12c90c3bd8b0a4741bbb45f6` | Equivalent patch already in BC; no replay. | BC history |
| `af7b28695f8dfca50d28573db3dccab383e80c27` | Equivalent patch already in BC; no replay. | BC history |
| `d6f29484925f7c7c3f61d5acc0934eb0d489ed58` | Equivalent patch already in BC; no replay. | BC history |
| `76d567fd44ea96dded9bb62d28e786b333a69890` | Equivalent patch already in BC; no replay. | BC history |
| `ee4e5997abe22ae4e1dcc91e7d80db17699edaa0` | Equivalent patch already in BC; no replay. | BC history |
| `380d667c7d3eeccebc6177a30ae554a7b01d82de` | Equivalent patch already in BC; no replay. | BC history |
| `8d6bc0294f6074091a1ad4a1b905edc2279ca373` | Equivalent patch already in BC; no replay. | BC history |
| `b8138f01971e1b6a0922d99e8db2f739a56e4b79` | Equivalent patch already in BC; no replay. | BC history |
| `129563009b1b16141497bc576e71ba77d6a2e823` | Equivalent patch already in BC; no replay. | BC history |
| `856e23f92ac3ce3f513f06be6b54b0b1c63dd369` | Equivalent patch already in BC; no replay. | BC history |
| `6a12f12c45bb125dae031d4095040daddbf04b6c` | Equivalent patch already in BC; no replay. | BC history |
| `b76ecd9a057000b68ead5407d154d2dcf6686fd6` | Equivalent patch already in BC; no replay. | BC history |
| `2406e4e8462cd943719c241d205873e41a6cac64` | Equivalent patch already in BC; no replay. | BC history |
| `167c4e804669e4ee5467ebca51b6b83d7d0a9142` | Equivalent patch already in BC; no replay. | BC history |
| `f3eb43e757a9e1dbf765207c978034602703b34a` | Equivalent patch already in BC; no replay. | BC history |
| `96af47febc69844cd5422af86e9ea0ef239d353f` | Equivalent patch already in BC; no replay. | BC history |
| `ae7cde3c4a71f0df0b0b38cf27711165c709915b` | Equivalent patch already in BC; no replay. | BC history |
| `5e1508d58d574b8e8440b0de213f20162614146e` | Equivalent patch already in BC; no replay. | BC history |
| `1e2e562fcb65f91b15823b215baadd2254975b29` | Equivalent patch already in BC; no replay. | BC history |
| `541da3175012bfebaa606f59b0d46c0d106795bc` | Equivalent patch already in BC; no replay. | BC history |
| `3db1baab6810d818d16b8b273e1a3ed64196a159` | Equivalent patch already in BC; no replay. | BC history |
| `fb4f3e38b1a05074c06633954a62060699ccd167` | Equivalent patch already in BC; no replay. | BC history |
| `da5278295550dd9a58383d7b77a05f7012d7eba1` | Equivalent patch already in BC; no replay. | BC history |
| `396df01e647324301ee306a9e378460b16be0d6f` | Equivalent patch already in BC; no replay. | BC history |
| `5947e88ea246eb9d087515c660ea3e5e039b3d3f` | Equivalent patch already in BC; no replay. | BC history |
| `896dd4032d1f1992c1bc2cf93c76974a5c84b547` | Equivalent patch already in BC; no replay. | BC history |
| `1fb48b162815a95d464782b0f002f6b3be1280fc` | Equivalent patch already in BC; no replay. | BC history |
| `11ea66b4f9848988f113a8eea8b7af57347f19b6` | Equivalent patch already in BC; no replay. | BC history |
| `7e2cc91aecd5f82ce682f927069a0903cffa3d37` | Fresh holdout support incorporated; retain later BC evaluation/resume safeguards. | `d77debb960f7a4a0c137ad42dd703d1d75f6fc63` |
| `f50d2f39f239a1bdfc078c4061b5e223f5bca3cc` | Equivalent patch already in BC; no replay. | BC history |
| `7639a214cb4ef80262291c7d1ccf521e2ec16ba8` | Equivalent patch already in BC; no replay. | BC history |
| `3a3cae3cee325450e381bb1b4f3257388b414600` | Equivalent patch already in BC; no replay. | BC history |
| `890c0e6867cbecbeb11187fd6afd43399e5f6cc3` | Equivalent patch already in BC; no replay. | BC history |
| `463316d34b81129986a171920369dd6067e91f7b` | Equivalent patch already in BC; no replay. | BC history |
| `7c0a26cf9cc44cee129251f6e7d7a13000673429` | BC ancestry merge; preserved, no independent replay. | BC history |
| `f9a30520bb79b7d6060826afcc8cb7254f55e4b0` | BC ancestry merge; preserved, no independent replay. | BC history |

## Live tracked source disposition

Compared with original BC, including uncommitted rig edits. A rig deletion relative to BC often means BC added the file later, not that the rig intentionally removed it.

| Path | Disposition |
|---|---|
| `scripts/cloud/train_basic_linear_modal.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. Preserve BC durable resume, sizing and evaluation safeguards. |
| `scripts/cloud/train_trace_extractor_modal.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `scripts/data/align_xctest_traces.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. Rig lacks decoded-frame-count safeguard. Replace obsolete FFmpeg -vsync with equivalent -fps_mode passthrough separately. |
| `scripts/data/audit_basic_linear_corpus.py` | BC-only addition retained. |
| `scripts/data/build_bc_clips.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `scripts/data/build_model1_scaling_manifests.py` | BC-only addition retained. |
| `scripts/data/build_model1_shards.py` | BC-only addition retained. |
| `scripts/data/collect_sls_xctest.py.bak_editorfix` | Historical backup/WIP preserved only; current BC implementation retained. |
| `scripts/data/collect_sls_xctest.py.bak_pre_foreground_guard` | Historical backup/WIP preserved only; current BC implementation retained. |
| `scripts/data/collect_sls_xctest.py.orig` | Historical backup/WIP preserved only; current BC implementation retained. |
| `scripts/data/flag_editor_samples.py.rig_wip` | Historical backup/WIP preserved only; current BC implementation retained. |
| `scripts/data/flag_menu_samples.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `scripts/inspect/render_linear_failures.py` | BC-only addition retained. |
| `scripts/inspect/review_modal_corpus.py` | BC-only addition retained. |
| `scripts/inspect/run_sequence_policy.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `scripts/ops/finalize_basic_linear_stage1.sh` | BC-only addition retained. |
| `scripts/ops/mvp_collect_linear.sh.orig` | Historical backup/WIP preserved only; current BC implementation retained. |
| `scripts/ops/offload_corpus_to_modal.sh.bak.20260820` | Historical backup/WIP preserved only; current BC implementation retained. |
| `scripts/ops/start_basic_linear_stage1.sh` | BC-only addition retained. |
| `scripts/rig_collect.sh` | Port entrypoint; remove endless recorder restart/notification loop; preserve explicit experimental SLS defaults. |
| `scripts/rig_collect.sh.spinfrac-backup` | Historical backup/WIP preserved only; current BC implementation retained. |
| `scripts/train/estimate_model1_scaling_cost.py` | BC-only addition retained. |
| `scripts/train/train_basic_linear_regressor.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. Preserve BC durable resume, sizing and evaluation safeguards. |
| `scripts/train/train_sequence_model.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `scripts/train/train_temporal_trace_extractor.py` | BC-only addition retained. |
| `scripts/train/train_trace_extractor.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `src/trueskate_ai/bc/infer.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. Preserve BC causal action groups and activity-mask contract. |
| `src/trueskate_ai/bc/model2.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. Preserve BC causal action groups and activity-mask contract. |
| `src/trueskate_ai/bc/sequence_dataset.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. Preserve BC causal action groups and activity-mask contract. |
| `src/trueskate_ai/data/cohort_manifest.py` | BC-only addition retained. |
| `src/trueskate_ai/data/sequential_shards.py` | BC-only addition retained. |
| `src/trueskate_ai/data/trajectory_resample.py` | BC-only addition retained. |
| `src/trueskate_ai/vision/basic_linear_audit.py` | BC-only addition retained. |
| `src/trueskate_ai/vision/basic_linear_bias.py` | BC-only addition retained. |
| `src/trueskate_ai/vision/basic_linear_dataset.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `src/trueskate_ai/vision/basic_linear_regressor.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `src/trueskate_ai/vision/basic_linear_training.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `src/trueskate_ai/vision/gameplay_filter.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `src/trueskate_ai/vision/gameplay_filter.py.bak_editorfix` | Historical backup/WIP preserved only; current BC implementation retained. |
| `src/trueskate_ai/vision/model1_certification.py` | BC-only addition retained. |
| `src/trueskate_ai/vision/model1_scaling.py` | BC-only addition retained. |
| `src/trueskate_ai/vision/temporal_trace_predictor.py` | BC-only addition retained. |
| `src/trueskate_ai/vision/temporal_trace_training.py` | BC-only addition retained. |
| `src/trueskate_ai/vision/touch_peaks.py` | BC-only addition retained. |
| `tests/test_align_video_frame_count.py` | BC-only addition retained. |
| `tests/test_basic_linear_audit.py` | BC-only addition retained. |
| `tests/test_basic_linear_bias.py` | BC-only addition retained. |
| `tests/test_basic_linear_experiment.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `tests/test_bc_multitouch_tracking.py` | BC-only addition retained. |
| `tests/test_bc_temporal_trace_inference.py` | BC-only addition retained. |
| `tests/test_collection_watchdog.py` | BC-only addition retained. |
| `tests/test_flag_menu_samples.py` | BC-only addition retained. |
| `tests/test_gameplay_filter.py` | BC-only addition retained. |
| `tests/test_model1_scaling_protocol.py` | BC-only addition retained. |
| `tests/test_model2_causal_timeline.py` | BC-only addition retained. |
| `tests/test_mvp3_trajectory.py` | BC-only addition retained. |
| `tests/test_offload_spin_provenance.py` | BC-only addition retained. |
| `tests/test_tap_timing_calibration.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |
| `tests/test_temporal_trace_dataset.py` | BC-only addition retained. |
| `tests/test_temporal_trace_predictor.py` | BC-only addition retained. |
| `tests/test_temporal_trace_trainer.py` | BC-only addition retained. |
| `tests/test_temporal_trace_training.py` | BC-only addition retained. |
| `tests/test_train_dashboard_preview.py` | Retain newer BC implementation and tests; rig version preserved at snapshot. |

## Operational state and evidence

- Dashboard, collector, service launcher, watchdog and offload source on disk match original BC once rig edits are included. Installed dashboard is running; loaded code identity was not inferred from disk hashes.
- Eight installed user launchd definitions are preserved under `preservation/deployed-launchagents/` in the rig snapshot. Root remotexpc tunnel is running; it remains installed in place.
- Both WDA endpoints refused connections and `idevice_id -l` returned no devices. Logs show USB wait loops. Physical smoke/deployment is blocked until phones return; no service restart was attempted.
- Baseline: 247 passed, one synthetic calibration failure with FFmpeg 9.0.1 (`-vsync` removed). Replacing it with `-fps_mode passthrough` passes all 16 calibration/alignment tests. Rig FFmpeg is 8.1.2.
- Local `data/`, `logs/`, `notebooks/models/` and rig counterparts remain untouched. Historical Modal volume names: `trueskate-corpus`, `trueskate-corpus-v2`, `trueskate-mvp`, `trueskate-models`. Availability/backups of cloud artifacts were not verified and no paid job was launched.
- Preserve .env on each machine independently; never put credentials in the archive. Data fingerprints and model manifests remain authoritative for individual experiments.

