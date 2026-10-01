import json, statistics as st
tot = 0
vals = []
for s in 0, 1, 2:
    d = json.load(open(f"/tmp/exp_s{s}.json")); v = d["validation"]; c = d["validation_curve"]
    secs = sum(e["seconds"] for e in d.get("epoch_history") or [])
    tot += secs
    vals.append(d["validation_plateau_mean_last10"])
    print(f"seed{s}: last10 {d['validation_plateau_mean_last10']*100:.2f} best {max(c)*100:.2f} (ep {d['best_epoch']}) final {c[-1]*100:.2f} sd10 {st.pstdev(c[-10:])*100:.2f} "
          f"start/end/dur {v['start_recovery_accuracy']*100:.1f}/{v['end_recovery_accuracy']*100:.1f}/{v['duration_recovery_accuracy']*100:.1f} "
          f"img {d['image_width']}x{d['image_height']} sched {d['lr_schedule']} hours {secs/3600:.2f} ckpt {d['checkpoint']} sha {d['checkpoint_sha256'][:12]}")
m = st.mean(vals)
print(f"mean last10 {m*100:.3f}  error {(1-m)*100:.3f}  reduction vs 10.83%: {(10.83-(1-m)*100)/10.83*100:.1f}%  train hours {tot/3600:.2f}  ~${tot/3600*1.3166:.2f}")
