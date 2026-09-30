import json, statistics as st, sys, math
d=json.load(open(sys.argv[1]))
def true_lift(r): return min(.24 + r["target_duration"]/2.27, .88)
def mcnemar(b,c):
    n=b+c; k=min(b,c)
    p=sum(math.comb(n,i) for i in range(k+1))/2**n*2 if n else 1
    return min(p,1)
for n in d["checkpoints"]:
    rows=d["per_sample"][n]; tag=n[-9:-4]
    fails=[r for r in rows if not r["baseline"]["end_ok"] and r["baseline"]["along"]<0]
    oks=[r for r in rows if r["baseline"]["end_ok"]]
    f=lambda rs,g: st.median(g(r) for r in rs)
    print(f"{tag}: end-short fails {len(fails)}")
    print(f"   attn time − true liftoff (norm, ×2.27 s): fails {f(fails,lambda r:r['attention_time']-true_lift(r))*2.27:+.3f}s  ok {f(oks,lambda r:r['attention_time']-true_lift(r))*2.27:+.3f}s")
    print(f"   attn earlier than true liftoff: fails {sum(r['attention_time']<true_lift(r) for r in fails)/len(fails):.0%}  ok {sum(r['attention_time']<true_lift(r) for r in oks)/len(oks):.0%}")
    print(f"   pred − true duration: fails {f(fails,lambda r:r['predicted_duration']-r['target_duration']):+.3f}s  ok {f(oks,lambda r:r['predicted_duration']-r['target_duration']):+.3f}s")
    print(f"   true duration: fails {f(fails,lambda r:r['target_duration']):.3f}s  ok {f(oks,lambda r:r['target_duration']):.3f}s")
    print(f"   baseline along-path error of short fails: median {f(fails,lambda r:r['baseline']['along']):+.4f}")
    for v in ["sigma_0.075","sigma_0.05","oracle_liftoff_sigma_0.05","oracle_frame"]:
        b=sum(r[v]["recovered"] and not r["baseline"]["recovered"] for r in rows)
        c=sum(r["baseline"]["recovered"] and not r[v]["recovered"] for r in rows)
        print(f"   {v:26s} gained {b:4d} lost {c:4d}  net {b-c:+d}  McNemar p={mcnemar(b,c):.2g}")
