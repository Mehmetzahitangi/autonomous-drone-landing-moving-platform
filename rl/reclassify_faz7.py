"""reclassify_faz7.py — eski eval_faz7 çıktılarındaki 'DIGER' satırlarını info değerleriyle yeniden sınıflar
(episodes.csv'deki horiz/h/vrel bitiş adımındaki info'dan gelir) ve summary.json'u günceller."""
import sys, pathlib, csv, json
PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from rl.eval_faz7 import wilson

for tag in sys.argv[1:]:
    d = PROJ / "reports/faz7_eval" / tag
    rows = list(csv.DictReader(open(d / "episodes.csv", encoding="utf-8")))
    fixed = 0
    for r in rows:
        if r["outcome"] == "DIGER" and float(r["horiz"]) < 0.3 and float(r["h"]) < 0.05:
            r["outcome"] = "BASARI" if float(r["vrel"]) < 0.5 else "KIRIM"; fixed += 1
    with open(d / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    S = json.load(open(d / "summary.json", encoding="utf-8"))
    cnt = {}
    for r in rows: cnt[r["outcome"]] = cnt.get(r["outcome"], 0) + 1
    n = len(rows); k = cnt.get("BASARI", 0); lo, hi = wilson(k, n)
    crash = [r for r in rows if r["outcome"] == "KIRIM"]
    S.update(success=k/n, success_ci95=[lo, hi], outcomes=cnt,
             crash_vz_mean=(sum(abs(float(r["vz_rel"])) for r in crash)/len(crash)) if crash else None,
             crash_vxy_mean=(sum(float(r["vxy_rel"]) for r in crash)/len(crash)) if crash else None)
    json.dump(S, open(d / "summary.json", "w", encoding="utf-8"), indent=2, default=str)
    print(f"{tag:>20}: {fixed} DIGER yeniden siniflandi -> BASARI {k}/{n} ({100*lo:.0f}-{100*hi:.0f}%)  {cnt}")
