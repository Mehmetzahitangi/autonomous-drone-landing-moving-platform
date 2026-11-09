"""
eval_faz7.py — Faz 7 standart değerlendirme (ablasyon tablosunun her satırı bununla üretilir).

Özellikler:
  * Sabit seed listesi (varsayılan 0..99) → her konfig AYNI senaryolarla (eşli) ölçülür.
  * Doğru model–kalkan eşleşmesi: faz4c CBF'siz eğitildi → CBF kapalı; faz6 CBF ile eğitildi → açık.
  * Sonuç türleri: BASARI / KIRIM / ISKALA / DEVRILME / TIMEOUT / SINIR.
  * Temasta dikey ve yatay bağıl hız AYRI (KIRIM sert oturma mı, kayma mı?).
  * Algı istatistikleri: tespit oranı (tag bulunan kare / yakalanan kare), maks bayatlık,
    kestirim hatası (ortalama / maks / temas anı).
  * Başarı için Wilson %95 güven aralığı.
  * Çıktı: reports/faz7_eval/<tag>/episodes.csv + summary.json

Örnek:
  python rl/eval_faz7.py --tag taban_ideal     --model faz4c --cam ideal
  python rl/eval_faz7.py --tag taban_gercekci  --model faz4c --cam realistic
  python rl/eval_faz7.py --tag faz6_gercekci   --model faz6  --cam realistic      (CBF otomatik açık)
"""
import sys, pathlib, argparse, json, csv, time, pickle
PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import numpy as np
from stable_baselines3 import SAC
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

MODELS = {   # ad: (model, vecnormalize, eğitimde CBF açık mıydı)
    "faz4c": ("reports/faz4c_heave/sac_landing.zip", "reports/faz4c_heave/vecnormalize.pkl", False),
    "faz6":  ("reports/faz6_cbf/sac_landing.zip",    "reports/faz6_cbf/vecnormalize.pkl",    True),
    # Faz 7 ortamında (kapı + PID reset + 12 s) Gauss sensörle eğitilenler — rl/train_faz7.py
    "faz7":     ("reports/faz7_res/sac_landing.zip",     "reports/faz7_res/vecnormalize.pkl",     False),
    "faz7_cbf": ("reports/faz7_res_cbf/sac_landing.zip", "reports/faz7_res_cbf/vecnormalize.pkl", True),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z*z/n
    c = (ph + z*z/(2*n)) / d
    hw = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-hw), min(1.0, c+hw))


def outcome(av, info, terminated, truncated):
    """Sonuç türü. Temas ölçütleri env'in BİTİŞ adımındaki info'dan (adım sayacı ilerlemeden önce
    hesaplanan horiz/h/vrel) okunur — sonradan _relative_state çağırmak bir adım kaymış platformla
    ölçer ve sınır bölümleri hiçbir sınıfa düşürmez (eski 'DIGER' hatası)."""
    s = av._getDroneStateVector(0)
    if info["horiz"] < 0.3 and info["h"] < 0.05:
        return "BASARI" if info["touchdown_speed"] < 0.5 else "KIRIM"
    if s[2] < av.PLATFORM_TOP - 0.05:
        return "ISKALA"
    if abs(s[7]) > 1.2 or abs(s[8]) > 1.2:
        return "DEVRILME"
    if truncated:
        if abs(s[0]) > 2.0 or abs(s[1]) > 2.0 or s[2] > 3.0:
            return "SINIR"
        return "TIMEOUT"
    return "DIGER"


def load_vecnorm(path):
    """VecNormalize'ı İKİNCİ bir PyBullet client açmadan yükle (yalnız normalize_obs gerekiyor)."""
    with open(path, "rb") as f:
        vn = pickle.load(f)
    vn.training = False
    return vn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="çıktı klasör adı (reports/faz7_eval/<tag>)")
    ap.add_argument("--model", default="faz4c", choices=MODELS)
    ap.add_argument("--cam", default="realistic", help="ideal | realistic | JSON dict")
    ap.add_argument("--cbf", default="auto", choices=["auto", "on", "off"],
                    help="auto = modelin eğitildiği gibi")
    ap.add_argument("--sensor", default="aruco", choices=["aruco", "gauss"],
                    help="aruco = kamera; gauss = eski idealize sensör (gerçek + N(0,3cm), sağlama)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--ep-len", type=float, default=None, help="bölüm süresi (s); varsayılan env'deki")
    ap.add_argument("--no-rl", action="store_true", help="residual=0 → saf klasik")
    ap.add_argument("--env", default="{}", help='ek env ayarlari JSON, or. {"align_cfg": {"vtol": 0.3}}')

    args = ap.parse_args()

    mpath, vpath, trained_cbf = MODELS[args.model]
    use_cbf = trained_cbf if args.cbf == "auto" else (args.cbf == "on")
    cam = json.loads(args.cam) if args.cam.strip().startswith("{") else args.cam

    out = PROJ / "reports/faz7_eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    model = SAC.load(str(PROJ / mpath))
    vn = load_vecnorm(PROJ / vpath)
    env_kw = json.loads(args.env)
    av = LandingAviary(gui=False, act=ActionType.PID, obs=ObservationType.KIN,
                       use_aruco=(args.sensor == "aruco"), use_cbf=use_cbf, cam_cfg=cam, **env_kw)
    if args.ep_len:
        av.EPISODE_LEN_SEC = args.ep_len

    cfg = dict(tag=args.tag, model=args.model, sensor=args.sensor, use_cbf=use_cbf, cam_cfg=av.cam_cfg,
               n=args.n, seed0=args.seed0, ep_len=av.EPISODE_LEN_SEC, env=env_kw, no_rl=args.no_rl,
               align_cfg=getattr(av, "align_cfg", None))
    print("KONFIG:", json.dumps(cfg, default=str))
    print(f"{'seed':>4} {'sonuc':>9} {'t(s)':>5} {'vz':>6} {'vxy':>5} {'horiz':>5} "
          f"{'tespit%':>7} {'maxBayat':>8} {'estOrt':>6} {'estMax':>6}")

    rows, t0 = [], time.time()
    for i in range(args.n):
        seed = args.seed0 + i
        np.random.seed(seed)
        obs, info = av.reset()
        term = trunc = False
        k = 0; max_stale = 0; est = []; cbf_n = 0
        while not (term or trunc):
            a, _ = model.predict(vn.normalize_obs(np.asarray(obs, dtype=np.float32)), deterministic=True)
            if args.no_rl:
                a = np.zeros_like(a)                     # residual = 0 → saf klasik taban
            obs, r, term, trunc, info = av.step(a)
            k += 1
            max_stale = max(max_stale, info.get("stale", 0))
            est.append(info.get("est_err", np.nan))
            if use_cbf and getattr(av, "_dbg_cbf", {}).get("intervened", False):
                cbf_n += 1
        res = outcome(av, info, term, trunc)
        det = info.get("cam_dets", 1) / max(info.get("cam_frames", 1), 1)
        row = dict(seed=seed, outcome=res, steps=k, t_end=k / av.CTRL_FREQ,
                   vz_rel=info["vz_rel"], vxy_rel=info["vxy_rel"], vrel=info["touchdown_speed"],
                   horiz=info["horiz"], h=info["h"], det_rate=det, max_stale=max_stale,
                   est_mean=float(np.nanmean(est)), est_max=float(np.nanmax(est)), est_end=est[-1],
                   cbf_interventions=cbf_n,
                   hold_frac=info.get("af_hold_steps", 0) / max(k, 1),
                   plat_amp=av.plat_amp, plat_freq=av.plat_freq, heave_amp=av.heave_amp,
                   drift=float(np.linalg.norm(av.plat_drift)))
        rows.append(row)
        print(f"{seed:>4} {res:>9} {row['t_end']:>5.2f} {row['vz_rel']:>6.2f} {row['vxy_rel']:>5.2f} "
              f"{row['horiz']:>5.2f} {100*det:>7.1f} {max_stale:>8} {100*row['est_mean']:>5.1f}c "
              f"{100*row['est_max']:>5.1f}c", flush=True)
    av.close()

    with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    cnt = {}
    for r_ in rows:
        cnt[r_["outcome"]] = cnt.get(r_["outcome"], 0) + 1
    k_ok = cnt.get("BASARI", 0)
    lo, hi = wilson(k_ok, n)
    contact = [r_ for r_ in rows if r_["outcome"] in ("BASARI", "KIRIM")]
    crash = [r_ for r_ in rows if r_["outcome"] == "KIRIM"]
    summ = dict(
        config=cfg, n=n, success=k_ok / n, success_ci95=[lo, hi], outcomes=cnt,
        det_rate_mean=float(np.mean([r_["det_rate"] for r_ in rows])),
        max_stale_median=float(np.median([r_["max_stale"] for r_ in rows])),
        est_err_mean_cm=100*float(np.mean([r_["est_mean"] for r_ in rows])),
        contact_vrel_p95=float(np.percentile([r_["vrel"] for r_ in contact], 95)) if contact else None,
        crash_vz_mean=float(np.mean([abs(r_["vz_rel"]) for r_ in crash])) if crash else None,
        crash_vxy_mean=float(np.mean([r_["vxy_rel"] for r_ in crash])) if crash else None,
        hold_frac_mean=float(np.mean([r_["hold_frac"] for r_ in rows])),
        t_end_mean=float(np.mean([r_["t_end"] for r_ in rows])),
        wall_time_min=(time.time() - t0) / 60,
    )
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"[{args.tag}] BASARI {k_ok}/{n} = {100*k_ok/n:.0f}%   (%95 GA: {100*lo:.0f}–{100*hi:.0f}%)")
    print("sonuclar:", cnt)
    print(f"tespit orani ort: {100*summ['det_rate_mean']:.1f}%   maks bayatlik medyan: {summ['max_stale_median']:.0f} adim"
          f"   kestirim hatasi ort: {summ['est_err_mean_cm']:.1f} cm")
    print(f"bekleme orani ort: {100*summ['hold_frac_mean']:.0f}%   bolum suresi ort: {summ['t_end_mean']:.2f} s")
    if crash:
        print(f"KIRIM ort |vz|={summ['crash_vz_mean']:.2f}  ort vxy={summ['crash_vxy_mean']:.2f} m/s "
              f"(hangisi buyukse sorun o)")
    print(f"sure: {summ['wall_time_min']:.1f} dk   cikti: {out}")


if __name__ == "__main__":
    main()
