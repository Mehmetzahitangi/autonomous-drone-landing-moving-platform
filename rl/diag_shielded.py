"""
diag_shielded.py — Shielded retrain DOĞRULAMASI.

İki DEPLOY edilebilir sistemi kendi tasarlandığı konfigde yan yana koyar:
  A) faz4c tek başına  (CBF KAPALI)  — Faz 5 kazananı, referans.
  B) shielded model    (CBF AÇIK)    — faz4c'den warm-start, kalkan+rüzgarla retrain.

Aynı held-out rüzgar ızgarası (8 yön × eşli tohum), her hücrede aynı platformlar.
Bakılan: (1) başarı(en-kötü-yön) B'de düşmedi mi? (bolt-on'da 100→87 düşmüştü)
         (2) maks vrel (temas hızı) B'de düşük mü? (kalkanın işi, yüksek rüzgarda).
İkisi birden iyiyse: "rüzgarda hem başarıyla hem güvenli temas hızıyla inen" TEK model.

Kullanım: python rl/diag_shielded.py
"""
import sys, pathlib, os
PROJ = pathlib.Path(__file__).resolve().parent.parent   # CWD'den bağımsız
sys.path.insert(0, str(PROJ))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID

# (etiket, model, vecnorm, use_cbf)
SYSTEMS = [
    ("faz4c (CBF kapali)",    PROJ/"reports/faz4c_heave/sac_landing.zip",      PROJ/"reports/faz4c_heave/vecnormalize.pkl",      False),
    ("shielded (CBF acik)",   PROJ/"reports/faz5_shielded/sac_shielded.zip",   PROJ/"reports/faz5_shielded/vecnorm_shielded.pkl", True),
]
MAGS    = [0.00, 0.02, 0.04, 0.06, 0.08]
N_DIRS  = 8
N_SEEDS = 15

DIRS = [2*np.pi*k/N_DIRS for k in range(N_DIRS)]


def load(model_path, vn_path):
    stat_venv = make_vec_env(LandingAviary, n_envs=1,
                             env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=False))
    vn = VecNormalize.load(str(vn_path), stat_venv)
    vn.training = False
    model = SAC.load(str(model_path))
    return model, vn


def run_cell(aviary, model, vn, wind_vec, seeds):
    succ = 0; vrels = []
    for s in seeds:
        np.random.seed(s)
        obs, info = aviary.reset()
        aviary._wind = np.array(wind_vec, dtype=float)
        term = trunc = False
        fin_vrel = np.nan
        while not (term or trunc):
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(nobs, deterministic=True)
            obs, r, term, trunc, info = aviary.step(action)
            aviary._wind = np.array(wind_vec, dtype=float)
            _, v_rel = aviary._relative_state()
            fin_vrel = float(np.linalg.norm(v_rel))
        succ += bool(info.get("is_success"))
        vrels.append(fin_vrel)
    return succ, np.array(vrels)


def eval_system(label, model_path, vn_path, use_cbf):
    if not (os.path.exists(model_path) and os.path.exists(vn_path)):
        print(f"[ATLANDI] {label}: dosya yok ({model_path})\n"); return None
    model, vn = load(model_path, vn_path)
    aviary = LandingAviary(gui=False, act=ACT, obs=ObservationType.KIN, use_cbf=use_cbf)
    seeds = list(range(N_SEEDS))
    rows = []
    print(f"Sistem: {label}  (use_cbf={use_cbf})")
    print(f"{'ruzgar(N)':>9} {'basari(ort)':>12} {'basari(en-kotu-yon)':>20} {'ort vrel':>9} {'maks vrel':>10}")
    print("-" * 64)
    for mag in MAGS:
        if mag == 0.0:
            succ, vrels = run_cell(aviary, model, vn, [0.0, 0.0, 0.0], seeds)
            rate = 100.0 * succ / N_SEEDS
            row = (mag, rate, rate, float(vrels.mean()), float(vrels.max()))
        else:
            dir_rates = []; all_vrels = []
            for ang in DIRS:
                wv = [mag*np.cos(ang), mag*np.sin(ang), 0.0]
                succ, vrels = run_cell(aviary, model, vn, wv, seeds)
                dir_rates.append(100.0 * succ / N_SEEDS); all_vrels.append(vrels)
            all_vrels = np.concatenate(all_vrels)
            row = (mag, float(np.mean(dir_rates)), float(np.min(dir_rates)),
                   float(all_vrels.mean()), float(all_vrels.max()))
        rows.append(row)
        print(f"{row[0]:>9.2f} {row[1]:>11.1f}% {row[2]:>19.1f}% {row[3]:>9.3f} {row[4]:>10.3f}")
    print()
    aviary.close()
    return rows


print(f"=== SHIELDED DOĞRULAMA | {N_DIRS} yön x {N_SEEDS} eşli tohum ===\n")
res = {}
for label, mp, vp, cbf in SYSTEMS:
    r = eval_system(label, mp, vp, cbf)
    if r is not None:
        res[label] = r

if len(res) == 2:
    la, lb = list(res.keys())
    print("=== Karsilastirma: basari(en-kotu-yon) ve MAKS VREL ===")
    print(f"{'ruzgar(N)':>9} | {la[:14]+' kotu':>20} {lb[:14]+' kotu':>20} | {'maksV A':>9} {'maksV B':>9}")
    print("-" * 74)
    for i, mag in enumerate(MAGS):
        print(f"{mag:>9.2f} | {res[la][i][2]:>19.1f}% {res[lb][i][2]:>19.1f}% | {res[la][i][4]:>9.3f} {res[lb][i][4]:>9.3f}")
    print("\nHedef: B(shielded) basarisi A(faz4c) ~= veya >= (nominal dusmedi)")
    print("ve B maks vrel yuksek ruzgarda A'dan belirgin dusuk (kalkan temas hizini kapatti).")
