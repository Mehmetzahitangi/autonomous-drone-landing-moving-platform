"""
diag_failtype.py — Başarısızlık TÜRÜ dökümü: "düşen yüzde kırım mı, ıskalama mı?"

Soru: shielded modelin başarısı faz4c'den düşük. Peki bu düşüşler TEHLİKELİ kırım mı
(sert çarpma / devrilme → donanım zararı), yoksa GÜVENLİ ıskalama mı (yavaş yaklaşıp
konamama / süre dolması → çarpma yok)? "Güvenlik önceliği" iddiası ancak düşüşler
yumuşaksa geçerlidir.

Her bölümün SON adımında env'in sonlanma dallarını yeniden kurup sınıflandırırız
(_computeTerminated ile birebir mantık):
  BASARI        : horiz<0.3, h<0.05, vmag<0.5
  KIRIM_carpma  : horiz<0.3, h<0.05, vmag>=0.5     [TEHLIKELI]
  DEVRILME      : |roll|>1.2 veya |pitch|>1.2       [TEHLIKELI]
  ISKALAMA_yer  : z < PLATFORM_TOP-0.05 (platformu kaçırıp yere inme)  [yumusak]
  TIMEOUT       : süre doldu, hâlâ havada/konamadı                     [yumusak]

Karşılaştırma: faz4c(CBF kapalı) vs shielded(CBF açık), aynı eşli rüzgâr.
Kritik satır: TEHLIKELI toplam (kırım+devrilme). Shielded'da ~0 olmalı.

Kullanım: python rl/diag_failtype.py
"""
import sys, pathlib, os
PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID

SYSTEMS = [
    ("faz4c (CBF kapali)",  PROJ/"reports/faz4c_heave/sac_landing.zip",    PROJ/"reports/faz4c_heave/vecnormalize.pkl",    False),
    ("shielded (CBF acik)", PROJ/"reports/faz5_shielded/sac_shielded.zip", PROJ/"reports/faz5_shielded/vecnorm_shielded.pkl", True),
]
MAGS    = [0.00, 0.04, 0.08]   # nominal + orta + yüksek
N_DIRS  = 8
N_SEEDS = 15
DIRS = [2*np.pi*k/N_DIRS for k in range(N_DIRS)]

CATS = ["BASARI", "KIRIM_carpma", "DEVRILME", "ISKALAMA_yer", "TIMEOUT", "DIGER"]


def classify(aviary, trunc):
    """Son adım durumunu _computeTerminated mantığıyla sınıflandır."""
    s = aviary._getDroneStateVector(0)
    r_rel, v_rel = aviary._relative_state()
    h = max(0.0, -r_rel[2])
    horiz = np.linalg.norm(r_rel[0:2])
    vmag = np.linalg.norm(v_rel)
    if horiz < 0.3 and h < 0.05 and vmag < 0.5:
        return "BASARI"
    if horiz < 0.3 and h < 0.05 and vmag >= 0.5:
        return "KIRIM_carpma"
    if s[2] < aviary.PLATFORM_TOP - 0.05:
        return "ISKALAMA_yer"
    if abs(s[7]) > 1.2 or abs(s[8]) > 1.2:
        return "DEVRILME"
    if trunc:
        return "TIMEOUT"
    return "DIGER"


def load(mp, vp):
    stat = make_vec_env(LandingAviary, n_envs=1,
                        env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=False))
    vn = VecNormalize.load(str(vp), stat); vn.training = False
    return SAC.load(str(mp)), vn


def run_cell(aviary, model, vn, wind_vec, seeds):
    counts = {c: 0 for c in CATS}
    for s in seeds:
        np.random.seed(s)
        obs, info = aviary.reset()
        aviary._wind = np.array(wind_vec, dtype=float)
        term = trunc = False
        while not (term or trunc):
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(nobs, deterministic=True)
            obs, r, term, trunc, info = aviary.step(action)
            aviary._wind = np.array(wind_vec, dtype=float)
        counts[classify(aviary, trunc)] += 1
    return counts


def eval_system(label, mp, vp, use_cbf):
    if not (os.path.exists(mp) and os.path.exists(vp)):
        print(f"[ATLANDI] {label}: dosya yok\n"); return
    model, vn = load(mp, vp)
    aviary = LandingAviary(gui=False, act=ACT, obs=ObservationType.KIN, use_cbf=use_cbf)
    seeds = list(range(N_SEEDS))
    print(f"Sistem: {label}")
    print(f"{'ruzgar':>7} {'N':>4} | {'BASARI':>7} {'KIRIM':>6} {'DEVRIL':>7} {'ISKALA':>7} {'TIMEOUT':>8} | {'TEHLIKELI':>10}")
    print("-" * 66)
    for mag in MAGS:
        tot = {c: 0 for c in CATS}
        if mag == 0.0:
            c = run_cell(aviary, model, vn, [0, 0, 0], seeds)
            for k in CATS: tot[k] += c[k]
        else:
            for ang in DIRS:
                wv = [mag*np.cos(ang), mag*np.sin(ang), 0.0]
                c = run_cell(aviary, model, vn, wv, seeds)
                for k in CATS: tot[k] += c[k]
        n = sum(tot.values())
        tehlikeli = tot["KIRIM_carpma"] + tot["DEVRILME"]
        print(f"{mag:>7.2f} {n:>4} | {tot['BASARI']:>7} {tot['KIRIM_carpma']:>6} "
              f"{tot['DEVRILME']:>7} {tot['ISKALAMA_yer']:>7} {tot['TIMEOUT']:>8} | "
              f"{tehlikeli:>10}")
    print()
    aviary.close()


print(f"=== BASARISIZLIK TURU DOKUMU | {N_DIRS} yon x {N_SEEDS} esli tohum ===")
print("TEHLIKELI = KIRIM(sert carpma) + DEVRILME. Guvenlik iddiasi icin bu ~0 olmali.\n")
for label, mp, vp, cbf in SYSTEMS:
    eval_system(label, mp, vp, cbf)
