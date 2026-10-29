"""
diag_landing.py — iniş teşhisi + IMM-EKF kestirim kalitesi (v3).

Rollout'u HAM LandingAviary üzerinde yapar (VecEnv DEĞİL) → auto-reset yok,
terminal durumu ve terminated/truncated DOĞRU okunur. VecNormalize yalnız
obs normalizasyonu için kullanılır.

HER episode için:
  İNİŞ (GERÇEK durumdan — _relative_state):
    - min h, o andaki horiz/vrel ; min horiz ; min drone_z ; max tilt
    - başarı kapıları (h<0.05 / horiz<0.3 / ikisi) ; bitiş nedeni
  KESTİRİM (IMM-EKF ↔ gerçek platform):
    - pos_RMSE : platform (x,y) konum kestirim hatası (m)
    - vel_RMSE : platform (x,y) hız kestirim hatası (m/s)
    - mu_osc   : ortalama osilasyon-modu olasılığı (y'de salınım → ~1 olmalı)

Amaç: (a) iniş gürültülü kestirimle hâlâ tutuyor mu, (b) filtre ne kadar iyi.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID
N   = 10

# VecNormalize istatistikleri (yalnız obs normalizasyonu)
stat_venv = make_vec_env(LandingAviary, n_envs=1,
                         env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=False))
vn = VecNormalize.load("reports/vecnormalize.pkl", stat_venv)
vn.training = False

model = SAC.load("reports/sac_landing")

# Rollout için HAM env (use_estimator varsayılan True → obs'ta kestirim)
aviary = LandingAviary(gui=False, act=ACT, obs=ObservationType.KIN)
print(f"use_estimator = {aviary.use_estimator} | meas_std = {aviary.meas_std} m")

g1 = g2 = g3 = succ = 0
all_pos_rmse, all_vel_rmse = [], []

print(f"\n{'ep':>2} | {'min_h':>6} {'min_hz':>6} {'min_z':>6} {'tilt':>5} | "
      f"{'posRMSE':>7} {'velRMSE':>7} {'mu_osc':>6} | {'ikisi':>5} | bitis")
print("-" * 92)

for ep in range(N):
    obs, info = aviary.reset()
    term = trunc = False

    min_h = 1e9; h_at_horiz = h_at_vrel = np.nan
    min_horiz = 1e9; min_z = 1e9; max_tilt = 0.0
    ever_h = ever_horiz = ever_both = False
    pos_sq, vel_sq, mu_osc = [], [], []
    s = aviary._getDroneStateVector(0)

    while not (term or trunc):
        nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = model.predict(nobs, deterministic=True)
        obs, reward, term, trunc, info = aviary.step(action)

        # --- GERÇEK durumdan iniş metrikleri ---
        r_rel, v_rel = aviary._relative_state()
        s = aviary._getDroneStateVector(0)
        h     = max(0.0, -float(r_rel[2]))
        horiz = float(np.linalg.norm(r_rel[0:2]))
        vrel  = float(np.linalg.norm(v_rel))
        max_tilt = max(max_tilt, abs(float(s[7])), abs(float(s[8])))
        if h < min_h: min_h, h_at_horiz, h_at_vrel = h, horiz, vrel
        min_horiz = min(min_horiz, horiz); min_z = min(min_z, float(s[2]))
        if h < 0.05: ever_h = True
        if horiz < 0.3: ever_horiz = True
        if h < 0.05 and horiz < 0.3: ever_both = True

        # --- IMM-EKF kestirim hatası (platform x,y) ---
        t = aviary.step_counter / aviary.PYB_FREQ
        plat_true  = np.array([aviary.plat_drift[0]*t,
                               aviary.plat_amp*np.sin(2*np.pi*aviary.plat_freq*t)])
        platv_true = np.array([aviary.plat_drift[0],
                               aviary.plat_amp*2*np.pi*aviary.plat_freq*np.cos(2*np.pi*aviary.plat_freq*t)])
        xe = np.ravel(aviary._imm.imm.x)     # [px, vx, py, vy] (filterpy (4,1) → (4,))
        pos_sq.append(np.sum((np.array([xe[0], xe[2]]) - plat_true)**2))
        vel_sq.append(np.sum((np.array([xe[1], xe[3]]) - platv_true)**2))
        mu_osc.append(float(aviary._imm.imm.mu[1]))   # mod 2 = harmonik/osilasyon

    is_succ = bool(info.get("is_success"))
    roll, pitch = float(s[7]), float(s[8])
    sz = float(s[2]); sim_t = aviary.step_counter / aviary.PYB_FREQ
    if term:
        if ever_both and is_succ:                      end = "BASARI"
        elif min_horiz < 0.3 and min_h < 0.05:         end = "temas-HIZLI (vrel>0.5)"
        elif sz < aviary.PLATFORM_TOP - 0.05:          end = f"yere-dustu (z={sz:.2f})"
        elif abs(roll) > 1.2 or abs(pitch) > 1.2:      end = f"DEVRILME (r={roll:+.2f} p={pitch:+.2f})"
        else:                                          end = f"term-? (r={roll:+.2f} p={pitch:+.2f})"
    elif trunc:
        if sim_t > aviary.EPISODE_LEN_SEC:             end = "gercek-timeout"
        else:                                          end = "ARENA-DISI"
    else:                                              end = "bitmedi?"

    pos_rmse = float(np.sqrt(np.mean(pos_sq)));  vel_rmse = float(np.sqrt(np.mean(vel_sq)))
    all_pos_rmse.append(pos_rmse); all_vel_rmse.append(vel_rmse)
    g1 += ever_h; g2 += ever_horiz; g3 += ever_both; succ += is_succ

    print(f"{ep:>2} | {min_h:6.3f} {min_horiz:6.3f} {min_z:6.3f} {max_tilt:5.2f} | "
          f"{pos_rmse:7.3f} {vel_rmse:7.3f} {np.mean(mu_osc):6.2f} | "
          f"{str(ever_both):>5} | {end}")

print("-" * 92)
print(f"ÖZET ({N} episode):")
print(f"  BAŞARI (iniş)          : {succ}/{N}")
print(f"  İkisi aynı anda (h&hz) : {g3}/{N}")
print(f"  Ort. konum RMSE        : {np.mean(all_pos_rmse):.3f} m   (eşik horiz<0.3 ile kıyasla)")
print(f"  Ort. hız RMSE          : {np.mean(all_vel_rmse):.3f} m/s")
aviary.close()
