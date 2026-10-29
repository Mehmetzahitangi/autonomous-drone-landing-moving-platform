"""
diag_landing.py — "success_rate=0 ama reward yükseldi" teşhisi (v2: ham env).

ÖNEMLİ: Rollout'u HAM LandingAviary üzerinde yapar (VecEnv DEĞİL). Böylece
episode bitince otomatik-reset OLMAZ ve terminal durumu (roll/pitch/z) ile
terminated/truncated bayraklarını DOĞRU okuruz. VecNormalize yalnızca obs'u
politikanın beklediği gibi normalize etmek için kullanılır (step edilmez).

HER episode için:
  - ulaşılan min h, o andaki horiz/vrel
  - min horiz, min drone_z (fiziksel alçalma)
  - episode boyunca görülen max |roll|/|pitch| (tilt)  — terminal adım DAHİL
  - her başarı kapısı ayrı ayrı (h<0.05 / horiz<0.3 / ikisi birden)
  - KESİN bitiş nedeni: terminated mı truncated mı, hangi koşul
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

# 1) VecNormalize istatistikleri — yalnız obs normalizasyonu için (step edilmez)
stat_venv = make_vec_env(LandingAviary, n_envs=1,
                         env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=False))
vn = VecNormalize.load("reports/vecnormalize.pkl", stat_venv)
vn.training = False

# 2) Politika (predict için env'e gerek yok)
model = SAC.load("reports/sac_landing")

# 3) Rollout için HAM env — auto-reset YOK
aviary = LandingAviary(gui=False, act=ACT, obs=ObservationType.KIN)

g1_cnt = g2_cnt = g3_cnt = succ_cnt = 0

print(f"{'ep':>2} | {'min_h':>6} {'@horiz':>6} {'@vrel':>6} | {'min_horiz':>9} {'min_z':>6} "
      f"| {'tilt':>5} | {'h<.05':>5} {'hz<.3':>5} {'ikisi':>5} {'vrel@':>6} | bitis")
print("-" * 104)

for ep in range(N):
    obs, info = aviary.reset()
    term = trunc = False

    min_h = 1e9;  h_at_horiz = h_at_vrel = np.nan
    min_horiz = 1e9
    min_z = 1e9
    ever_h = ever_horiz = ever_both = False
    vrel_at_both = np.nan
    max_tilt = 0.0
    steps = 0
    s = aviary._getDroneStateVector(0)

    while not (term or trunc):
        nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = model.predict(nobs, deterministic=True)
        obs, reward, term, trunc, info = aviary.step(action)
        steps += 1

        r_rel, v_rel = aviary._relative_state()
        s = aviary._getDroneStateVector(0)      # auto-reset YOK -> gerçek terminal durum
        drone_z = float(s[2])
        h     = max(0.0, -float(r_rel[2]))
        horiz = float(np.linalg.norm(r_rel[0:2]))
        vrel  = float(np.linalg.norm(v_rel))
        max_tilt = max(max_tilt, abs(float(s[7])), abs(float(s[8])))

        if h < min_h:
            min_h, h_at_horiz, h_at_vrel = h, horiz, vrel
        min_horiz = min(min_horiz, horiz)
        min_z     = min(min_z, drone_z)

        if h < 0.05:                      ever_h = True
        if horiz < 0.3:                   ever_horiz = True
        if h < 0.05 and horiz < 0.3:
            ever_both = True
            if np.isnan(vrel_at_both) or vrel < vrel_at_both:
                vrel_at_both = vrel

    is_succ = bool(info.get("is_success"))

    # KESİN bitiş nedeni — doğrudan term/trunc + gerçek terminal durum
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    roll, pitch = float(s[7]), float(s[8])
    sim_t = aviary.step_counter / aviary.PYB_FREQ
    if term:
        if ever_both and is_succ:
            end = "BASARI"
        elif min_horiz < 0.3 and min_h < 0.05:
            end = f"temas-HIZLI (vrel>0.5)"
        elif sz < aviary.PLATFORM_TOP - 0.05:
            end = f"yere-dustu (z={sz:.2f})"
        elif abs(roll) > 0.4 or abs(pitch) > 0.4:
            end = f"EGIM>0.4 (r={roll:+.2f} p={pitch:+.2f})"
        else:
            end = f"term-? (r={roll:+.2f} p={pitch:+.2f} z={sz:.2f})"
    elif trunc:
        if sim_t > aviary.EPISODE_LEN_SEC:
            end = f"gercek-timeout({steps}ad)"
        elif abs(sx) > 2.0 or abs(sy) > 2.0 or sz > 3.0:
            end = f"ARENA-DISI (x={sx:+.1f} y={sy:+.1f} z={sz:.1f})"
        else:
            end = "trunc-?"
    else:
        end = "bitmedi?"

    g1_cnt += ever_h
    g2_cnt += ever_horiz
    g3_cnt += ever_both
    succ_cnt += is_succ

    print(f"{ep:>2} | {min_h:6.3f} {h_at_horiz:6.2f} {h_at_vrel:6.2f} | "
          f"{min_horiz:9.3f} {min_z:6.3f} | {max_tilt:5.2f} | "
          f"{str(ever_h):>5} {str(ever_horiz):>5} {str(ever_both):>5} "
          f"{vrel_at_both:6.2f} | {end}")

print("-" * 104)
print(f"ÖZET ({N} episode):")
print(f"  h<0.05 en az bir kez   : {g1_cnt}/{N}")
print(f"  horiz<0.3 en az bir kez : {g2_cnt}/{N}")
print(f"  İKİSİ AYNI ANDA        : {g3_cnt}/{N}")
print(f"  BAŞARI (3 kapı+vrel)   : {succ_cnt}/{N}")
aviary.close()
