"""diag_term.py — bölümler neden ~3 adımda bitiyor, hangi koşul öldürüyor?

faz4c (deterministic) + RANDOM aksiyon, dr_wind açık/kapalı kombinasyonlarında
ep_len ve ÖLÜM SEBEBİ (başarı / hızlı-çarpma / yere-çakılma / eğim-kill / timeout).
İlk bölümün ilk adımlarını da satır satır basar (eğim nasıl büyüyor görelim).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID


def term_reason(av):
    s = av._getDroneStateVector(0)
    r_rel, v_rel = av._relative_state()
    h = max(0, -r_rel[2]); horiz = np.linalg.norm(r_rel[0:2]); vmag = np.linalg.norm(v_rel)
    if horiz < 0.3 and h < 0.05 and vmag < 0.5:  return "BASARI"
    if horiz < 0.3 and h < 0.05 and vmag >= 0.5: return f"crash-hizli(vrel={vmag:.2f})"
    if s[2] < av.PLATFORM_TOP - 0.05:            return f"yere-cakilma(z={s[2]:.3f})"
    if abs(s[7]) > 1.2 or abs(s[8]) > 1.2:       return f"egim-kill(roll={s[7]:.2f},pitch={s[8]:.2f})"
    return "timeout/bilinmiyor"


def run(av, policy, vn, n=5):
    lens = []
    for ep in range(n):
        obs, info = av.reset()
        term = trunc = False; k = 0
        while not (term or trunc):
            if policy == "random":
                a = av.action_space.sample()
            else:
                nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
                a, _ = policy.predict(nobs, deterministic=True)
            obs, r, term, trunc, info = av.step(a); k += 1
            if ep == 0 and k <= 6:
                s = av._getDroneStateVector(0)
                print(f"      adim{k}: z={s[2]:.3f}  roll={s[7]:+.2f}  pitch={s[8]:+.2f}")
        lens.append(k)
        print(f"   ep{ep}: len={k:3d}  ->  {'TIMEOUT' if trunc else term_reason(av)}")
    print(f"   >> ort ep_len = {np.mean(lens):.1f}")


stat = make_vec_env(LandingAviary, n_envs=1, env_kwargs=dict(obs=ObservationType.KIN, act=ACT))
vn = VecNormalize.load("reports/faz4c_heave/vecnormalize.pkl", stat); vn.training = False
model = SAC.load("reports/faz4c_heave/sac_landing", device="cpu")

for drw in [False, True]:
    av = LandingAviary(obs=ObservationType.KIN, act=ACT, dr_wind=drw)
    print(f"\n=== dr_wind={drw} | faz4c (deterministic) ===")
    run(av, model, vn, n=5)
    print(f"=== dr_wind={drw} | RANDOM aksiyon (fresh benzeri) ===")
    run(av, "random", vn, n=5)
    av.close()
