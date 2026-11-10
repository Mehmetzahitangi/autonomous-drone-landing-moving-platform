"""diag_gimbal.py — ASIL darbogaz algi dropout'u mu? Gimbal (nadir-stabilize) kamera testi.

Uretim kodunu DEGISTIRMEDEN, monkeypatch ile:
  * target_vel ileri-beslemeyi NOTRLER (Adim 2'yi kapatir; kullanicinin dosya durumundan bagimsiz).
  * ArucoCamera'yi gimbal=<flag> ile kurar.
Iki konfig, AYNI seed'ler (0..n-1), faz4c, aruco realistic:
  A) FF kapali + gimbal KAPALI   -> baz'i (~%52) dogrulamali (patch saglamasi)
  B) FF kapali + gimbal ACIK     -> deney: tespit orani ve basari artiyor mu?
"""
import sys, pathlib
PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
import numpy as np
from stable_baselines3 import SAC

import env.landing_aviary as la
import gym_pybullet_drones.control.DSLPIDControl as Dmod
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from rl.eval_faz7 import outcome, load_vecnorm, wilson

# --- Adim 2'yi notrle: computeControl'a hangi target_vel gelirse gelsin 0 yap ---
_orig_cc = Dmod.DSLPIDControl.computeControl
def _cc_noff(self, *a, **k):
    k["target_vel"] = np.zeros(3)
    return _orig_cc(self, *a, **k)
Dmod.DSLPIDControl.computeControl = _cc_noff

# --- gimbal'i global bayrakla zorla ---
_G = {"on": False}
_OrigCam = la.ArucoCamera
class _GCam(_OrigCam):
    def __init__(self, *a, **k):
        k["gimbal"] = _G["on"]
        super().__init__(*a, **k)
la.ArucoCamera = _GCam

MPATH = "reports/faz4c_heave/sac_landing.zip"
VPATH = "reports/faz4c_heave/vecnormalize.pkl"


def run(label, gimbal, n=40, seed0=0, cam="realistic"):
    _G["on"] = gimbal
    model = SAC.load(str(PROJ / MPATH))
    vn = load_vecnorm(PROJ / VPATH)
    av = la.LandingAviary(gui=False, act=ActionType.PID, obs=ObservationType.KIN,
                          use_aruco=True, use_cbf=False, cam_cfg=cam)
    cnt, dets, stales = {}, [], []
    print(f"\n=== {label} (gimbal={gimbal}) ===")
    for i in range(n):
        seed = seed0 + i
        np.random.seed(seed)
        obs, info = av.reset()
        term = trunc = False; max_stale = 0
        while not (term or trunc):
            a, _ = model.predict(vn.normalize_obs(np.asarray(obs, dtype=np.float32)), deterministic=True)
            obs, r, term, trunc, info = av.step(a)
            max_stale = max(max_stale, info.get("stale", 0))
        res = outcome(av, info, term, trunc)
        cnt[res] = cnt.get(res, 0) + 1
        dets.append(info.get("cam_dets", 1) / max(info.get("cam_frames", 1), 1))
        stales.append(max_stale)
    av.close()
    k_ok = cnt.get("BASARI", 0)
    lo, hi = wilson(k_ok, n)
    print(f"{label}: BASARI {k_ok}/{n} = {100*k_ok/n:.0f}%  (%95 GA {100*lo:.0f}-{100*hi:.0f})")
    print(f"  sonuclar: {cnt}")
    print(f"  tespit orani ort: {100*np.mean(dets):.1f}%   maxBayat medyan: {np.median(stales):.0f} adim")
    return k_ok, n


if __name__ == "__main__":
    import sys as _s
    if "--final" in _s.argv:
        run("FINAL gimbal-ACIK n=100", gimbal=True, n=100)
    elif "--c-only" in _s.argv:
        run("C) gimbal-ACIK + kamera 30Hz (throttle kalk)", gimbal=True,
            cam={"base": "realistic", "hz": 30.0})
    else:
        run("A) FF-kapali gimbal-KAPALI", gimbal=False)
        run("B) FF-kapali gimbal-ACIK",  gimbal=True)
