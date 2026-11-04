"""
diag_cbf.py — CBF kalkanı OFF vs ON karşılaştırması (aynı model, EŞLİ platformlar).

Her bölümde `np.random.seed(s)` ile platform randomizasyonu sabitlenir; böylece OFF ve ON
koşuları AYNI 10..N platformu görür → adil (paired) kıyas. Planın "CBF'siz vs CBF'li" tablosu.

Ölçülen (her mod):
  - BAŞARI (env is_success) ve temas hızı (fin_vrel) dağılımı
  - vrel>0.5 (hızlı-temas) bölüm sayısı
  - ON modunda: kalkanın adımların yüzde kaçında müdahale ettiği + ortalama Δz
Rollout HAM LandingAviary üzerinde (VecNormalize yalnız obs normalizasyonu).
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
N   = 50
SEEDS = list(range(N))

stat_venv = make_vec_env(LandingAviary, n_envs=1,
                         env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=False))
vn = VecNormalize.load("reports/vecnormalize.pkl", stat_venv)
vn.training = False
model = SAC.load("reports/sac_landing")

aviary = LandingAviary(gui=False, act=ACT, obs=ObservationType.KIN)


def run(use_cbf):
    succ = 0; vrels = []; fast = 0
    interv_frac = []; interv_dz = []
    for s in SEEDS:
        np.random.seed(s)               # platformu sabitle (OFF ve ON ayni s'te ayni yorunge)
        obs, info = aviary.reset()
        aviary.use_cbf = use_cbf
        term = trunc = False
        steps = 0; interv = 0; dz_sum = 0.0
        fin_vrel = np.nan
        while not (term or trunc):
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(nobs, deterministic=True)
            obs, r, term, trunc, info = aviary.step(action)
            _, v_rel = aviary._relative_state()
            fin_vrel = float(np.linalg.norm(v_rel))
            steps += 1
            if use_cbf and getattr(aviary, "_dbg_cbf", None) is not None:
                if aviary._dbg_cbf["intervened"]:
                    interv += 1
        is_succ = bool(info.get("is_success"))
        succ += is_succ; vrels.append(fin_vrel); fast += (fin_vrel > 0.5)
        if use_cbf:
            interv_frac.append(100.0 * interv / max(steps, 1))
    return dict(succ=succ, vrels=np.array(vrels), fast=fast,
                interv_frac=np.array(interv_frac) if interv_frac else None)


print(f"Model: reports/sac_landing | N={N} eşli bölüm (aynı platformlar)\n")
off = run(False)
on  = run(True)

print(f"{'metrik':<30} {'CBF OFF':>10} {'CBF ON':>10}")
print("-" * 52)
print(f"{'BAŞARI (is_success)':<30} {off['succ']:>8}/{N} {on['succ']:>8}/{N}")
print(f"{'vrel>0.5 (hızlı-temas)':<30} {off['fast']:>10} {on['fast']:>10}")
print(f"{'ort. temas hızı vrel':<30} {off['vrels'].mean():>10.3f} {on['vrels'].mean():>10.3f}")
print(f"{'maks temas hızı vrel':<30} {off['vrels'].max():>10.3f} {on['vrels'].max():>10.3f}")
if on['interv_frac'] is not None:
    print(f"{'kalkan müdahale (adım %)':<30} {'-':>10} {on['interv_frac'].mean():>9.1f}%")

print("\n--- EŞLİ bölüm bazında temas hızı (vrel) ---")
print(f"{'seed':>4} {'OFF vrel':>9} {'ON vrel':>9}  {'fark':>7}")
for i, s in enumerate(SEEDS):
    d = on['vrels'][i] - off['vrels'][i]
    flag = "  <-- OFF hızlı" if off['vrels'][i] > 0.5 else ""
    print(f"{s:>4} {off['vrels'][i]:>9.3f} {on['vrels'][i]:>9.3f}  {d:>+7.3f}{flag}")

aviary.close()
