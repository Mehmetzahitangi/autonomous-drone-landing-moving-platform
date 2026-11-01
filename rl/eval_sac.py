"""
eval_sac.py — eğitilmiş SAC politikasını değerlendirir / izler.

Kullanım:
  python rl/eval_sac.py                 # GUI'de 5 bölüm izle (varsayılan)
  python rl/eval_sac.py --n 3           # 3 bölüm
  python rl/eval_sac.py --no-gui        # headless, hızlı (sayı için)
  python rl/eval_sac.py --no-gui --n 30 # headless güvenilir başarı oranı
  python rl/eval_sac.py --speed 0       # beklemesiz (hızlı izleme)

Not: GUI gerçek-zamanlı render ağırdır; çok bölüm + eğitim/TensorBoard açıkken
bilgisayarı kasar. Sayı için --no-gui kullan; GUI'yi yalnız birkaç bölüm izlemek için aç.
Her durumda (Ctrl+C / hata dahil) env.close() çağrılır → asılı GUI bağlantısı kalmaz.
"""
import sys, pathlib, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5, help="bölüm sayısı")
ap.add_argument("--gui", dest="gui", action="store_true", default=True, help="GUI aç (varsayılan)")
ap.add_argument("--no-gui", dest="gui", action="store_false", help="headless (hızlı)")
ap.add_argument("--speed", type=float, default=1.0, help="gerçek-zaman bekleme çarpanı; 0 = beklemesiz")
ap.add_argument("--model", type=str, default="reports/sac_landing")
ap.add_argument("--vecnorm", type=str, default="reports/vecnormalize.pkl")
args = ap.parse_args()

env = make_vec_env(LandingAviary, n_envs=1, env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=args.gui))
env = VecNormalize.load(args.vecnorm, env)
env.training = False
env.norm_reward = False
model = SAC.load(args.model, env=env)

sleep_dt = (1.0/30.0) * args.speed if args.gui else 0.0   # ctrl_freq=30
successes, speeds = 0, []
try:
    for ep in range(args.n):
        obs = env.reset()                 # VecEnv: sadece obs döner
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)   # VecEnv: 4'lü, info bir liste
            if sleep_dt:
                time.sleep(sleep_dt)
        info0 = info[0]
        ok = bool(info0.get("is_success"))
        successes += ok
        speeds.append(info0.get("touchdown_speed", np.nan))
        print(f"  ep{ep:>2}: {'BASARI' if ok else 'basarisiz'}  temas_hiz={info0.get('touchdown_speed', float('nan')):.3f} m/s")
    print(f"\nBaşarı oranı : {successes}/{args.n} = {100*successes/max(args.n,1):.0f}%")
    print(f"Ort. temas hızı: {np.nanmean(speeds):.3f} m/s (yumuşaklık)")
finally:
    env.close()   # Ctrl+C / hata dahil her durumda kapat → asılı GUI bağlantısı kalmaz
