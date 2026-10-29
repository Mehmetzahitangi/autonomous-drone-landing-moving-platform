import sys, pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np

#from rl.train_sac import ACT , bu satır açılırsa train tekrar tetiklenir. BUNA DİKKAT (ACT aşağıda tanımlı)

ACT = ActionType.PID
env_kwargs=dict(obs=ObservationType.KIN, act=ACT, initial_xyzs=np.array([[0.0, 0.0, 0.8]]))

env = make_vec_env(LandingAviary, n_envs=1, env_kwargs=dict(act=ACT, gui=True))
env = VecNormalize.load("reports/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False
model = SAC.load("reports/sac_landing", env=env)


N = 20
successes, speeds = 0, []
for ep in range(N):
    obs = env.reset()                 # VecEnv: sadece obs döner
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)   # VecEnv: 4'lü, info bir liste
        time.sleep(1/60)      # ctrl_freq=30  gerçek zamanlı
    info0 = info[0]
    if info0.get("is_success"):
        successes += 1
    speeds.append(info0.get("touchdown_speed", np.nan))

print(f"Başarı oranı : {successes}/{N} = {100*successes/N:.0f}%")
print(f"Ort. temas hızı: {np.nanmean(speeds):.3f} m/s (yumuşaklık)")
env.close()