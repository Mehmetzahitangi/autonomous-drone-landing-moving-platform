import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


class ProtEnv(gym.Env):          # protagonist eğitilirken: adversary DONUK, rüzgârı o seçer
    def __init__(self, av, adv, adv_vn):
        self.av, self.adv, self.adv_vn = av, adv, adv_vn
        self.observation_space = av.observation_space
        self.action_space      = av.action_space
    def reset(self, *, seed=None, options=None):
        self._obs, info = self.av.reset(seed=seed, options=options)
        return self._obs, info
    def step(self, a_prot):
        if self.adv is not None:
            na = self.adv_vn.normalize_obs(self._obs.astype(np.float32))
            a_adv, _ = self.adv.predict(na, deterministic=False)
            self.av.set_wind(a_adv)
        else:
            self.av.set_wind(np.zeros(2))
        self._obs, r, term, trunc, info = self.av.step(a_prot)
        return self._obs, r, term, trunc, info


class AdvEnv(gym.Env):           # adversary eğitilirken: protagonist DONUK, ödül NEGATİF
    def __init__(self, av, prot, prot_vn):
        self.av, self.prot, self.prot_vn = av, prot, prot_vn
        self.observation_space = av.observation_space
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    def reset(self, *, seed=None, options=None):
        self._obs, info = self.av.reset(seed=seed, options=options)
        return self._obs, info
    def step(self, a_adv):
        self.av.set_wind(a_adv)
        npobs = self.prot_vn.normalize_obs(self._obs.astype(np.float32))
        a_prot, _ = self.prot.predict(npobs, deterministic=False)
        self._obs, r, term, trunc, info = self.av.step(a_prot)
        return self._obs, -r, term, trunc, info          # ← negatif ödül


ACT = ActionType.PID
av = LandingAviary(obs=ObservationType.KIN, act=ACT, use_cbf=False)   # TEK paylaşılan env

prot_core = ProtEnv(av, adv=None, adv_vn=None)
adv_core  = AdvEnv (av, prot=None, prot_vn=None)

# --- protagonist: faz4c'den WARM-START (DR'de çalışan reçete) ---
prot_vn = VecNormalize.load("reports/faz4c_heave/vecnormalize.pkl",
                            DummyVecEnv([lambda: Monitor(prot_core)]))
prot_vn.training = True; prot_vn.norm_reward = True
prot = SAC.load("reports/faz4c_heave/sac_landing", env=prot_vn, device="cuda")
prot.lr_schedule = lambda _: 1e-4; prot.learning_rate = 1e-4   # nazik ince ayar

# --- adversary: FRESH ---
adv_vn = VecNormalize(DummyVecEnv([lambda: Monitor(adv_core)]),
                      norm_obs=True, norm_reward=True, clip_obs=10.0)
adv = SAC("MlpPolicy", adv_vn, device="cuda", buffer_size=200_000,
          batch_size=256, ent_coef="auto", verbose=1)

# --- karşı tarafı bağla (ŞART — yoksa adversary fazı çöker) ---
prot_core.adv,  prot_core.adv_vn  = adv,  adv_vn
adv_core.prot,  adv_core.prot_vn  = prot, prot_vn

# --- alternating döngü + rüzgâr curriculum (DR'nin göremediği 0.06-0.08'i kapsar) ---
ROUNDS, N_P, N_A = 6, 25_000, 10_000
WIND_SCHED = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]   # tur başına WIND_MAX (N)

for rnd in range(ROUNDS):
    av.WIND_MAX = WIND_SCHED[rnd]
    print(f"\n=== TUR {rnd+1}/{ROUNDS} | WIND_MAX={av.WIND_MAX:.2f} N: protagonist ===")
    prot._last_obs = None
    prot.learn(N_P, reset_num_timesteps=False)
    print(f"=== TUR {rnd+1}/{ROUNDS}: adversary ===")
    adv._last_obs = None
    adv.learn(N_A, reset_num_timesteps=False)

prot.save("reports/sac_rarl")
prot_vn.save("reports/vecnorm_rarl.pkl")   # SADECE protagonist dağıtılır
print("RARL bitti -> reports/sac_rarl.zip")