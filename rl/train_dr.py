import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID

# DR rüzgarlı eğitim ortamı
train_env = make_vec_env(LandingAviary, n_envs=1, seed=0,
                         env_kwargs=dict(obs=ObservationType.KIN, act=ACT,
                                         use_cbf=False, dr_wind=True))
# obs/return istatistiklerini faz4c'den YÜKLE (warm-start uyumu) — fresh DEĞİL
train_env = VecNormalize.load("reports/faz4c_heave/vecnormalize.pkl", train_env)
train_env.training = True
train_env.norm_reward = True

eval_env = make_vec_env(LandingAviary, n_envs=1, seed=1,
                        env_kwargs=dict(obs=ObservationType.KIN, act=ACT,
                                        use_cbf=False, dr_wind=True))
eval_env = VecNormalize.load("reports/faz4c_heave/vecnormalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

# faz4c AĞIRLIKLARINDAN warm-start + nazik ince ayar (rastgele warmup YOK)
model = SAC.load("reports/faz4c_heave/sac_landing", env=train_env, device="cuda")
model.lr_schedule   = lambda _: 1e-4     # düşük lr: iyi politikayı bozma
model.learning_rate = 1e-4

eval_cb = EvalCallback(eval_env, best_model_save_path="reports/sac_dr_best/",
                       log_path="reports/sac_dr_eval/", eval_freq=5000,
                       n_eval_episodes=10, deterministic=True)

# reset_num_timesteps=False → sayaç yüksek kalır → rastgele warmup TETİKLENMEZ,
# politika ilk adımdan faz4c ağırlıklarıyla iner (obs kirlenmesi olmaz)
model.learn(total_timesteps=150_000, reset_num_timesteps=False, callback=eval_cb)

model.save("reports/sac_dr")
train_env.save("reports/vecnorm_dr.pkl")
print("DR (warm-start) bitti -> reports/sac_dr.zip")