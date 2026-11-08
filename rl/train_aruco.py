import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID
os.makedirs("reports/faz7_aruco", exist_ok=True)   # faz4c'yi EZMEZ

# SIFIRDAN egitim: taze kritik ArUco deger fonksiyonunu dogru ogrenir (warm-start iraksiyordu).
# use_aruco=True -> olcum kameradan (dropout'lu) + env'deki gorunurluk odulu aktif.
train_env = make_vec_env(LandingAviary, n_envs=1, seed=0,
                         env_kwargs=dict(obs=ObservationType.KIN, act=ACT, use_aruco=True))
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                         clip_obs=10.0, clip_reward=10.0)

eval_env = make_vec_env(LandingAviary, n_envs=1, seed=1,
                        env_kwargs=dict(obs=ObservationType.KIN, act=ACT, use_aruco=True))
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                        clip_obs=10.0, training=False)

# Taze SAC (warm-start YOK): tam kesif, kritik bastan ogrenir
model = SAC("MlpPolicy", train_env, verbose=1, device="cuda",
            buffer_size=200_000, batch_size=256, ent_coef="auto",
            tensorboard_log="reports/faz7_aruco/tb/")

eval_cb = EvalCallback(eval_env, best_model_save_path="reports/faz7_aruco/best/",
                       log_path="reports/faz7_aruco/eval/", eval_freq=5000,
                       n_eval_episodes=10, deterministic=True)

model.learn(total_timesteps=300_000, callback=eval_cb)

model.save("reports/faz7_aruco/sac_aruco")
train_env.save("reports/faz7_aruco/vecnorm_aruco.pkl")
print("ArUco (sifirdan) egitim bitti -> reports/faz7_aruco/sac_aruco.zip")