# scripts/train_sac.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np

ACT = ActionType.PID   # RL için waypoint kontrolü (residual yoluna da uyar)

env_kwargs=dict(obs=ObservationType.KIN, act=ACT, initial_xyzs=np.array([[0.0, 0.0, 0.8]]))

# Eğitim ve değerlendirme ortamları
train_env = make_vec_env(LandingAviary, n_envs=1, seed=0,
                         env_kwargs=dict(obs=ObservationType.KIN, act=ACT))

train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

# eval_env → EvalCallback için. SB3, eğitim sırasında politikayı periyodik olarak bu env’de dener: ilerlemeyi ölçer (eval/mean_reward) ve en iyi modeli kaydeder (sac_best/). Bu eğitimimin bir parçası —  Monitoring ve checkpointing için. 
eval_env = make_vec_env(LandingAviary, n_envs=1, seed=1,env_kwargs=dict(obs=ObservationType.KIN, act=ACT))

eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,clip_obs=10.0, training=False)



# SAC modeli (sadece buffer'ı küçülttük)
model = SAC("MlpPolicy", train_env, verbose=1, device="cuda",
            buffer_size=200_000, batch_size=256, ent_coef="auto",
            tensorboard_log="reports/tb/")

eval_cb = EvalCallback(eval_env, best_model_save_path="reports/sac_best/",
                       log_path="reports/sac_eval/", eval_freq=5000,
                       n_eval_episodes=5, deterministic=True)

model.learn(total_timesteps=300_000, callback=eval_cb)  
model.save("reports/sac_landing")
train_env.save("reports/vecnormalize.pkl")   # ŞART — normalizasyon istatistikleri

print("Eğitim Bitti -> reports/sac_landing.zip")