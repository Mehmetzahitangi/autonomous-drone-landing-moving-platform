import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID
OUT = "reports/faz7_curriculum"
os.makedirs(OUT, exist_ok=True)

# Ayni gorsel-algi (use_aruco=True) obs'u; tek degisen baslangic zorlugu (curriculum)
train_env = make_vec_env(LandingAviary, n_envs=1, seed=0,
                         env_kwargs=dict(obs=ObservationType.KIN, act=ACT, use_aruco=True))
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

eval_env = make_vec_env(LandingAviary, n_envs=1, seed=1,
                        env_kwargs=dict(obs=ObservationType.KIN, act=ACT, use_aruco=True))
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

# Taze SAC + surdurulen kesif (erken entropi cokusunu engelle)
model = SAC("MlpPolicy", train_env, verbose=1, device="cuda",
            buffer_size=200_000, batch_size=256, ent_coef="auto",
            tensorboard_log=f"{OUT}/tb/")
model.target_entropy = -1.0

# CURRICULUM: (yatay baslangic ofset siniri, adim sayisi) — kolaydan zora
STAGES = [(0.15,  80_000),    # Asama 1: yakin/merkezi -> marker hep gorunur, dropout dusuk
          (0.30, 100_000),    # Asama 2: orta
          (0.50, 150_000)]    # Asama 3: tam/gercekci (su anki zorluk)

first = True
for i, (xy, steps) in enumerate(STAGES, 1):
    train_env.env_method("set_init_difficulty", xy_max=xy)
    eval_env.env_method("set_init_difficulty", xy_max=xy)
    eval_cb = EvalCallback(eval_env, best_model_save_path=f"{OUT}/best_s{i}/",
                           log_path=f"{OUT}/eval_s{i}/", eval_freq=5000,
                           n_eval_episodes=10, deterministic=True)
    print(f"\n===== ASAMA {i}: xy_max={xy} m, {steps} adim =====")
    model.learn(total_timesteps=steps, reset_num_timesteps=first, callback=eval_cb)
    first = False
    model.save(f"{OUT}/sac_s{i}")
    train_env.save(f"{OUT}/vecnorm_s{i}.pkl")
    print(f"Asama {i} bitti -> {OUT}/sac_s{i}.zip")

print("Curriculum bitti -> reports/faz7_curriculum/sac_s3.zip")