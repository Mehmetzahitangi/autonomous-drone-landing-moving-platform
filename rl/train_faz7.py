"""
train_faz7.py — Faz 7 residual eğitimi (ACIK_NOTLAR N9).

faz4c'den farkı: politika, Faz 7'de dağıtılan TABANIN İÇİNDE eğitilir:
  * "önce hizalan, sonra in" kapısı açık (align_cfg={}) → politika HİZALAN fazını da görür
  * PID durum taşıma hatası düzeltilmiş ortam (her bölüm temiz PID)
  * bölüm 12 s (güdüm çizelgesi T_LAND=8 ayrı)
Sensör: Gauss (gerçek platform + N(0, 3 cm), her adım) — render yok, hızlı, Faz 4'te kararlı.
Politika sensörü değil IMM kestirimini görür; değerlendirme gerçekçi KAMERADA yapılır (eval_faz7.py).

Kullanım (proje kökünden):
  python rl/train_faz7.py --cbf off --out reports/faz7_res
  python rl/train_faz7.py --cbf on  --out reports/faz7_res_cbf
Sonra:
  python rl/eval_faz7.py --tag f7_res --model faz7 --cam realistic --ep-len 12 --env '{"align_cfg": {}}'
"""
import sys, pathlib, argparse, json, time
PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbf", choices=["on", "off"], required=True, help="CBF kalkanı eğitim döngüsünde")
    ap.add_argument("--out", required=True, help="çıktı klasörü (proje köküne göreli ya da mutlak)")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = PROJ / out
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(obs=ObservationType.KIN, act=ActionType.PID,
                      use_cbf=(args.cbf == "on"), align_cfg={}, episode_len_sec=12.0)
    cfg = dict(args=vars(args), env_kwargs={k: str(v) for k, v in env_kwargs.items()},
               sensor="gauss (use_aruco=False)", algo="SAC", buffer_size=200_000, batch_size=256,
               ent_coef="auto", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("KONFIG:", json.dumps(cfg))

    train_env = make_vec_env(LandingAviary, n_envs=1, seed=args.seed, env_kwargs=env_kwargs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # EvalCallback normalizasyon istatistiklerini train_env'den otomatik eşitler (sync_envs_normalization)
    eval_env = make_vec_env(LandingAviary, n_envs=1, seed=args.seed + 1, env_kwargs=env_kwargs)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = SAC("MlpPolicy", train_env, verbose=1, device="cuda", seed=args.seed,
                buffer_size=200_000, batch_size=256, ent_coef="auto",
                tensorboard_log=str(PROJ / "reports/tb"))

    callbacks = CallbackList([
        EvalCallback(eval_env, best_model_save_path=str(out / "sac_best"), log_path=str(out / "eval"),
                     eval_freq=5000, n_eval_episodes=10, deterministic=True),
        CheckpointCallback(save_freq=50_000, save_path=str(out / "checkpoints"), name_prefix="sac",
                           save_vecnormalize=True),
    ])
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name=out.name)

    model.save(str(out / "sac_landing"))
    train_env.save(str(out / "vecnormalize.pkl"))   # ŞART — normalizasyon istatistikleri
    print(f"Egitim bitti -> {out / 'sac_landing.zip'}")


if __name__ == "__main__":
    main()
