"""SAC aktörü + VecNormalize istatistiklerini numpy .npz'ye aktarır. Kullanım: python tools/export_policy.py reports/faz7_res"""
import sys, pathlib, pickle
import numpy as np
from stable_baselines3 import SAC

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "reports/faz7_res")
m = SAC.load(str(d / "sac_landing.zip"), device="cpu")
vn = pickle.load(open(d / "vecnormalize.pkl", "rb"))
a = m.policy.actor
t = lambda x: x.detach().cpu().numpy().astype(np.float64)
np.savez(d / "policy_np.npz",
         W0=t(a.latent_pi[0].weight), b0=t(a.latent_pi[0].bias),
         W1=t(a.latent_pi[2].weight), b1=t(a.latent_pi[2].bias),
         Wmu=t(a.mu.weight), bmu=t(a.mu.bias),
         mean=np.asarray(vn.obs_rms.mean, np.float64).ravel(), var=np.asarray(vn.obs_rms.var, np.float64).ravel(),
         clip=np.float64(vn.clip_obs), eps=np.float64(vn.epsilon))
print("yazildi:", d / "policy_np.npz", "| gozlem boyutu:", np.asarray(vn.obs_rms.mean).size)