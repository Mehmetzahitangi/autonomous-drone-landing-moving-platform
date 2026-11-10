"""numpy ile SAC aktörü (deterministik aksiyon). WSL'de torch/SB3 gerektirmez — yalnız numpy."""
import numpy as np


class PolicyNP:
    def __init__(self, npz_path):
        p = np.load(npz_path)
        self.p = {k: p[k] for k in p.files}

    def normalize(self, obs):
        p = self.p
        return np.clip((np.asarray(obs, np.float64).ravel() - p["mean"]) / np.sqrt(p["var"] + p["eps"]),
                       -p["clip"], p["clip"])

    def act(self, obs):
        """obs: ham (normalize EDİLMEMİŞ) 63'lü gözlem → aksiyon (3,), [-1, 1]."""
        p, x = self.p, self.normalize(obs)
        h = np.maximum(p["W0"] @ x + p["b0"], 0.0)
        h = np.maximum(p["W1"] @ h + p["b1"], 0.0)
        return np.tanh(p["Wmu"] @ h + p["bmu"])