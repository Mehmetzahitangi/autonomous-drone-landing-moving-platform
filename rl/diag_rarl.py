"""
diag_rarl.py — RARL sağlamlık tanısı: SABİT (held-out) rüzgâr altında başarı ızgarası.

Fikir: Eğitimde adversary ÖĞRENEN rüzgârdı. Burada adil, tekrarlanabilir bir sınav için
rüzgârı DIŞARIDAN sabitleriz: 8 yön × birkaç şiddet. Her hücrede platformlar eşli
(np.random.seed(s) → her model/rüzgâr AYNI yörüngeleri görür). Böylece:
  - Rüzgârsız (0 N): sağlamlık için nominali feda ettik mi? (RARL ≈ nominal olmalı)
  - Rüzgâr arttıkça: RARL başarısı daha yavaş düşmeli — asıl RARL ispatı.
  - EN-KÖTÜ-YÖN başarısı: adversarial sağlamlık worst-case ile ölçülür, ortalamayla değil.

Rüzgâr env'e `aviary._wind` (3B, Newton, WORLD_FRAME) olarak girer; `_physics` her fizik
alt-adımında uygular. Bölüm başında bir kez set edilir, bölüm boyunca sabit kalır
(reset() dışında kimse sıfırlamaz). set_wind'in WIND_MAX ölçeğini ATLAYIP doğrudan
Newton veririz → eğitim bütçesinin ÜstÜnde (ekstrapolasyon) de sınayabiliriz.

ÖN KOŞUL: Faz 5 Adım 1 (env rüzgâr kancası: _physics override + self._wind) yapılmış olmalı.

Kullanım: her modelin (etiket, model.zip, vecnormalize.pkl) üçlüsünü MODELS'e koy.
Dosyası olmayan model atlanır (önce yalnız nominal baseline'ı da koşabilirsin).
"""
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID

# ---- Ayarlar ----
MODELS = [
    ("nominal(faz4c)", "reports/faz4c_heave/sac_landing.zip", "reports/faz4c_heave/vecnormalize.pkl"),
    ("DR",             "reports/sac_dr.zip",                   "reports/vecnorm_dr.pkl"),
    ("RARL",           "reports/sac_rarl.zip",                 "reports/vecnorm_rarl.pkl"),
]
MAGS    = [0.00, 0.02, 0.04, 0.06, 0.08]   # rüzgâr şiddeti (N). Nominal 0.08'de %0'a çöküyor →
                                            # diz noktasını (degradation knee) haritalamak için alçak aralık.
N_DIRS  = 8                       # yön sayısı (0..315°)
N_SEEDS = 15                      # her (yön,şiddet) için eşli platform sayısı
# -----------------

DIRS = [2*np.pi*k/N_DIRS for k in range(N_DIRS)]   # rad


def load(model_path, vn_path):
    stat_venv = make_vec_env(LandingAviary, n_envs=1,
                             env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=False))
    vn = VecNormalize.load(vn_path, stat_venv)
    vn.training = False
    model = SAC.load(model_path)
    return model, vn


def run_cell(aviary, model, vn, wind_vec, seeds):
    """Verilen sabit rüzgârda `seeds` boyunca koş; (başarı_sayısı, vrel_listesi) döndür."""
    succ = 0; vrels = []
    for s in seeds:
        np.random.seed(s)                       # platformu sabitle (eşli)
        obs, info = aviary.reset()
        aviary._wind = np.array(wind_vec, dtype=float)   # bölüm boyu sabit rüzgâr
        term = trunc = False
        fin_vrel = np.nan
        while not (term or trunc):
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(nobs, deterministic=True)
            obs, r, term, trunc, info = aviary.step(action)
            aviary._wind = np.array(wind_vec, dtype=float)   # step içinde değişmediğinden emin ol
            _, v_rel = aviary._relative_state()
            fin_vrel = float(np.linalg.norm(v_rel))
        succ += bool(info.get("is_success"))
        vrels.append(fin_vrel)
    return succ, np.array(vrels)


def eval_model(label, model_path, vn_path):
    if not (os.path.exists(model_path) and os.path.exists(vn_path)):
        print(f"[ATLANDI] {label}: dosya yok ({model_path} / {vn_path})\n")
        return None
    model, vn = load(model_path, vn_path)
    aviary = LandingAviary(gui=False, act=ACT, obs=ObservationType.KIN)
    if not hasattr(aviary, "_wind"):
        print("[HATA] env'de `_wind` yok — önce Faz 5 Adım 1 (rüzgâr kancası) yapılmalı.")
        aviary.close(); return None
    seeds = list(range(N_SEEDS))

    rows = []   # (mag, mean_succ_rate, worst_dir_rate, mean_vrel, max_vrel)
    print(f"Model: {label}")
    print(f"{'ruzgar(N)':>9} {'basari(ort)':>12} {'basari(en-kotu-yon)':>20} {'ort vrel':>9} {'maks vrel':>10}")
    print("-" * 64)
    for mag in MAGS:
        if mag == 0.0:
            succ, vrels = run_cell(aviary, model, vn, [0.0, 0.0, 0.0], seeds)
            rate = 100.0 * succ / N_SEEDS
            row = (mag, rate, rate, float(vrels.mean()), float(vrels.max()))
        else:
            dir_rates = []; all_vrels = []
            for ang in DIRS:
                wv = [mag*np.cos(ang), mag*np.sin(ang), 0.0]
                succ, vrels = run_cell(aviary, model, vn, wv, seeds)
                dir_rates.append(100.0 * succ / N_SEEDS); all_vrels.append(vrels)
            all_vrels = np.concatenate(all_vrels)
            row = (mag, float(np.mean(dir_rates)), float(np.min(dir_rates)),
                   float(all_vrels.mean()), float(all_vrels.max()))
        rows.append(row)
        print(f"{row[0]:>9.2f} {row[1]:>11.1f}% {row[2]:>19.1f}% {row[3]:>9.3f} {row[4]:>10.3f}")
    print()
    aviary.close()
    return rows


print(f"=== RARL sağlamlık ızgarası | {N_DIRS} yön × {N_SEEDS} eşli tohum (aynı platformlar) ===\n")
results = {}
for label, mp, vp in MODELS:
    r = eval_model(label, mp, vp)
    if r is not None:
        results[label] = r

# --- Karşılaştırma tablosu (iki model de varsa) ---
if len(results) >= 2:
    labels = list(results.keys())
    print("=== Karşılaştırma: başarı(ort) ve başarı(en-kötü-yön) ===")
    hdr = f"{'ruzgar(N)':>9}"
    for lb in labels: hdr += f" | {lb+' ort':>18} {lb+' kotu':>14}"
    print(hdr); print("-" * len(hdr))
    for i, mag in enumerate(MAGS):
        line = f"{mag:>9.2f}"
        for lb in labels:
            line += f" | {results[lb][i][1]:>17.1f}% {results[lb][i][2]:>13.1f}%"
        print(line)
    print("\nBeklenti: 0 N'de RARL ~= nominal; siddet arttikca RARL hem ortalamada hem")
    print("EN-KÖTÜ-YÖNde nominalden belirgin üstün (adversarial sağlamlığın ispatı).")
