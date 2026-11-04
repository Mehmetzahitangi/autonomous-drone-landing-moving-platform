"""
diag_integre.py — FINAL ENTEGRE DOĞRULAMA: kazanan politika (faz4c) + CBF kalkanı,
sabit (held-out) rüzgâr altında. "Rüzgârda iniyor MU ve temas hızı güvenli Mİ?" tek test.

Fikir: Faz 5'in kazananı faz4c çıktı (DR/RARL onu geçemedi). faz4c zaten CBF AÇIK
eğitildi (train_sac: use_cbf=True), yani CBF'i deploy'da açmak eğitim-deploy UYUMLU
doğal konfig. Burada aynı eşli rüzgâr ızgarasında iki konfigi yan yana koyarız:
  - CBF KAPALI: kalkansız (diag_rarl'daki nominal ile aynı sayılar — referans).
  - CBF AÇIK : kalkan devrede. Beklenti: başarı ~= aynı kalır AMA maks vrel (en-kötü
               temas hızı) belirgin düşer. CBF'in işi budur — çarpma hızını sınırlamak.

Rüzgâr env'e `aviary._wind` (3B, Newton, WORLD_FRAME) olarak girer; _physics her fizik
alt-adımında dronun KÜTLE MERKEZİNE uygular (tork bug'ı düzeltildi). Bölüm başında bir
kez set edilir, bölüm boyu sabit. set_wind'in WIND_MAX ölçeğini ATLAYIP doğrudan Newton
veririz → eğitim bütçesi üstünde de sınanır.

Kullanım: python rl/diag_integre.py
"""
import sys, pathlib, os
PROJ = pathlib.Path(__file__).resolve().parent.parent   # proje kökü (CWD'den bağımsız)
sys.path.insert(0, str(PROJ))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env.landing_aviary import LandingAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ACT = ActionType.PID

# ---- Kazanan politika (CWD'den bağımsız, mutlak yol) ----
MODEL_PATH = str(PROJ / "reports/faz4c_heave/sac_landing.zip")
VN_PATH    = str(PROJ / "reports/faz4c_heave/vecnormalize.pkl")

MAGS    = [0.00, 0.02, 0.04, 0.06, 0.08]   # rüzgâr şiddeti (N)
N_DIRS  = 8                                 # yön sayısı (0..315°)
N_SEEDS = 15                                # her (yön,şiddet) için eşli platform
# --------------------------

DIRS = [2*np.pi*k/N_DIRS for k in range(N_DIRS)]


def load():
    stat_venv = make_vec_env(LandingAviary, n_envs=1,
                             env_kwargs=dict(obs=ObservationType.KIN, act=ACT, gui=False))
    vn = VecNormalize.load(VN_PATH, stat_venv)
    vn.training = False
    model = SAC.load(MODEL_PATH)
    return model, vn


def run_cell(aviary, model, vn, wind_vec, seeds):
    """Sabit rüzgârda `seeds` boyunca koş; (başarı_sayısı, vrel_listesi) döndür."""
    succ = 0; vrels = []
    for s in seeds:
        np.random.seed(s)
        obs, info = aviary.reset()
        aviary._wind = np.array(wind_vec, dtype=float)
        term = trunc = False
        fin_vrel = np.nan
        while not (term or trunc):
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(nobs, deterministic=True)
            obs, r, term, trunc, info = aviary.step(action)
            aviary._wind = np.array(wind_vec, dtype=float)
            _, v_rel = aviary._relative_state()
            fin_vrel = float(np.linalg.norm(v_rel))
        succ += bool(info.get("is_success"))
        vrels.append(fin_vrel)
    return succ, np.array(vrels)


def eval_config(label, model, vn, use_cbf):
    aviary = LandingAviary(gui=False, act=ACT, obs=ObservationType.KIN, use_cbf=use_cbf)
    seeds = list(range(N_SEEDS))
    rows = []   # (mag, mean_succ, worst_dir_succ, mean_vrel, max_vrel)
    print(f"Konfig: {label}  (use_cbf={use_cbf})")
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


print(f"=== FINAL ENTEGRE DOĞRULAMA | faz4c + CBF | {N_DIRS} yön x {N_SEEDS} eşli tohum ===\n")
if not (os.path.exists(MODEL_PATH) and os.path.exists(VN_PATH)):
    print(f"[HATA] model yok: {MODEL_PATH} / {VN_PATH}"); sys.exit(1)

model, vn = load()
off = eval_config("CBF KAPALI (kalkansiz)", model, vn, use_cbf=False)
on  = eval_config("CBF ACIK  (kalkanli)",   model, vn, use_cbf=True)

# --- Karşılaştırma ---
print("=== Karsilastirma: basari(en-kotu-yon) ve MAKS VREL (temas hizi) ===")
print(f"{'ruzgar(N)':>9} | {'kotu OFF':>9} {'kotu ON':>9} | {'maksVrel OFF':>13} {'maksVrel ON':>12} {'kazanc':>8}")
print("-" * 70)
for i, mag in enumerate(MAGS):
    kotu_off, kotu_on = off[i][2], on[i][2]
    mv_off, mv_on = off[i][4], on[i][4]
    kazanc = mv_off - mv_on
    print(f"{mag:>9.2f} | {kotu_off:>8.1f}% {kotu_on:>8.1f}% | {mv_off:>13.3f} {mv_on:>12.3f} {kazanc:>+8.3f}")

print("\nBeklenti: basari ON ~= OFF (kalkan inisi bozmaz) AMA maks vrel ON << OFF")
print("(kalkan en-kotu temas hizini kapatir). Ikisi birden olursa TAM SISTEM gosterilmis olur.")
