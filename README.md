# Hareketli Platforma Otonom İniş

Dron'un hareket hâlindeki bir platforma (gemi güvertesi / araç / USV) yumuşak otonom inişi
— sivil arama-kurtarma / lojistik senaryosu.

## Durum
- ✅ **Faz 1** — Hareketli platform ortamı (gym-pybullet-drones üstüne) + IMM-EKF göreli durum kestirimi
- ✅ **Faz 2** — ZEM/ZEV + PD klasik güdüm tabanı (doğrulanmış iniş)
- ✅ **Faz 3 (düz SAC)** — Hareketli platforma **%100 otonom yumuşak iniş** (10/10 değerlendirme, temas hızı < 0.5 m/s, nazik uçuş; tilt 0.08–0.17 rad)
- ⏳ **Sıradaki** — IMM-EKF'i kapalı çevrime almak (ground-truth yerine **kestirilen** platform durumu), ardından residual RL → CBF güvenlik → görsel (ArUco) → PX4 SITL

> RL'in rolü **güdüm** katmanı: nereye nişan alınacağını öğrenir. Alçak seviye kontrol DSL-PID'de,
> platform durumu kestirimi IMM-EKF'te kalır (GNC katmanlaması).

## Klasör yapısı
| klasör | içerik | faz |
|---|---|---|
| `env/` | gym-pybullet-drones üstüne hareketli platform ortamı | Faz 1 |
| `estimation/` | IMM-EKF göreli durum kestirimi | Faz 1 |
| `guidance/` | ZEM + alçalma profili (klasik taban) | Faz 2 |
| `rl/` | SAC eğitim/değerlendirme giriş noktaları (`train_sac.py`, `eval_sac.py`, `diag_landing.py`); residual RL, RARL | Faz 3–5 |
| `safety/` | CBF kalkanı: koridor + temas limiti + abort | Faz 6 |
| `vision/` | ArUco ped tespiti, pikselden göreli poz | Faz 7 |
| `px4_bridge/` | politikanın ROS2 düğümü sarmalı, PX4 SITL | Faz 8 |
| `reports/` | tablolar, grafikler, SONUC notları | — |

## Ortam
```
conda create -n inis_drone312 python=3.12
conda activate inis_drone312
pip install -r requirements.txt
```

> ⚠ Proje yolu **yalnız ASCII karakter** içermeli (Türkçe İ/ş/ğ yok). PyBullet/PX4 gibi
> native araçlar Windows'ta Unicode yollu dosyaları açamıyor.
