# Faz 4-C / C1 — Platform Domain Randomization — SONUÇ

**Tarih:** 2026-11-01
**Model:** `reports/faz4c_randomization/` (SAC, 300k adım, `RES_SCALE=0.05`)

## Ne değişti
Platform artık her bölümde **farklı** yörünge izliyor (eskiden hep aynıydı → ezber riski).
`reset()`'te rastgeleleştirilenler:
- `plat_phase` ∈ [0, 2π) — salınım başlangıç fazı
- `plat_amp` ∈ [0.10, 0.30] m — salınım genliği
- `plat_freq` ∈ [0.25, 0.35] Hz — salınım frekansı (IMM `wave_hz` bununla kurulur)
- `plat_drift` — rastgele yön (açı) + hız [0, 0.12] m/s

Platform konumu tek kaynağa toplandı: **`_platform_state(t)`** (konum+hız). `_move_platform`,
`_relative_state`, `_relative_state_est` hepsi bunu çağırır → randomization hepsini tutarlı etkiler.
(Heave = **C2**'ye bırakıldı; z şimdilik sabit.)

## Sonuçlar (diag — 10 RASTGELE platform, deterministik)
| Metrik | Değer | Yorum |
|---|---|---|
| BAŞARI (iniş) | **10/10** | Çeşitli yörüngelere **genelleme tuttu** |
| Ort. konum RMSE | **0.024 m** | Hâlâ ~2 cm; kestirim bozulmadı |
| Ort. hız RMSE | 0.271 m/s | — |
| mu_osc | 0.47–0.67 | Frekans rastgele olduğu için IMM osilasyon moduna daha az bağlanıyor; posRMSE iyi olduğundan sorun değil |
| res% (residual/taban) | **~%30** | base ~22 cm, res ~6 cm → taban baskın, RL ±5 cm rötuş |
| eval success | %80 (5 bölüm) | Küçük örneklem gürültüsü; temsili değil |

**Ana sonuç:** Yalnız **±5 cm sınırlı residual** ile politika, rastgeleleştirilmiş hareketli
platformlara **10/10** iniyor. `RES_SCALE=0.05` yeterliydi — büyütmeye gerek kalmadı.
"Güvenli, sınırlı RL genelleşiyor" tezi için güçlü kanıt.

## Headless stres testi (30 rastgele bölüm)
- **NaN/Inf yok**, en uzun bölüm **61 adım** (truncation sağlam), en büyük |pos| 0.99 m (arena içinde), **30/30 başarı**.
- Simülasyon ve politika sağlam; mantık hatası yok.

## GUI donması — not
`eval_sac.py` canlı GUI'de (PyBullet OpenGL) bilgisayarı donduruyordu. Teşhis: **iş yükü/logic değil**
(headless 30 bölüm anında, temiz). Sebep PyBullet'in Windows'ta canlı GUI render'ı (sürücü/OpenGL etkileşimi).
**Çözüm:**
- `eval_sac.py` hafifletildi/sağlamlaştırıldı: `--n/--gui/--no-gui/--speed` parametrik, `try/finally` ile her
  durumda `env.close()`.
- İzleme için **headless video üreticisi** (`rl/make_video.py`, şimdilik gitignore'da): DIRECT + tinyrenderer
  (CPU) → GPU penceresi yok → donma yok. Örnek çıktı: `reports/faz4c_landing.mp4` (3 başarılı iniş).
- Sayı için: `python rl/eval_sac.py --no-gui --n 30` veya `diag_landing.py`.

## Sıradaki
- **C2 — dikey heave**: platform z salınımı (gerçek gemi güvertesi). PLATFORM_TOP hareketli olur;
  estimator'ın z'ye genişletilmesi gündeme gelir. Pürüzsüzlük + CBF ileri fazlarda.
