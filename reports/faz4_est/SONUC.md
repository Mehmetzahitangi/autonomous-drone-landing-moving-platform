# Faz 4-A — IMM-EKF Kapalı Çevrim (POMDP) — SONUÇ

**Tarih:** 2026-10-29
**Model:** `reports/faz4_est/` (SAC, 300k adım, GPU/CUDA)

## Ne değişti (Faz 3 → Faz 4-A)

Politikanın gördüğü gözlem artık **ground-truth platform durumu değil**, gürültülü
ölçümden (`meas_std = 0.03 m`) **IMM-EKF ile kestirilen** göreli durum.

- Gözlem (obs) → `_relative_state_est()` (kestirim)
- Ödül / sonlandırma → `_relative_state()` (gerçek durum)

Bu **asimetrik dünya bilgisi**dir (POMDP): ajan belirsizlik altında uçmayı öğrenir,
ama değerlendirme fiziksel gerçeğe göre yapılır. GNC katmanlaması:
- **Navigasyon** = IMM-EKF (CV + harmonik mod)
- **Güdüm** = SAC (nereye nişan alınacağı)
- **Kontrol** = DSL-PID (sabit, `_preprocessAction` içinde ±0.1 m göreli hedef)

## Karşılaştırma: Faz 3 (düz SAC, ground-truth) vs Faz 4-A (kestirim)

| Metrik | Faz 3 (`faz3_1a`, ground-truth) | Faz 4-A (`faz4_est`, IMM-EKF) |
|---|---|---|
| diag iniş başarısı (deterministik, 10 ep) | 10/10 | **10/10** |
| rollout success_rate (300k, stokastik) | ~%100 | %96 |
| eval success_rate (5 ep, deterministik) | %100 | %80 (küçük örneklem gürültüsü) |
| max tilt (rad) | 0.08–0.17 | 0.10–0.18 |
| Gözlemde platform durumu | gerçek | **kestirilen (gürültülü)** |

**Sonuç:** Gözlem gürültülü kestirime geçtiği hâlde iniş başarısı korundu (10/10).

## IMM-EKF kestirim kalitesi (diag)

| Metrik | Değer | Yorum |
|---|---|---|
| **pos RMSE** | **0.022 m** | Yatay başarı eşiğinin (0.3 m) ~14× altında → ihmal edilebilir |
| vel RMSE | 0.214 m/s | Konumdan yüksek (normal: hız türevden, ölçümde yok); iniş hız kapı 0.5 m/s |
| mu_osc | ~0.67 | IMM, CV ve harmonik modu ~yarı yarıya karıştırıyor |

### `mu_osc` neden ~0.9 değil?
Platform 0.3 Hz çok yavaş salınıyor; kısa ufukta sabit-hız (CV) modeli de sinüsü iyi
fit ettiği için filtre iki mod arasında kararsız. **Ama karışımın da pos RMSE'si 2 cm**,
yani pratikte iniş performansını etkilemiyor. Tuning gerektirmez (gözlem).

### eval %80 vs rollout %96 — çelişki değil
- rollout = 2296 bölüm ortalaması, **stokastik** politika (keşif gürültüsü dahil).
- eval = yalnız **5 bölüm**, deterministik; `±81.98` reward sapmasıyla 1 kötü bölüm %20 oynatır.
- **diag 10/10 deterministik** → en temsili sonuç.

## Bu fazda kapanan madde
- **A) IMM-EKF kapalı çevrim** ✅ — ground-truth → kestirim geçişi, iniş korundu.

## Sıradaki
- **B) Residual RL** — ZEM/ZEV klasik güdüm tabanı (Faz 2) üstüne SAC artık (residual) politika.
- **C) Zorlaştırma** — dikey heave (N3), curriculum, RARL (Faz 5).
