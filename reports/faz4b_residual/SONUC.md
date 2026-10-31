# Faz 4-B — Residual RL (ZEM/ZEV tabanı + SAC düzeltmesi) — SONUÇ

**Tarih:** 2026-10-31
**Model (v2, teslim):** `reports/faz4b_residual/` (SAC, 300k adım) — **terminal hız ileri-beslemesi** dahil.

## Aksiyon mimarisi
    hedef = a_base (ZEM/ZEV klasik, guidance/landing_guidance.py)  +  RES_SCALE · a_residual (SAC)
`RES_SCALE = 0.05`. Taban kestirilen platform durumuyla çalışır (POMDP); SAC yalnız düzeltme öğrenir.
Terminal faz: `target_xy = platform_pos[:2] + platform_vel[:2]·lookahead` → platform hareketini **öngörür**
(IMM-EKF hız kestirimini kullanır), geriden kovalamaz.

## v2 sonuçları
| Metrik | Değer |
|---|---|
| eval success | **%100** (ep_len 49.4, reward 65.7) |
| rollout success | %95 (ep_len 49.6, reward 66.5) |
| diag BAŞARI (10 ep, deterministik) | **10/10** |
| Ort. konum RMSE | 0.020 m |
| Ort. hız RMSE | 0.267 m/s |

> Not: v2, v1'e (feed-forward öncesi) göre belirgin daha iyi — ep_len 85→49, reward 12→66.

## RESIDUAL katkısı (diag v4)
| Metrik | Değer | Yorum |
|---|---|---|
| Ort. taban offset | 20.53 cm | ZEM/ZEV tabanı gövdeyi yapıyor |
| Ort. residual offset | 6.31 cm | SAC'ın düzeltmesi |
| Ort. residual/taban | **%31** | Taban baskın, RL sınırlı rötuş → istenen denge |
| Ort. TERMINAL fazı | %42 | Adımların yüzdesi |

## Yörünge pürüzsüzlüğü — glide-slope ERTELENDİ
GUI'de "hızlı yaklaşıp son anda merkeze kayma, geldiği açıda inmeme" gözlendi. `profile_traj`/`base_profile`
ile kök sebep bulundu: **yaklaşmada ZEM/ZEV dikey hatayı yataydan hızlı kapatıyor** → drone dik inip switch'e
yanal açıkla varıyor, kalanı terminalde toparlıyor. (İniş başarısını değil, yörünge estetiğini etkiliyor.)

Denenen çözüm — **glide-slope** (alçalma hızını yatay hizaya bağla) + yatay/dikey ayrıştırma:
- Ayrıştırma `a_max` bütçe-çalınmasını düzeltti (bazı epizotlar düzeldi).
- Ama **yeni kök sebep** ortaya çıktı: alçalma kısılınca drone havada kalıyor, ZEM/ZEV'in **`t_go`'ya bağlı
  yatay kazançları düşük** kalıp **hareketli platformu takip edemiyor** → bazı epizotlar ıraksıyor.
- Bunu düzeltmek **yatay takibi `t_go` çizelgesinden ayrı, saf bir izleme yasası** yapmayı gerektirir —
  1-2 parametre değil, **güdüm tasarımı** işi.

**Karar:** glide-slope geri alındı (terminal feed-forward korundu, v2 onunla eğitildi). Yörünge pürüzsüzlüğü
**doğru yeri** olan sonraki fazda ele alınacak: platform randomization + **ödül şekillendirme (jerk/hız cezası)**
+ curriculum, ve kesin güvenlik için **CBF (Faz 6)**.

## Sıradaki
- **Faz 4-C / robustluk** — platform domain randomization (faz/genlik/frekans/drift) + dikey heave (N3),
  curriculum; yörünge pürüzsüzlüğü burada ilkeli ele alınır.
