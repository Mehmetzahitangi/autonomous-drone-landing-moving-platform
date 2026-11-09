# Faz 7 — Görsel Algıyla İniş (AprilTag, gerçekçi kamera) — SONUÇ

**Tarih:** 2026-11-08
**Kapsam:** Platform konumunu simülatörden doğrudan almak ("sihirli sensör") yerine drona monte, gövdeye
sabit, aşağı bakan bir kameradan AprilTag ile ölçmek. Ölçüm IMM-EKF'e girer; güdüm (ZEM/ZEV) + RL residual
aynen kalır.
**Dağıtılacak sistem:** **faz7_res ağırlıkları** (`reports/faz7_res/`, Faz 7 ortamında yeniden eğitildi, §3d)
+ gerçekçi kamera + **"önce hizalan, sonra in" kapısı** (`align_cfg={}`), CBF kapalı. Kamerada **%81**.
**Değerlendirme:** `rl/eval_faz7.py`, 100 eşli senaryo (seed 0–99), Wilson %95 GA, eşli testlerde kesin McNemar.

---

## 0. Özet
Gerçek kamerayla taban sistem **%59** başarı verdi; ideal sensörle aynı kontrolcü %95. Aradaki farkın sebebi
ölçümle bulundu: dron **pad'in yanında alçalıyor**, alçak irtifada yandaki tag kadraja sığmıyor ve dron son
~0.3 m'yi bayat kestirimle uçup **yatay kayarak** temas ediyordu. PX4 hassas inişinin ve MBZIRC şampiyonu
NimbRo'nun kullandığı **"önce hizalan, sonra in"** kapısı eklenince başarı **%78**'e çıktı (p = 0.005).
IMM-EKF (ideal ölçüm) tavanı %99. Residual Faz 7 ortamında yeniden eğitilince kamerada **%81**, en kötü temas
hızı 1.71 → 0.76 m/s (§3d). **RL residual'ı bu yapıda belirgin katkı veriyor:** RL'siz %52 → RL'li %78
(p = 0.0001). Yol üstünde, sıralı bölüm koşan tüm eski değerlendirmeleri etkileyen bir **PID durum taşıma
hatası** bulundu ve düzeltildi.

---

## 1. Kurulan algı hattı
- **Marker:** iç içe AprilTag 36h11 (büyük 0.20 m, ofset +0.18 m; küçük 0.10 m, merkez). `perception/tag_layout.py`.
- **Kamera:** gövdeye sabit, 120° FOV, 480 px, `ER_TINY_RENDERER`. Gimbal kullanılmadı: bu sınıf
  dronlarda iniş kamerası sabittir (gimbal yalnız uzak takipte, bkz. `LITERATUR_ORNEK_PROJELER.md`).
- **Gerçekçi kamera modeli** (`perception/aruco_detector.py`, `CAM_PRESETS["realistic"]`):
  15 Hz kare, 100 ± 20 ms gecikme (env'de zaman damgalı ölçüm kuyruğu), pozlama süresine göre hareket
  bulanıklığı, optik bulanıklık, ışık kazancı (0.6–1.3), sensör gürültüsü, IMU tutum hatası (0.5° sapma +
  0.2° gürültü), SUBPIX köşe iyileştirme.
- **Ölçüm:** göreli (pad − kamera, dünya eksenlerinde), pad yönüne (yaw) duyarlı tag ofseti. PX4
  `LANDING_TARGET` ile aynı biçim.

## 2. Bulunan hatalar
| Hata | Etki | Düzeltme |
|---|---|---|
| **PID iç durumu bölümler arası taşınıyordu** (`DSLPIDControl.reset()` env reset'inde çağrılmıyordu) | Aynı seed sıralı koşuda farklı sonuç; faz6+kamera "uzaklaşma" çöküşü (%3) tamamen buydu → düzeltince %54 | `LandingAviary.reset`'te `ctrl.reset()`. **Sıralı bölüm koşan eski diag/eğitimler etkilenmiş olabilir** (bkz. ACIK_NOTLAR N8). |
| Supervisor faz4c modunda hiç çağrılmıyordu (yalnız `use_residual=False` dalında) | Faz 7'deki "supervisor no-op" bunun sonucuydu, dropout değil | Supervisor kaldırıldı, yerine doğru bağlanmış alçalma kapısı |
| `loadTexture` yanlış PyBullet client'ına (önceki oturum) | %100 dropout | `physicsClientId` |
| `diag_integre.py` başlığı "faz4c CBF açık eğitildi" diyor | Yanlış: faz4c CBF'siz, faz6 CBF'li eğitildi | Not düzeltildi; eval scriptinde doğru eşleşme |

## 3. Deneyler ve sonuçlar

### 3a. Algının bedeli (taban, kapı yok)
| Model | Gauss (ideal) sensör | Kamera — ideal | Kamera — gerçekçi |
|---|---|---|---|
| faz4c (CBF'siz) | **95%** (89–98) | 54% (44–63) | 59% (49–68) |
| faz6 (CBF'li) | 90% (83–94) | 54% (44–63) | 51% (41–61) |

- Gerçekçilik eklemelerinin (gecikme, 15 Hz, bulanıklık) ek bedeli ölçülemeyecek kadar küçük. Bedel kameranın kendisinde.
- CBF kamera modunda katkı vermiyor (yatay bariyer kestirilen platform hızına güveniyor).
- Ölü kaynak ayrıştırması (40 bölüm): tespit oranı yüksekliğe göre 0.40 m üstü %99, 0.15–0.25 m **%48**,
  0.07 m altı **%0** (kamera–pad < 5 cm: yakın kırpma + kadraj). KIRIM'larda tag 0.27 m'de kayboluyor
  (başarılılarda 0.14 m); temastaki kaymanın ana kaynağı **takip** hatası (0.56 m/s), kestirim değil (0.17 m/s).

### 3b. Denenen ve işe yaramayan kaldıraçlar
| Kaldıraç | Sonuç | Neden |
|---|---|---|
| PID'ye hedef hız (velocity feed-forward) | **Negatif**: Gauss 94→40%, kamera 52→15%, takla | Tarif hatası: yaklaşmada `target_vel = v + a·look` türev sönümlemesini sıfırladı; terminalde gürültülü kestirim hızı PID'e girdi. Geri alındı. |
| Gimbal (nadir-stabilize) kamera | 59 → 63% (53–72), GA içinde | Kayıp eğimden değil, yanal ofsetten. Bu dron sınıfı için gerçekçi de değil. |
| Kamera 30 Hz (15 yerine) | 60 → 60% (n=40) | Kör süre kare hızından bağımsız: geometri |
| Monolitik SAC yeniden eğitim (önceki oturum) | Iraksadı / çöktü | Yüksek dropout + POMDP + PID taşıma hatası (o sırada bilinmiyordu) |

### 3c. "Önce hizalan, sonra in" kapısı (kazanan)
Kural: yatay hata `tol = 0.10 + 0.30·h` içine girene kadar irtifa tut, yatayda pad'e git (hedef en fazla
0.15 m ötede); 10 adımdan uzun ölçüm yoksa alçalma; h < 0.12 m'de iptal yok. Beklerken güdüm saati donar.
Tarif: `reports/FAZ7_ADIM3_HIZALAN_SONRA_IN.md`.

| Koşu (gerçekçi kamera, ep 12 s) | RL | Başarı (%95 GA) | KIRIM | ISKALA | TIMEOUT | Tespit |
|---|---|---|---|---|---|---|
| b0 taban | ✔ | 59% (49–68) | 36 | 5 | 0 | 59% |
| **b1 hizalan** | ✔ | **78% (69–85)** | 19 | 2 | 0 | 76% |
| b2 + hız şartı | ✔ | 75% (66–82) | 19 | 6 | 0 | 79% |
| b3 + hız şartı + tırmanma | ✔ | 78% (69–85) | 14 | 8 | 0 | 78% |
| b1 hizalan | ✘ | 52% (42–62) | 29 | 5 | 14 | 72% |
| b1 hizalan, Gauss | ✔ | **99% (95–100)** | 1 | 0 | 0 | — |
| b1 hizalan, Gauss | ✘ | 90% (83–94) | 7 | 0 | 3 | — |

- b1 vs b0: 31 kazanılan / 12 kaybedilen → **p = 0.005**. b3 vs b1: 12/12 → fark yok → en sade olan b1 seçildi.
- Kapı kontrol tavanını bozmadı (IMM-EKF ideal ölçüm: 95 → 99%).

### 3d. Residual'ın Faz 7 ortamında yeniden eğitimi (ACIK_NOTLAR N9)
faz4c, kapının olmadığı, PID durum taşıma hatalı, 8 s'lik bir ortamda eğitilmişti (kapının "irtifa tut" fazını
hiç görmedi). `rl/train_faz7.py`: aynı SAC tarifi (300k adım), IMM-EKF (ideal ölçüm), ama **kapı açık, PID
reset düzeltilmiş, bölüm 12 s**. İki varyant: CBF kapalı (`faz7_res`) ve CBF açık (`faz7_res_cbf`). İkisi de
kararlı eğitildi (~65k adımdan sonra değerlendirme bölümlerinde %100; Faz 7 başındaki kameralı eğitimlerin aksine
çöküş yok). Değerlendirme gerçekçi kamerada, b1 konfigi, aynı 100 seed:

| Model | Kamera başarı (GA) | İdeal ölçüm | Temas vrel p95 | **Temas vrel maks** | Bekleme | Ort. süre |
|---|---|---|---|---|---|---|
| faz4c (b1) | 78% (69–85) | 99% | 0.64 | **1.71** | 33% | 2.4 s |
| **faz7_res** (CBF kapalı) | **81% (72–87)** | 98% | 0.59 | **0.76** | 24% | 2.2 s |
| faz7_res_cbf (CBF açık) | 79% (70–86) | 94% | 0.64 | 0.94 | 27% | 2.2 s |

- faz7_res vs faz4c (kamera): 15'e 12, p = 0.70 → **başarıda anlamlı fark yok** (dürüst sonuç).
- Ama **en kötü temas belirgin iyileşti:** maks 1.71 → 0.76 m/s; 1 m/s üstü temas 1 → 0. Model kapıyla uyumlu:
  daha az bekliyor (33% → 24%), maks bayatlık medyanı 11 → 8 adım.
- **CBF bu yapıda katkı vermedi:** kamerada 79% vs 81% (7'ye 9, p = 0.80), ideal ölçümde 94% vs 98–99%
  (eğilim kötü yönde, p = 0.13), en kötü temas da CBF'siz modelden iyi değil (0.94 vs 0.76). Kapı + uyumlu
  residual, CBF'in Faz 6'da kestiği çarpma kuyruğunu zaten kesiyor.
- **Karar:** Faz 7 modeli = **faz7_res** (CBF kapalı). Gerekçe: başarı eşit-veya-biraz-iyi, en kötü durum
  belirgin iyi, nihai mimaride eğitilmiş (train/deploy uyumlu; Faz 8 sim-to-sim için temiz başlangıç).

## 4. RL'in rolü
Aynı konfig, residual sıfırlanarak (`--no-rl`):
- Kamera: **78% vs 52%**, 34'e 8 → **p = 0.0001**. Gauss: 99% vs 90%, p = 0.012.
- Temas hızı p95: RL 0.64, RL'siz 1.10 m/s.
- Mekanizma: RL'siz klasik taban yavaş (ort. 8 s, saat tabanlı çizelge), uzun süre bayat kestirimle uçuyor,
  TIMEOUT'a düşüyor. Residual alçalmayı hızlandırıyor ve temas hızının kuyruğunu kısıyor.
- **Dürüst not:** Katkının bir kısmı klasik tabanın yavaş ayarını telafi ediyor olabilir; daha iyi ayarlı bir
  klasik taban farkı daraltabilir.
- Mimari yorum: Klasik GNC omurgası + sınırlı residual + çalışma anı güvenlik katmanı = Run-Time Assurance
  (Simplex) yapısı. Literatürde gerçek uçuşta iniş yapan sistemlerin hiçbiri uçtan uca RL değil.

## 5. Sınırlar
- Kalan ana hata KIRIM (yatay kayma, vxy ≈ 0.6 m/s).
- Son ~7 cm'de kamera geometrik olarak kör: gerçek sistemlerde de var, "kararlı otur" ile yönetiliyor.
- Platform sabit yönelimli (yaw/roll yok); ölçüm kodu dönen pada hazır ama ortam bunu test etmiyor.
- Residual IMM-EKF (ideal ölçüm) ile eğitildi, kamerayla test edildi; kamera boşlukları eğitimde görülmedi.
  İleride algı vekili (ölçülen tespit-yükseklik haritasından sentetik dropout) ile eğitim denenebilir.

## 6. Gelecek çalışma
- **Sensör füzyonu:** kamera + platformun RTK-GNSS/IMU yayını veya UWB (ideal tavan %99 ile %78 arasındaki
  21 puanın asıl kaynağı algı). Borowczyk vd. (50 km/h) bu yaklaşımı kullanıyor.
- Faz 8: PX4 SITL — kapı ≈ `PLD_HACC_RAD` / precland yatay yaklaşma; ölçüm `LANDING_TARGET` biçiminde hazır.

## 7. Nasıl çalıştırılır
```bash
python rl/eval_faz7.py --tag faz7_final --model faz7 --cam realistic --ep-len 12 --env '{"align_cfg": {}}'
```
RL'siz karşılaştırma için `--no-rl`, IMM-EKF (ideal ölçüm) tavanı için `--sensor gauss`.
Yeniden eğitim: `python rl/train_faz7.py --cbf off --out reports/faz7_res` (~40 dk, GPU).

## 8. Dosyalar
- Model: `reports/faz7_res/` (dağıtılan), `reports/faz7_res_cbf/` (karşılaştırma)
- Kod: `rl/train_faz7.py` (eğitim), `perception/aruco_detector.py` (kamera modeli), `env/landing_aviary.py` (ölçüm kuyruğu, `_descent_gate`,
  `align_cfg`, PID reset), `guidance/landing_guidance.py` (`hold` modu), `rl/eval_faz7.py`, `rl/reclassify_faz7.py`
- Sonuçlar: `reports/faz7_eval/<koşu>/{episodes.csv, summary.json}`, loglar `reports/faz7_eval/logs/`
- Teşhis ve tarifler: `FAZ7_TESHIS_ve_COZUM_ONERILERI.md`, `FAZ7_ADIM2_YATAY_HIZ_ESLEME.md` (negatif sonuç),
  `FAZ7_ADIM3_HIZALAN_SONRA_IN.md`, `LITERATUR_ORNEK_PROJELER.md`
