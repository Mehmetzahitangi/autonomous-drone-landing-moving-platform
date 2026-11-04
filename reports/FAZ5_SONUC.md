# Faz 5 — Rüzgâr Dayanıklılığı + Güvenlik Kalkanı — SONUÇ

**Tarih:** 2026-11-04
**Kapsam:** Rüzgâr altında hareketli platforma iniş. Üç yaklaşım denendi (Domain
Randomization, RARL, CBF kalkanı) ve hepsi aynı mühendislik dersine çıktı.
**Dağıtılacak model:** **`reports/faz4c_heave/`** (faz4c). Gerekçesi aşağıda.

---

## 0. Özet (tek paragraf)
Rüzgâr dayanıklılığını artırmak için önce bir **fizik hatası** bulundu ve düzeltildi
(rüzgâr kuvveti dronun kütle merkezine değil, dünya orijinine uygulanıyordu → en küçük
rüzgârda takla). Hata düzeltilince ortaya çıkan gerçek tablo şuydu: **katmanlı klasik
GNC tabanımız (faz4c) rüzgâra zaten dayanıklı.** Rüzgâr için özel eğitilen iki model
(DR ve RARL) faz4c'yi **geçemedi**. Güvenlik kalkanı (CBF) ise en kötü çarpma şiddetini
provalı biçimde düşürdü (2.67→1.2 m/s) ama kazaları elemedi ve nominal başarıyı düşürdü —
temiz bir kazanç değil. **Sonuç: iyi kurulmuş klasik taban zor yenilir; eklenen ML/kalkan
katmanları bu rejimde marjinal ya da negatif getiri veriyor.**

---

## 1. En kritik bulgu: tork hatası (ve düzeltmesi)
Haftalarca "RL rüzgârı öğrenemiyor / sürekli takla atıyor" sanılan her şeyin tek kök
nedeni bir fizik uygulama hatasıydı.

**Hata:** Rüzgâr kuvveti şöyle uygulanıyordu:
```python
p.applyExternalForce(..., posObj=[0,0,0], flags=p.WORLD_FRAME)
```
`posObj=[0,0,0]` kuvveti **dünya orijinine** uyguluyor, dronun kütle merkezine değil.
Dron z≈0.7 m'de olduğu için ~0.7 m **kuvvet kolu** oluşuyor → 0.03 N'luk küçük bir rüzgâr
bile ~0.021 N·m tork üretiyor → eylemsizlik I≈1.4e-5 ile ~1500 rad/s² açısal ivme →
**anında devrilme.**

**Düzeltme** (`env/landing_aviary.py`, `_physics`): kuvveti dronun o anki dünya
konumuna uygula:
```python
pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[nth_drone], ...)
p.applyExternalForce(..., forceObj=self._wind.tolist(), posObj=pos,
                     flags=p.WORLD_FRAME, ...)
```

**Ders:** Önce ölç, sonra teori. Düzeltme öncesi tüm "RL çöküyor / rüzgâr öğrenilemez"
yorumları yanlıştı; teşhis scriptiyle (diag) üç adımda takla görülünce kök neden netleşti.

---

## 2. faz4c'nin gerçek dayanıklılığı (held-out, sabit rüzgâr)
Hata düzeltilince faz4c'nin (rüzgâr eğitimi **almamış** model) sabit, dışarıdan verilen
rüzgâr altındaki sınavı. 8 yön × 15 eşli tohum, kalkan kapalı:

| rüzgâr (N) | başarı (ort) | başarı (en-kötü-yön) | ort vrel | maks vrel |
|---|---|---|---|---|
| 0.00 | 100.0% | 100.0% | 0.281 | 0.471 |
| 0.02 | 95.8% | 86.7% | 0.312 | 0.758 |
| 0.04 | 90.8% | 80.0% | 0.323 | 0.798 |
| 0.06 | 82.5% | 66.7% | 0.364 | 1.796 |
| 0.08 | 67.5% | 53.3% | 0.477 | 2.668 |

**Nazik degradasyon.** Rüzgâr arttıkça başarı yumuşakça düşüyor. Tek zayıf nokta: yüksek
rüzgârda **temas hızı** sertleşebiliyor (0.08'de en-kötü vrel 2.67 m/s).

**Neden taban zaten dayanıklı?** Rüzgâra karşı asıl işi klasik katman yapıyor: DSL PID'in
integral terimi sabit kuvveti bastırır, IMM-EKF platformu takip eder, ZEM/ZEV güdümü iner.
RL yalnızca **±%5 artık (residual, RES_SCALE=0.05)** yetkisine sahip.

---

## 3. DR ve RARL: rüzgâr için özel eğitim — negatif sonuç
İki model faz4c'den warm-start alınıp eğitildi:
- **DR (Domain Randomization):** her bölümde rastgele sabit rüzgâr.
- **RARL (Robust Adversarial RL):** öğrenen bir "düşman" en kötü yönden rüzgâr estirir;
  curriculum ile WIND_MAX 0.03→0.08 rampalandı. Kalkan kapalı.

3'lü karşılaştırma (başarı ort / en-kötü-yön):

| rüzgâr (N) | faz4c ort / kötü | DR ort / kötü | RARL ort / kötü |
|---|---|---|---|
| 0.00 | **100.0 / 100.0** | 93.3 / 93.3 | 93.3 / 93.3 |
| 0.02 | **95.8** / 86.7 | 95.0 / 86.7 | 91.7 / 86.7 |
| 0.04 | **90.8 / 80.0** | 87.5 / 80.0 | 85.0 / 73.3 |
| 0.06 | **82.5 / 66.7** | 73.3 / 53.3 | 76.7 / 53.3 |
| 0.08 | **67.5 / 53.3** | 64.2 / 40.0 | 60.8 / 40.0 |

**Sonuç: faz4c her seviyede kazandı.** DR ve RARL faz4c'yi hiçbir yerde geçemedi; üstelik
rüzgârsızda 100→93.3'e düştüler (nominali feda ettiler, karşılığında dayanıklılık almadılar).

**Neden?** Darboğaz residual'ın küçüklüğü değil — **rüzgârı taban zaten çözüyor.** Ortada
RL'in düzeltebileceği bir açık kalmadığı için, rastgele rüzgâr (DR) ve düşman (RARL) eğitim
sinyaline çoğunlukla **gürültü** ekledi, keskin nominali törpüledi. (RL'e tam yetki vermek
de yardım etmezdi: bozucu etki tabanın kapasitesini aşmadığı sürece büyük yetki yalnızca
hata yapma payı ekler, dayanıklılık eklemez.)

---

## 4. CBF güvenlik kalkanı: iki entegrasyon yolu
CBF (Control Barrier Function) kalkanı, iniş anında temas hızını (vrel) sınırlamak içindir —
amacı donanımı sert çarpmadan korumak.

### 4a. Deploy-time stacking (kalkanı sonradan takma) — KÖTÜ
faz4c'yi (kalkan görmeden eğitilmiş) deploy'da kalkanla çalıştırmak:

| rüzgâr (N) | başarı (en-kötü) | maks vrel |
|---|---|---|
| 0.00 | 86.7% (100→87 ⬇) | 0.513 |
| 0.04 | 53.3% | 1.385 |
| 0.08 | 46.7% | 1.189 |

Nominal düştü **ve** vrel yarım kapandı. Politika kalkanla "kavga ediyor" (train/deploy
uyumsuzluğu). **Yapılmamalı.**

### 4b. Shielded retrain (kalkanı eğitim-içi öğretme)
faz4c'den warm-start + `use_cbf=True` + rüzgâr (DR) ile 150k adım devam eğitimi →
`reports/faz5_shielded/`. Kalkan açıkken sınav:

| rüzgâr (N) | başarı (ort / kötü) | maks vrel |
|---|---|---|
| 0.00 | 80.0 / 80.0 | 0.546 |
| 0.02 | 80.8 / 73.3 | 0.820 |
| 0.04 | 80.0 / 73.3 | 0.878 |
| 0.06 | 79.2 / 66.7 | 0.779 |
| 0.08 | 70.8 / 53.3 | **1.204** |

Shielded, bolt-on'u **her yerde yendi** ("kalkanı sonradan takma, eğitim içinde öğret"
dersi doğrulandı) ve **en kötü çarpma şiddetini** provalı biçimde düşürdü (0.08'de
2.67→1.20). Ama **nominal başarıyı toparlamadı** (%80 vs faz4c %100).

---

## 5. Dürüst güvenlik analizi (başarısızlık türü)
"Shielded'ın düşen başarısı güvenli ıskalama mı, yoksa gerçek kaza mı?" sorusu ölçüldü
(0.08 N, 120 bölüm). **Not:** bu sayım scripti başarıyı son adım geometrisinden türetir;
env ise bir latch ile belirler → ~13-16/120 ayrışma var, yani KIRIM sayıları bir miktar
şişkin. O yüzden kesin sayıya değil, sağlam sinyallere bakıldı.

| 0.08 N | BAŞARI | KIRIM | DEVRİLME | ISKALA | TIMEOUT |
|---|---|---|---|---|---|
| faz4c (kalkansız) | 68 | 29 | 5 | 0 | 0 |
| shielded (kalkanlı) | 69 | 17 | 5 | 11 | 5 |

**Sağlam okunanlar:**
1. Kalkan, en kötü çarpma **şiddetini** düşürüyor (maks vrel 2.67→1.2) ama kazayı
   **sıfırlamıyor** — hâlâ 0.5–1.2 m/s aralığında temaslar var.
2. İstenen yönde kısmi kayış: shielded'da ISKALA+TIMEOUT = 16 bölüm (faz4c'de 0) — kalkan
   bazı sert çarpmaları güvenli "konamama"ya çeviriyor.
3. Ama bedava değil: shielded nominali ~%15-20 düşürüyor ve 0 rüzgârda bile faz4c'nin hiç
   yapmadığı birkaç tehlikeli sonuç veriyor (kalkanın ani müdahalesi duruşu bozabiliyor).

**Karar:** "Başarı düştü ama güvenli" iddiası **tam geçerli değil** — düşüşlerin bir kısmı
hâlâ orta şiddette çarpma. Kalkan net bir kazanç değil; yalnızca **nadir şiddetli çarpma
kuyruğunu törpüler, bunun karşılığında başarıyı düşürür.**

---

## 6. Faz 5 ana dersi
DR, RARL ve CBF — üç bağımsız yaklaşım da aynı yere çıktı:

> **İyi kurulmuş katmanlı klasik GNC tabanı (faz4c) bu rejimde zor yenilir. Üstüne eklenen
> ML (DR/RARL) ya da güvenlik katmanı (CBF) marjinal veya negatif getiri verir.**

Bu bir olgunluk göstergesidir: **ne zaman yeni eğitim/katman gerekir, ne zaman gerekmez**
ayrımını ölçerek yapabilmek. Negatif sonuç da sonuçtur.

**Dağıtılacak model: `reports/faz4c_heave/` (faz4c).** En yüksek başarı, düşük rüzgârda
sıfır tehlikeli sonuç, nazik degradasyon. Bilinen tek zayıflığı yüksek rüzgârda temas hızı;
bunu adresleme yolu CBF'tir ama yukarıdaki maliyetlerle.

---

## 7. Üretilen dosyalar
| Dosya | İçerik |
|---|---|
| `reports/faz4c_heave/` | **Dağıtılacak model** (faz4c): sac_landing.zip + vecnormalize.pkl |
| `reports/sac_dr.zip`, `vecnorm_dr.pkl` | DR (warm-start) modeli |
| `reports/faz5_dr/` | DR yedeği |
| `reports/sac_rarl.zip`, `vecnorm_rarl.pkl` | RARL modeli |
| `reports/faz5_shielded/` | Shielded retrain modeli + best + eval |
| `rl/train_dr.py`, `train_rarl.py`, `train_shielded.py` | Eğitim scriptleri |
| `rl/diag_rarl.py` | 3'lü dayanıklılık karşılaştırması |
| `rl/diag_integre.py` | faz4c CBF kapalı vs açık (bolt-on) |
| `rl/diag_shielded.py` | faz4c vs shielded doğrulaması |
| `rl/diag_failtype.py` | Başarısızlık türü dökümü (kırım vs ıskalama) |
| `reports/ILERI_CALISMA_NOTLARI.md` | Kavram arşivi (tork bug, warm-start dersleri, CBF) |
| `reports/FINAL_ENTEGRE_SISTEM_ONERISI.md` | Entegre sistem önerisi |

---

## 8. Sıradaki (ertelendi)
- **(Opsiyonel) Başarısızlık türünü resmî is_success ile kesinleştirmek** — latch
  belirsizliğini ortadan kaldıran bir sayım.
- Faz 7: görüntü/ArUco relatif poz. Faz 8: PX4 SITL.
- Teacher-student distillation; CBF abort/go-around.
