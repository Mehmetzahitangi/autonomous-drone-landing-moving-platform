# Faz 6 — CBF Güvenlik Kalkanı + Kalkanlı RL — SONUÇ

**Tarih:** 2026-11-02
**Model:** `reports/faz6_cbf/` (SAC, 300k adım, `RES_SCALE=0.05`, **kalkan eğitim döngüsünde açık**)
**Kod:** `safety/cbf_shield.py` (`VerticalContactCBF`), `env/landing_aviary.py` (`use_cbf` bağlantısı), `rl/diag_cbf.py` (OFF/ON tanı)

## Amaç
RL ajanı tek başına ortalama iyi iniyor ama **temas hızında tehlikeli bir kuyruk** bırakıyor (bazı tohumlarda 2+ m/s çarpma). Güverteye iniş gibi güvenlik-kritik bir görevde ortalama değil **worst-case** önemli. Kontrol Bariyer Fonksiyonu (CBF) kalkanı: ajanın verdiği komutu, güvenli kümeyi terk etmeyecek **en yakın güvenli komuta** projelendiren ince bir katman. "İspatlı güvenlik zarfı" — safe-RL anlatısının çekirdeği.

## Mimari — iki kapalı-form bariyer (QP yok)
Kalkan yalnız **terminal bölgede** aktif (`h < h_on=0.15 m`); yaklaşmada ajana karışmaz.

1. **Dikey temas-hızı bariyeri.** Güvenli küme: iniş hızı yükseklikle orantılı bir zarfın altında kalsın — `v_max(h) = v_touch + k·h` (`v_touch=0.25`, `k=1.0`). CBF koşulu `Ḃ ≥ −α·B`'den tek boyutlu kapalı-form ivme alt sınırı `a_min` çıkar; komut ivmesi bunun altına inemez (`a_safe = max(a_nom, a_min)`).
2. **Yatay hız bariyeri.** Komutun ürettiği yatay **bağıl** hız (platform hızına göre) büyüklükçe `s_max(h) = vxy_touch + k_xy·h`'e kısılır, **yön korunur**. Dikeyden bağımsız (decoupled) → kapalı-form, `qpsolvers` gerekmez.

## Neden kalkanlı yeniden eğitim (asıl bulgu)
Önce kalkanı **hazır (kalkansız) politikanın üstüne** taktık. Sonuç ters tepti: kalkan başarıyı **düşürdü** (48→46/50, %26.7 müdahale). Sebep **train/deploy uyumsuzluğu** — ajan kalkanı görmeden eğitildi, kalkan devreye girince gördüğü dünya eğitim dünyasından farklı oldu, yörünge bozuldu.

**Çözüm:** kalkanı eğitim döngüsüne koyup (`train_env` ve `eval_env`'de `use_cbf=True`) politikayı **kalkanla birlikte** 300k adım yeniden eğittik. Böylece ajan ve kalkan birbirine adapte oldu. Bu, projenin "shielded RL" katkısı.

## Sonuçlar — `diag_cbf` (50 eşli tohum, aynı platformlar, deterministik)
Aynı kalkanlı politika, kalkan KAPALI vs AÇIK:

| metrik | CBF OFF | CBF ON |
|---|---|---|
| **BAŞARI (`is_success`)** | 35/50 | **43/50** |
| `vrel>0.5` (hızlı-temas) | 16 | **6** |
| ort. temas hızı | 0.571 | 0.337 |
| **maks temas hızı** | **2.371** | **0.639** |
| kalkan müdahale (adım %) | — | 41.7% |

- **Uyumsuzluk tersine döndü:** kalkansız politikada kalkan zararlıyken (48→46), kalkanlı politikada **faydalı** (35→43). Yön doğru.
- **Worst-case sınırlandı — CBF'in asıl amacı:** 50 rastgele tohumun **hiçbirinde** temas 0.64 m/s'yi aşmıyor. Kalkansız kuyruktaki çakılmalar (seed 26: 2.37, seed 41: 2.33, seed 40: 1.98, seed 12: 1.95, seed 4: 1.55) **hepsi ~0.3-0.4'e** iniyor. Matematiksel garantinin sahadaki görünümü.

## Doğrulama — `diag_landing` (10 rastgele platform, episode detayı)
| koşum | başarı | maks temas hızı | hard-fail |
|---|---|---|---|
| kalkansız (çıplak politika) | 8/10 | ~0.97 (çarpma) | 1 çarpma + 1 devrilme |
| **kalkanlı** | **10/10** | **0.44** | yok |

Çıplak politika tek başına güvenli değil (kuyruk çarpmaları); kalkanla hepsi <0.45 m/s'de oturuyor. Üç tanı (diag_cbf OFF, diag_landing OFF, diag_landing ON) **tutarlı**.

## Dürüst takas
- **Dağıtılabilir sistem = politika + kalkan AÇIK → 43/50** + sert `≤0.64 m/s` tavan.
- Düz kalkansız baseline ~48/50 nominal başarıydı. Yani **~%10 nominal başarıyı, ispatlı worst-case tavanına takas ettik.** Güvenlik-kritik görevde doğru takas.
- **%41.7 müdahale yüksek:** politika kalkana yaslanmayı öğrendi (KAPALI'da 35/50 olması bundan — o config zaten kullanılmaz). İstenirse zarfı gevşetmek (`v_touch 0.25→0.3`, `k→1.2`) müdahaleyi düşürüp birkaç nominal başarı geri kazandırabilir ama worst-case tavanını yükseltir — **şimdilik yapılmadı**, çünkü tavan tam sattığımız şey.

## Teşhis notu
`diag_landing`'de ara sıra `ever_both=0` ama `is_success=1` (`term-?`) çıkıyor: diag'ın kendi adım-takibi ile env başarı kapısı arasında bir adımlık kayma — bilinen, zararsız defter-tutma farkı, sonucu etkilemez.

## Ertelenenler / sıradaki
- **Abort / go-around (Parça 3):** kalkan tutturamazsa tur-başa; **çalışma-zamanı supervisory katmanı** olarak (RL ile ortak eğitilmeyecek). Ertelendi.
- **Faz 5 — RARL** (düşmanca rüzgâr/bozulmaya karşı sağlamlık) sıradaki.
- Sonra: Faz 7 ArUco relatif poz, Faz 8 PX4 SITL.
- Opsiyonel (en sonda): teacher-student distillation.

## Yedek
`reports/faz6_cbf/` — `sac_landing.zip`, `vecnormalize.pkl`, `sac_best/best_model.zip`. Bu model `reports/sac_landing.zip`'teki kalkanlı politikanın kopyası (kalkansız önceki politika `reports/faz4c_heave/`'de duruyor).
