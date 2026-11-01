# Faz 4-C / C2 — Dikey Heave + Terminal z Feed-Forward — SONUÇ

**Tarih:** 2026-11-01
**Model:** `reports/faz4c_heave/` (SAC, 300k adım, `RES_SCALE=0.05`)

## Ne değişti
Platform artık **dikey de salınıyor** (gerçek gemi güvertesi/heave). İki ayrı iş yapıldı:

### 1) Ortam + kestirim 3B'ye geçti
- `_platform_state(t)` artık z'yi de üretiyor: `z = PLATFORM_TOP + heave_amp·sin(w·t + heave_phase)`, `vz = heave_amp·w·cos(...)`.
- `reset()`'te `heave_amp ∈ [0.0, 0.06] m`, `heave_phase ∈ [0, 2π)` rastgeleleştiriliyor (y salınımıyla aynı frekans = aynı deniz durumu).
- `_move_platform` pedi gerçek z'ye taşıyor; `_relative_state` / `_relative_state_est` z'yi içeriyor.
- **IMM-EKF 3B'ye genişledi** (`estimation/imm_ekf.py`): durum `[px,vx,py,vy,pz,vz]`, ölçüm `[px,py,pz]`. Mod 1 = 3 eksen CV; Mod 2 = x:CV, y & z: harmonik (aynı omega). `block_diag` ile kuruldu.

### 2) Terminal z feed-forward (asıl düzeltme)
Önce heave'li eğitimde **iki bölüm `vrel>0.5` ile başarısız** oldu (hızlı-temas). Kök neden: terminal fazda **yatay** için öngörü vardı ama **dikey** için yoktu — `_cmd_z` sabit çizelgeyle iniyor, platformun dikey hareketini umursamıyordu. Platform yukarı heave yaparken dron aşağı inince bağıl dikey hız patlıyordu.

**Çözüm** (`guidance/landing_guidance.py`, terminal blok): dikeyde de yataydaki feed-forward'ın ikizi —
```python
z_surf = platform_pos[2] + platform_vel[2]·lead + contact_offset   # öngörülen ped yüzeyi
_desc_off = max(_desc_off - descent_step, 0.0)                     # kapanan bağıl pay
target_z = z_surf + _desc_off
```
Artık dikeyde **mutlak çizelge değil, platforma göreli** iniyoruz: hedef z platformun öngörülen yüzeyini taşıdığı için komut hızı ≈ platform vz → temasta bağıl dikey hız ~0. Eski `_cmd_z` kaldırıldı (ölü kod).

## Sonuçlar (diag — 10 rastgele platform, heave dahil, deterministik)
| Metrik | z-ff ÖNCE | z-ff SONRA | Yorum |
|---|---|---|---|
| **BAŞARI (iniş, `is_success`)** | 8/10 | **9–10/10** | Hızlı-temas büyük ölçüde çözüldü (koşuya göre oynar) |
| `vrel>0.5` hızlı-temas | 2 | **0–1** | z feed-forward azalttı; nadiren hâlâ çıkar |
| eval success (eğitim) | %100 | **%100** | — |
| Ort. konum RMSE | 0.033 m | 0.034 m | z eklendi, bozulma yok |
| Ort. hız RMSE | 0.321 | 0.345 m/s | z hız bileşeni yüzünden hafif artış, normal |
| mu_osc | ~0.55 | 0.50–0.64 | IMM y & z harmonik moduna makul bağlanıyor |
| res% (residual/taban) | %26 | **%30** | base ~21 cm, res ~6 cm → taban baskın, RL ±5 cm rötuş |

**Ana sonuç:** Dikey heave eklendiğinde çıkan tek zayıflık (temas hızı) **güdüm tabanından** çözüldü — RL'i büyütmeye gerek kalmadı, `RES_SCALE=0.05` yine yeterli. "Güvenli, sınırlı residual + sağlam klasik taban" mimarisi dikey harekette de tuttu.

## Teşhis: `term-?` etiketi ve gerçek başarı oranı
Diag'a terminal-adım teşhisi eklendi (`[DBG]` satırı: `fin_h`, `fin_horiz`, `fin_vrel`, `term/trunc/is_succ`, bitiş nedeni). Bulgular:
- **Mekanizma:** her bölüm `h<0.05`'e ulaşınca sonlanıyor (`fin_h ≈ min_h`, 0.043–0.049); `is_success` tamamen **vrel kapısı** ile belirleniyor.
- **`term-?` neden çıkmıştı:** diag'da **sabit seed yok** → her koşu farklı 10 rastgele platform çekiyor. İlk koşuda iki bölüm sınırda (min_h ≈ 0.051, 5 cm eşiğinin hemen üstü) kalıp `ever_both` latch'ini tutturamadı ama env kapısı başarı saydı → "term-?" etiketi. Bu kısmı etiketleme kenar durumuydu.
- **ÖNEMLİ düzeltme:** ikinci (teşhisli) koşu **9/10** verdi; tek başarısızlık **gerçek bir vrel çarpması** (ep7, vrel=0.886). Yani z feed-forward hızlı-temasları **azalttı ama %100 elemedi**. İki koşu birlikte: başarı **~%90-100, koşudan koşuya oynuyor**; **temas hızı C2'nin kalan zayıf noktası** (gerçek, seyrek bir mod — kozmetik değil).
- Hard başarısızlıklardan yere-düşme ve devrilme **hiç görülmedi**; tek gerçek risk terminal vrel.

**Kesinleştirme için öneri:** diag'a sabit seed (örn. `--seed`) eklenirse sayı tekrarlanabilir olur. Terminal vrel'i daha sıkı bağlamak reward shaping / jerk cezası (pürüzsüzlük, Faz 6) işi — şimdilik ertelendi.

## Sıradaki
- **Pürüzsüzlük (Faz 6 adayı):** GUI'de hâlâ keskin/hızlı yaklaşma gözlendi; reward shaping (jerk/aksiyon cezası) veya CBF ile yumuşatma.
- IMM z modu için opsiyonel mod-genişletme (mu_osc'u yükseltmek) — posRMSE iyi olduğundan şart değil.
- İleride: görüntü/ArUco relatif poz, PX4 SITL.
