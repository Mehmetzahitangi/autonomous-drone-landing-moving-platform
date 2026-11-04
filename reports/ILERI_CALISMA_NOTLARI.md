# İleri Çalışma Notları — Bahsedilen / Ertelenen / Vazgeçilen Konular

> **Tarih:** 2026-11-03
> **Amaç:** Proje boyunca konuştuğumuz ama uygulamadığımız, ertelediğimiz ya da
> vazgeçtiğimiz kavramları tek yerde toplamak — ileride ayrı ayrı çalışabilmen için.
> Her başlıkta: **ne olduğu**, **bizim bağlamımızda neden gündeme geldiği**, **durumu**,
> ve **nasıl çalışılacağı / kaynak yönü** var.

---

## 0. EN ÖNEMLİ BULGU — rüzgâr "tork" bug'ı (ve eski teşhislerin düzeltilmesi)  —  [ÇÖZÜLDÜ 2026-11-03]

**Bug:** `_physics`'te rüzgâr kuvveti `p.applyExternalForce(..., posObj=[0,0,0], flags=WORLD_FRAME)`
ile **dünya orijinine** uygulanıyordu — dronun kütle merkezine değil. Dron z≈0.7'de olduğundan
~0.7 m kaldıraç kolu oluşuyor: 0.03 N'lik kuvvet ~0.021 N·m tork → atalet 1.4e-5 ile ~1500 rad/s²
→ dron **anında takla**. Yani yatay itiş sanılan şey, dev bir devirme torkuydu.

**Düzeltme:** kuvveti kütle merkezine uygula —
`pos,_ = p.getBasePositionAndOrientation(...)` alıp `posObj=pos` (artık `posObj=[0,0,0]` değil).

**Neyi çürüttü (ders — semptom kovalama):** Bu tek satır Faz 5'in TÜM çilesinin kaynağıydı.
Aşağıdakiler bu bug'ın semptomuydu, sandığımız sebepler DEĞİL:
- "Warm-start boş-buffer çöküşü / obs kirlenmesi" (§2, §3) → gerçekte takla bug'ıydı.
- "Nominal 0.02 N'de %0'a düşüyor, sabit kuvvet integral olmadan reddedilemez" (§5) → bug'dı;
  **düzeltince faz4c hiç eğitimsiz 0.08 N'e kadar kademeli iniyor** (0.04'te %90, 0.08'de %67).
- "Fresh DR ep_len 3'te takılıyor (entropy çöküşü / reçete sapması)" → yine bu bug (her bölüm takla).

**Alınan ders:** Birden fazla "akıllı" eğitim-dinamiği açıklaması üretmeden önce, **tek bir
mekanik/fizik testi** (diag_term: iyi politika + rastgele aksiyon, ölüm sebebini logla) kök
sebebi dakikalar içinde verir. **Önce ölç, sonra teori.**

---

## 1. Teacher–Student (privileged → student) distillation  —  [ERTELENDİ, N6]

**Ne:** Önce "ayrıcalıklı" (privileged) bilgiye sahip güçlü bir **teacher** politika eğitilir
(ör. platformun gerçek durumunu, rüzgârı, kütleyi tam bilir). Sonra sadece **gerçekte
ölçülebilen** gürültülü girdiyi gören bir **student** politika, teacher'ı taklit ederek eğitilir.
Student dağıtıma (deploy) gider; teacher atılır.

**Bizde neden gündeme geldi:** POMDP'yi (kısmi gözlem) işlevsel olarak zaten çözdük —
Faz 4-A'da gözlem = IMM-EKF kestirimi (ground-truth değil). Yani teacher-student bir
**işlevsel gereklilik değil**, bir **portföy vitrini / sağlamlık artışı** olurdu.

**Durum:** Ertelendi. Tüm fazlar bitince opsiyonel ek deney.

**Nasıl yapılır:**
1. **Teacher:** ground-truth gözlemli politika (≈ Faz 3 / tam bilgili SAC).
2. **Student:** sadece gürültülü göreli ölçüm geçmişi (LSTM/GRU) **veya** IMM çıktısı + MLP.
3. **DAgger (önemli):** student'ı koştur; ziyaret ettiği **her durumda** teacher'ın aksiyonunu
   sor; `(student_obs, teacher_action)` çiftleriyle denetimli regresyon. Bu, "student kendi
   hatalarının yarattığı durumları görmüyor" dağılım-kayması problemini çözer.
4. **Alternatif:** asimetrik actor-critic — critic ayrıcalıklı bilgiyi görür (eğitimde),
   actor sadece gözlemi; deploy'da actor yeter.

**Kaynaklar:**
- Ross, Gordon, Bagnell, *"A Reduction of Imitation Learning... to No-Regret Online Learning"* (DAgger), AISTATS 2011.
- Pinto et al., *"Asymmetric Actor Critic for Image-Based Robot Learning"*, RSS 2018.
- Lee et al., *"Learning quadrupedal locomotion over challenging terrain"*, Science Robotics 2020 — teacher (privileged) → student (proprioception), bizim senaryoya çok yakın mantık.
- Kumar et al., *"RMA: Rapid Motor Adaptation for Legged Robots"*, RSS 2021 — ayrıcalıklı parametreleri bir **latent**'e sıkıştırıp student'a tahmin ettirme (aşağıdaki §5 ile bağlantılı).

---

## 2. RARL — Robust Adversarial RL  —  [DR'den SONRA TEKRAR DENENDİ — bug fix sonrası]

**Ne:** Protagonist (iniş politikası) ile bir **adversary** (bozucu) sıfır-toplamlı oyun oynar.
Adversary, protagonisti bozmaya çalışan bir kuvvet/aksiyon uygular; ödülü protagonistinkinin
**negatifidir**. Dönüşümlü (alternating) eğitilir. Deploy'da adversary atılır → en-kötü-duruma
karşı sağlam politika kalır.

**Bizde:** Adversary = 2B yatay **rüzgâr kuvveti**. Protagonist = residual SAC (use_cbf=False).
6 tur × (protagonist + adversary), WIND_MAX bütçesi.

**Neden vazgeçtik — üç çöküş modu (hepsi değerli ders):**
1. **Sıfırdan + güçlü düşman:** Protagonist inmeyi *öğrenirken* düşman itince tutunamadı;
   shaping ödülünü hack'leyip (pada dalıp) sert çakmayı öğrendi → 0 rüzgârda bile %0.
2. **Warm-start boş-buffer çöküşü:** faz4c'den başlattık ama `SAC.load` replay buffer'ı
   getirmez → boş buffer'la ilk gradyanlar overfit → iyi politika birkaç bin adımda bozuldu.
3. **Warm-start obs-normalizasyon kirlenmesi:** `learning_starts` rastgele warmup'ı,
   `vn.training=True` iken obs istatistiklerini (obs_rms) bozdu → politika yanlış-ölçekli
   obs görüp anında devrildi. (bkz. §3)

**Ek olarak yapısal sorun:** Sabit yan kuvveti integral olmadan reddetmek zor olduğundan
(bkz. §5), öğrenen düşman "kazanılamaz" bir sınav kurabiliyordu → oyun kararsız.

**Ne zaman mantıklı:** RARL kötü bir fikir değil; sadece (a) protagonist görevi önce stabil
çözebilmeli, (b) düşman bütçesi curriculum'la yavaş artmalı, (c) düşmanın bozulması
**reddedilebilir** olmalı (sabit kuvvet yerine geçici gust gibi). Doğru kurulunca güçlü.

> **GÜNCELLEME (§0):** Yukarıdaki "çöküş modu 2 ve 3" (boş-buffer, obs kirlenmesi) gerçekte
> **tork bug'ının** semptomuydu. Bug düzeltilince RARL tekrar masaya geldi ve DOĞRU kuruldu.

**Doğru warm-start RARL kurulumu — eski `train_rarl.py`'dan farklar (hepsi öğrendiğimiz dersler):**
1. **Protagonist warm-start:** faz4c ağırlıkları + faz4c vecnorm yüklenir, `lr=1e-4` nazik ince
   ayar — sıfırdan DEĞİL. (Sıfırdan protagonist düşmanla aynı anda öğrenemiyordu.)
2. **Binding satırları YERİNDE:** `prot_core.adv/adv_vn` ve `adv_core.prot/prot_vn` iki SAC de
   kurulduktan sonra bağlanır. (Silinirse adversary fazı `NoneType.predict` ile çöker — yaşadık.)
3. **Curriculum:** `WIND_MAX` tur başına 0.03→0.08 rampalanır. DR'nin eğitimde göremediği
   yüksek rüzgârı (0.06-0.08) kapsar ve **worst-case'i doğrudan hedefler** — RARL'ın asıl amacı.
4. **`use_cbf=False`** (faz4c ile adil kıyas) + tork bug'ı env'de düzeltili (kuvvet COM'a).
5. **Adversary fresh**, ödülü protagonistinkinin negatifi; deploy'da atılır.

**Kaynak:** Pinto, Davidson, Sukthankar, Gupta, *"Robust Adversarial Reinforcement Learning"*, ICML 2017.

---

## 3. SAC warm-start / fine-tuning tuzakları  —  [GENEL DERS]

Off-policy (SAC gibi) bir modeli checkpoint'ten devam ettirmek incedir. Üç tuzağı bizzat yaşadık:

1. **Boş replay buffer:** `model.save/load` ağ ağırlıklarını getirir, **buffer'ı getirmez.**
   Boş buffer'la hemen gradyan atmak kritik ağı (Q) ıraksatır. → Çözüm: ya buffer'ı da
   kaydet/yükle (`save_replay_buffer`/`load_replay_buffer`), ya buffer'ı iyi politikayla
   doldurup sonra eğit, ya da çok düşük lr ile nazik git.
2. **VecNormalize obs_rms kirlenmesi:** `vn.training=True` iken rastgele/warmup adımlarının
   çılgın gözlemleri normalizasyon istatistiğini bozar; yüklü politika yanlış-ölçekli obs
   görür. → Çözüm: warm-start'ta obs istatistiğini **dondur** (yüklenen istatistik kalsın),
   veya hiç rastgele warmup yapma.
3. **Ödül normalizasyon ölçeği:** Politika A ölçeğinde eğitildiyse, devam ederken B ölçeğinde
   ödül verirsen kritik şaşırır. → norm_reward ayarını tutarlı tut.

**Pratik sonuç (bizim kararımız):** Bu env'de warm-start üç kez ısırdı; **fresh (sıfırdan)
eğitim** tüm bu baş ağrılarını kaldırır ve bu env'de çalıştığı kanıtlı (faz4c fresh'ti).
Zaman karşılığında güvenilirlik. Warm-start gerekiyorsa yukarıdaki üç maddeyi aynı anda
düzeltmeden yapma.

---

## 4. Domain Randomization (DR)  —  [UYGULANIYOR, Faz 5'in yerine]

**Ne:** Öğrenen düşman yerine, her bölümde ortamı **rastgele** değiştir (rüzgâr, kütle,
sürtünme, gecikme, sensör gürültüsü). Politika geniş bir dağılıma karşı sağlam olmayı öğrenir.
Stabil, basit, sim2real'in temeli.

**Bizde:** `reset()`'te rastgele **sabit rüzgâr** (yön + şiddet ∈ [0, WIND_MAX]), `dr_wind`
bayrağıyla. Eğitimde açık, eval/diag'da kapalı (rüzgârı diag kendi kontrol eder).

**Önemli ayrım — gözlemlenebilir mi:** Bozulma gözlemde yoksa (bizim rüzgâr obs'ta yok),
politika onu ancak **etkisinden (kayma)** çıkarsayıp reaktif reddeder; mükemmel olmayabilir
ama nominali açık ara geçer. Gözlemlenebilir yapmak (rüzgârı obs'a koymak veya kestirmek)
işi kolaylaştırır → §5 ve RMA (§1).

**Varyantlar (ileride):**
- **Geçici gust** (birkaç adımlık darbe) — reddedilebilir, iyi bir sağlamlık testi.
- **Kütle / atalet randomizasyonu** — payload belirsizliği.
- **Sensör gürültüsü / gecikme randomizasyonu** — gerçek algı için.

**Kaynak:** Tobin et al., *"Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"*, IROS 2017. Ayrıca Peng et al. 2018 (dynamics randomization).

---

## 5. Sabit bozulma reddi, integral terim ve gözlemlenemez rüzgâr  —  [KAVRAM]

> **DÜZELTME (§0):** "Nominal 0.02 N'de %0'a düşüyor" ölçümü bir **tork bug'ındandı**; düzeltince
> faz4c rüzgârı iyi reddediyor (0.04'te %90). Yani buradaki "steady kuvvet reddedilemez" korkusu
> pratikte geçersizdi. Yine de aşağıdaki kavramsal tartışma, ileride **steady-state hassasiyeti**
> için geçerli ve değerli.

**Mesele (kavramsal):** Memoryless bir politika sabit, gözlemlenemez bir bozulmayı ne kadar reddedebilir?

- **DSL PID'de integral VAR** (`integral_pos_e`, `I_COEFF_FOR`; anti-windup ±2 xy). Yani
  sistem sabit bozulmayı *bir ölçüde* reddedebiliyor.
- **Ama yetmiyor çünkü:** (a) integral her bölümde sıfırlanıyor; (b) iniş ~1 saniyelik
  **geçici** manevra, integralin oturacak vakti yok; (c) RL her adım yeni hedef komutluyor
  (hareketli platform) → sistem hiç sabit setpoint'e oturmuyor; (d) temas kapısı sıkı, tam
  temas anındaki küçük kayma bile başarısız.
- **Memoryless MLP politika** sabit rüzgârı yine de kaymadan çıkarsayıp proportional
  reddedebilir (RL bunu DR ile öğrenir), ama kalıcı küçük hata (P-kontrol gibi) kalabilir.

**İyileştirme yönleri (ileride):**
1. **Rüzgârı gözleme koy:** IMM-EKF'i rüzgâr/bias kestirecek şekilde genişlet → politika
   doğrudan görür ve ileri-besler.
2. **Geçmiş / tekrarlayan ağ (LSTM):** politika son N adımı görüp bias'ı kendi çıkarsar.
3. **Integral-benzeri durum:** gözleme biriken konum-hatası entegralini ekle.
4. **RMA-tarzı latent:** ayrıcalıklı parametreleri (rüzgâr dâhil) bir latent'e sıkıştır,
   student latent'i geçmişten tahmin etsin (§1 ile birleşir).

---

## 6. CBF abort / go-around (Parça 3)  —  [ERTELENDİ, Faz 6]

**Ne:** Dikey + yatay temas-hızı bariyerlerini (uyguladık, Faz 6) tamamlayan **çalışma-zamanı
denetleyici (supervisory)** katman: iniş koşulları kötüyse (çok yüksek vrel, aşırı kayma)
**iptal edip tekrar yaklaşma (go-around)** tetikle.

**Durum:** Ertelendi. **RL ile ortak eğitilmeyecek** — runtime'da kural/mantık katmanı olarak
eklenecek (politikanın üstünde süpervizör). Böylece öğrenme dinamiğini bozmaz, deterministik
güvenlik mantığı olur.

**Kaynak:** Ames et al., *"Control Barrier Functions: Theory and Applications"*, ECC 2019.

---

## 6b. Faz 5'te CBF neden KAPALI + final entegre sistem  —  [KARAR + ÖNERİ]

**Faz 5'te neden `use_cbf=False` (tüm eğitim + diag):**
1. **Adil kıyas:** Taban faz4c kalkansız; rüzgâr dayanıklılığını onunla elma-elmaya ölçmek için
   DR/RARL da kalkansız eğitiliyor ve test ediliyor.
2. **Ayrı/ortogonal katkılar:** CBF (Faz 6) rüzgâr dayanıklılığını değil, **temas hızını (vrel)**
   sınırlar. Kalkan açık olsa iki katkı birbirine karışır; diag'daki yüksek maks vrel (2.2-2.7)
   de kalkan kapalı olduğu içindir.

**Final entegre sistem (fikir):** Faz 5'i kazanan politika (faz4c / DR / RARL hangisiyse) +
deploy'da üstüne **CBF kalkanı** → hem rüzgâra dayanıklı hem vrel-sınırlı tam sistem. İki yol:
- **(a) Deploy-time stacking:** hazır politikanın üstüne kalkanı inference'ta tak. Ucuz, retrain yok.
  AMA Faz 6 bunu gördü: kalkansız eğitilmiş politikanın üstüne kalkan takınca train/deploy
  uyumsuzluğu (48→46) — politika kalkanla "kavga ediyor".
- **(b) Shielded RL retrain:** kazanan politikayı `use_cbf=True` ile (rüzgâr DR'siyle) yeniden eğit.
  Faz 6 bunu zaten yaptı, işe yaradı (35→43, vrel 2.37→0.64).

**ÖNERİM (sade):** Ağır birleşik retrain'e GİRME; bunun yerine **hafif bir entegre DOĞRULAMA** yap:
kazanan Faz 5 politikasını al, CBF'i AÇ, **rüzgâr altında** diag koş. Tutarsa (hem başarı yüksek
hem vrel sınırlı) → tam sistem gösterilmiş olur, yeni eğitim gerekmez.
- **Neden:** Faz 6'nın shielded modeli zaten "iyi inici + kalkan birlikte"ydi; faz4c de rüzgâra
  zaten dayanıklı çıktı → sistem büyük ölçüde ELİMİZDE. Eksik tek şey "rüzgâr + kalkan birlikte"
  bir doğrulama koşusu. Ucuz, net, hikâyeyi bağlar.
- **Önermediğim:** DR+RARL+CBF'i sıfırdan birlikte ko-eğitmek — çok emek, az kazanç, ayrı
  ispatladığımız temiz katkıları karıştırır.
- **Tek istisna:** hafif doğrulamada vrel hâlâ yüksek çıkarsa, o zaman kazanan politikayı kısa bir
  `use_cbf=True` + DR ince ayarıyla (yol b) sertleştir.

---

## 7. Glide-slope terminal yaklaşım  —  [VAZGEÇİLDİ, N5]

**Ne denendi:** Terminal yaklaşımı pürüzsüzleştirmek için yatay/dikey ayrıştıran glide-slope
(t_go-çizelgeli kazanç). **Iraksadı:** düşük yatay kazanç dikey a_max bütçesini yiyordu.
**Tavşan deliği** olarak görülüp geri alındı; terminal feed-forward'lı v2 korundu.

**Nereye kaldı:** Pürüzsüzlük işi **reward shaping (jerk/temas cezası) + CBF** ile ele alınacak.

---

## 8. Diğer kısa fikirler / sıradaki gerçek adımlar

- **Curriculum learning:** zorluğu (rüzgâr/heave/drift) yavaş artırmak. RARL'da eksikti;
  DR'de örnekleme aralığını zamanla büyüterek uygulanabilir.
- **Reddedilebilir adversary seçimi:** RARL'ı ileride denersek, sabit kuvvet yerine
  **impulsive gust** veya kütle/atalet bozulması daha adil ve öğrenilebilir bir oyun kurar.
- **Faz 7 — ArUco:** görsel marker ile platform pozu (gerçek algı köprüsü).
- **Faz 8 — PX4 SITL:** yazılım-döngüde-donanım, gerçek otopilot entegrasyonu.

---

## Toplu Kaynak Listesi

| Konu | Kaynak |
|---|---|
| RARL | Pinto et al., ICML 2017 |
| DAgger | Ross, Gordon, Bagnell, AISTATS 2011 |
| Asimetrik Actor-Critic | Pinto et al., RSS 2018 |
| Teacher→Student (quadruped) | Lee et al., Science Robotics 2020; Miki et al. 2022 |
| RMA (latent adaptasyon) | Kumar et al., RSS 2021 |
| Domain Randomization | Tobin et al., IROS 2017; Peng et al. 2018 |
| Control Barrier Functions | Ames et al., ECC 2019 |

> Not: Bu dosya "yapılacak" listesi değil — **kavram arşivi.** Fiili TODO ve faz durumu için
> `ACIK_NOTLAR_TODO.md`'ye bak.
