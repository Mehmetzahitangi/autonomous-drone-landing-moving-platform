"""
tag_layout.py — Nested AprilTag yerleşimi (make_marker ve aruco_detector TEK kaynaktan okur).

Pad 0.60 m. İki tag, PAD MERKEZİNE göre bilinen ofsetlerde (çakışmaz):
  - ID 0 BÜYÜK (0.26 m), üstte (+y): yüksek irtifadan görünür.
  - ID 1 KÜÇÜK (0.10 m), altta (-y): dron yaklaşınca (terminal) kadrajda kalır.
Herhangi bir tag görülünce pad merkezi = tag_dünya − ofset ile geri çözülür.
"""
CANVAS_PX = 800
PAD_M     = 0.60
PXPM      = CANVAS_PX / PAD_M          # piksel/metre (1333.3)

APRILTAG_DICT = "DICT_APRILTAG_36h11"

# id -> (fiziksel kenar m, pad-merkezine göre ofset (x,y) m)
# KÜÇÜK tag tam MERKEZDE (dron oraya iner → terminalde kadrajda kalır).
# BÜYÜK tag yukarı kaydırılmış (sadece yüksek irtifada görünür, çakışmaz).
TAGS = {
    0: dict(size=0.20, offset=(0.0,  0.18)),   # BÜYÜK / yüksek irtifa
    1: dict(size=0.10, offset=(0.0,  0.00)),   # KÜÇÜK / terminal (merkez)
}
