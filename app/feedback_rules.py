import numpy as np

def squat_depth_feedback(min_knee_angle, target=110, tolerance=5):
    """
    Sprawdza, czy przysiad jest wystarczająco głęboki
    na podstawie minimalnego kąta kolana.

    min_knee_angle -> najmniejszy kąt kolana w całym przysiadzie
    target=110 -> docelowa głębokość (ok. kąt dla poprawnego przysiadu)
    tolerance=5 -> tolerancja błędu
    """
    if min_knee_angle > (target - tolerance):
        return "Zejdź niżej"
    return None

def thigh_inward_angle_feedback(hip, knee, max_angle_deg=20):
    # Sprawdzenie, czy kolano nie idzie za bardzo do środka
    # max_angle_deg -> maksymalny dopuszczalny kąt uciekania kolana do środka

    # Rozpakowujemy współrzędne biodra i kolana
    hx, hy = hip
    kx, ky = knee

    # Tworzymy wektor od biodra do kolana (pokazuje, w jakim kierunku idzie kolano)
    vx = kx - hx
    vy = ky - hy

    # Liczymy kąt tego wektora względem pionu
    angle = abs(np.degrees(np.arctan2(vx, vy)))

    # Jeśli kąt jest za duży ->>> kolano ucieka do środka
    if angle > max_angle_deg:
        return "Wypchnij kolana na zewnątrz"
    return None

# Sprawdzenie, czy kolana nie są zbyt blisko siebie
def narrow_knees_feedback(left_knee, right_knee, warn_px=115, critical_px=85):
    # warn_px -> dystans ostrzegawczy
    # critical_px -> dystans krytyczny
    
    # Bierzemy tylko współrzędne x, bo interesuje nas szerokość
    lx, _ = left_knee
    rx, _ = right_knee

    # Liczymy odległość poziomą między kolanami
    dist = abs(lx - rx)

    if dist < critical_px:
        return "Kolana za blisko!"
    elif dist < warn_px:
        return "Ustaw kolana szerzej"
    return None

# Śledzenie uciekania kolan (valgus -> koślawość) w czasie ruchu
# Klasa zbiera dane z całego przysiadu
class KneeValgusTracker:
    def __init__(self):
        # min_ratio -> zapamiętujemy najgorszy moment (odchylenie kolana względem kostki)
        self.min_ratio = 1.0

    def reset(self):
        # Czyścimy dane przed kolejnym przysiadem
        self.min_ratio = 1.0

    def update(self, hip, knee, ankle):
        # Aktualizacja w każdej klatce
        # Bierzemy tylko współrzędne x
        hx, _ = hip
        kx, _ = knee
        ax, _ = ankle

        knee_offset = abs(kx - hx) # Jak bardzo kolano odchyla się od biodra w bok
        ankle_offset = abs(ax - hx) + 1e-6 # Jak daleko w bok jest kostka (1e-6 ->>> zabezpieczenie przed dzieleniem przez zero)

        # Stosunek kolana względem biodra do kostki względem biodra
        ratio = knee_offset / ankle_offset
        self.min_ratio = min(self.min_ratio, ratio)

    def feedback(self, threshold=0.85):
        """
        threshold -> próg wykrycia koslawienia kolan
        self.min_ratio to najmniejszy (najgorszy) stosunek pozycji kolana do pozycji kostki
            liczony w trakcie całego przysiadu, klatka po klatce. 
            Im mniejsza wartość, tym bardziej kolano ucieka do środka
        """
        if self.min_ratio < threshold:
            return "Wypchnij kolana na zewnątrz"
        return None
