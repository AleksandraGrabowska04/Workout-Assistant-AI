from rep_counter import angle_3pts
from feedback_rules import (
    squat_depth_feedback,
    KneeValgusTracker,
    thigh_inward_angle_feedback,
    narrow_knees_feedback,
)

# Indeksy MediaPipe Pose dla nóg
L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
R_HIP, R_KNEE, R_ANKLE = 24, 26, 28

# Od jakiego kąta zaczynamy sprawdzać uciekanie kolan
KNEE_TRACKER_ACTIVATION_ANGLE = 140

# Minimalna głębokość przysiadu do oceny techniki
MIN_SQUAT_DEPTH_ANGLE = 130

# Próg mówiący, czy jest już przysiad czy jeszcze pozycja stojąca
STANCE_CHECK_ANGLE = 160

# Ile klatek trzymać komunikat o wąskich kolanach
NARROW_KNEES_HOLD_FRAMES = 30


class SquatExercise:
    """
    Klasa obsługująca logikę przysiadu:
    - liczenie powtórzeń
    - analiza głębokości
    - analiza kolan (koślawość / zbyt wąsko)
    - generowanie feedbacku
    """

    def __init__(self, counter, smoother):
        # Licznik powtórzeń
        self.counter = counter

        # Wygładzanie kąta kolana
        self.knee_smoother = smoother

        # Tracker koślawienia kolan (valgus)
        self.knee_tracker = KneeValgusTracker()

        # Licznik trzymania komunikatu o wąskich kolanach
        self.narrow_knees_counter = 0

        # Minimalny kąt kolana w trakcie jednego przysiadu
        self.min_angle_in_down = 999.0

        # Ostatni feedback (np. "OK", "Zejdź niżej")
        self.last_feedback = ""

    def update(self, lm):
        """
        Główna funkcja aktualizująca stan przysiadu w każdej klatce.
        Zwraca:
            reps, state, angle_value, rpm, feedback
        """

        # Prawa noga (kąt kolana)
        hip = lm[R_HIP][:2]
        knee = lm[R_KNEE][:2]
        ankle = lm[R_ANKLE][:2]

        # Obie nogi (kolana) do sprawdzania szerokości
        l_knee = lm[L_KNEE][:2]
        r_knee = lm[R_KNEE][:2]

        # Kąt w kolanie (biodro - kolano - kostka)
        knee_angle = angle_3pts(hip, knee, ankle)
        knee_angle = self.knee_smoother.update(knee_angle)

        # Aktualizacja licznika powtórzeń
        reps, state = self.counter.update(knee_angle)
        rpm = self.counter.rpm()

        fb_stance = None
        fb_posture = None

        # Sprawdzanie wąskiego ustawienia kolan
        if knee_angle < STANCE_CHECK_ANGLE:
            fb_stance = narrow_knees_feedback(
                l_knee,
                r_knee,
                warn_px=110,
                critical_px=90
            )

            if fb_stance:
                self.narrow_knees_counter = NARROW_KNEES_HOLD_FRAMES
            else:
                self.narrow_knees_counter = max(0, self.narrow_knees_counter - 1)

        if self.narrow_knees_counter > 0:
            fb_posture = fb_stance

        # Logika pojedynczego przysiadu -> UP -> DOWN -> UP
        if state == "DOWN":
            # Zapamiętujemy najniższy punkt przysiadu
            self.min_angle_in_down = min(self.min_angle_in_down, knee_angle)

            # Sprawdzamy, czy kolana nie uciekają do środka
            if knee_angle < KNEE_TRACKER_ACTIVATION_ANGLE:
                self.knee_tracker.update(hip, knee, ankle)

        else:  # UP
            # Jeśli był wykonany pełny przysiad
            if self.min_angle_in_down < 999:
                # fb1 -> głębokość przysiadu
                fb1 = squat_depth_feedback(self.min_angle_in_down, target=105)

                fb2 = None  # Koślawienie kolan (valgus)
                fb3 = None  # Kierunek kolana

                if self.min_angle_in_down < MIN_SQUAT_DEPTH_ANGLE:
                    fb2 = self.knee_tracker.feedback()
                    fb3 = thigh_inward_angle_feedback(hip, knee)

                # Składamy komunikaty w jeden tekst
                msgs = [m for m in [fb1, fb2, fb3] if m]
                self.last_feedback = " | ".join(set(msgs)) if msgs else "OK"

                # Reset przed następnym przysiadem
                self.min_angle_in_down = 999.0
                self.knee_tracker.reset()

        # Priorytetem jest feedback postawy > feedback po powtórzeniu
        shown_feedback = fb_posture if fb_posture else self.last_feedback

        return reps, state, knee_angle, rpm, shown_feedback
