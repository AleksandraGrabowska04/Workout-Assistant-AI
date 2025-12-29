import cv2
import numpy as np
from pose_utils import PoseEstimator, draw_pose, landmarks_to_np
from rep_counter import angle_3pts, SmoothValue, RepCounter
from feedback_rules import (
    squat_depth_feedback,
    KneeValgusTracker,
    thigh_inward_angle_feedback,
    narrow_knees_feedback,  
)

NARROW_KNEES_HOLD_FRAMES = 30  # Ile klatek trzymać komunikat

# Indeksy MediaPipe Pose
L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
R_HIP, R_KNEE, R_ANKLE = 24, 26, 28

KNEE_TRACKER_ACTIVATION_ANGLE = 140 # Od jakiego kąta zaczynamy sprawdzać uciekanie kolan
MIN_SQUAT_DEPTH_ANGLE = 135 # Minimalna głębokość przysiadu do oceny techniki

# Próg mówiący, czy jest już przysiad czy jeszcze pozycja stojąca
STANCE_CHECK_ANGLE = 160

# Funkcja do wyświetlania tekstu na ekranie
def put_text(frame, text, y, scale=0.7):
    cv2.putText(
        frame, text, (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, (255, 255, 255), 2, cv2.LINE_AA
    )


def main():
    cap = cv2.VideoCapture(0) # Uruchomienie kamery
    pe = PoseEstimator() # Tworzymy obiekt do wykrywania pozy ciała

    knee_smoother = SmoothValue(window=5) # Wygładzamy kąt kolana z ostatnich 5 klatek
    counter = RepCounter(down_threshold=140, up_threshold=160)

    knee_tracker = KneeValgusTracker()
    
    narrow_knees_counter = 0 # Licznik, jak długo pokazywać komunikat o wąskich kolanach
    min_angle_in_down = 999.0 # Zmienna zapamiętuje najmniejszy kąt kolana podczas jednego przysiadu 
    # 999.0, bo ta wartość startowa jest na pewno większa niż każdy prawdiwy kąt
    last_feedback = ""

    # Główna pętla
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1) # Odbicie lustrzane

        result = pe.process(frame)
        lm = landmarks_to_np(result, frame.shape)

        posture_feedback = None

        if lm is not None:
            # Prawa noga (kąt)
            # Bierzemy tylko (x, y) punktów
            hip = lm[R_HIP][:2]
            knee = lm[R_KNEE][:2]
            ankle = lm[R_ANKLE][:2]

            # obie nogi (kolana)
            l_knee = lm[L_KNEE][:2]
            r_knee = lm[R_KNEE][:2]

            knee_angle = angle_3pts(hip, knee, ankle)
            knee_angle = knee_smoother.update(knee_angle)

            reps, state = counter.update(knee_angle)
            rpm = counter.rpm()

            # Sprawdzanie wąskiego ustawienia kolan
            if knee_angle < STANCE_CHECK_ANGLE:
                fb_stance = narrow_knees_feedback(
                    l_knee,
                    r_knee,
                    warn_px=110,
                    critical_px=85
                )

                # Sprawdzamy odległość między kolanami
                if fb_stance:
                    narrow_knees_counter = NARROW_KNEES_HOLD_FRAMES
                else:
                    narrow_knees_counter = max(0, narrow_knees_counter - 1)

            if narrow_knees_counter > 0:
                posture_feedback = fb_stance


            # Logika pojedynczego przysiadu ->> UP -> DOWN -> UP
            if state == "DOWN":
                # Zapamiętujemy najniższy punkt przysiadu
                min_angle_in_down = min(min_angle_in_down, knee_angle)

                # Sprawdzamy, czy kolana nie uciekają do środka
                if knee_angle < KNEE_TRACKER_ACTIVATION_ANGLE:
                    knee_tracker.update(hip, knee, ankle)

            # Powrót do góry 
            else:  # UP
                # Jeśli był wykonany pełny przysiad
                if min_angle_in_down < 999:
                    # fb -> feedback 

                    # fb1 -> głębokość przysiadu
                    fb1 = squat_depth_feedback(min_angle_in_down, target=105)

                    fb2 = None # Uciekanie kolan (valgus)
                    fb3 = None # Kierunek kolana w dole przysiadu

                    if min_angle_in_down < MIN_SQUAT_DEPTH_ANGLE:
                        # fb2 i fb3 liczone są tylko, jeśli przysiad był wystarczająco wysoki
                        fb2 = knee_tracker.feedback()
                        fb3 = thigh_inward_angle_feedback(hip, knee)

                    # Składamy komunikaty w jeden tekst
                    msgs = [m for m in [fb1, fb2, fb3] if m]
                    last_feedback = " | ".join(set(msgs)) if msgs else "OK"

                    # Reset przed następnym przysiadem
                    min_angle_in_down = 999.0
                    knee_tracker.reset()

            put_text(frame, f"REPS: {reps}   STATE: {state}", 30)
            put_text(frame, f"KNEE ANGLE: {knee_angle:.1f}", 60)
            put_text(frame, f"RPM(30s): {rpm:.1f}", 90)

            shown_feedback = posture_feedback if posture_feedback else last_feedback
            put_text(frame, f"FEEDBACK: {shown_feedback}", 120, scale=0.6)

        # Rysowanie szkieletu i okna
        frame = draw_pose(frame, result)
        cv2.imshow("Workout Assistant - Squat MVP", frame)

        # Zamknięcie przez ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
