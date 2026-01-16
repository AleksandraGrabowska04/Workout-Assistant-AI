import cv2
import csv

from pose_utils import PoseEstimator, draw_pose, landmarks_to_np
from rep_counter import SmoothValue, RepCounter

from exercises import SquatExercise, BicepsExercise

# Funkcja wyboru ćwiczenia 
def choose_exercise():
    print("Wybierz ćwiczenie:")
    print("1 - Przysiad (Squat)")
    print("2 - Biceps Curl")

    choice = input("Podaj numer ćwiczenia: ")

    if choice == "1":
        return "squat"
    elif choice == "2":
        return "biceps"
    else:
        print("Nieprawidłowy wybór, domyślnie: squat")
        return "squat"


# Funkcja do wyświetlania tekstu na ekranie
def put_text(frame, text, y, scale=0.7):
    cv2.putText(
        frame, text, (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, (255, 255, 255), 2, cv2.LINE_AA
    )


def main():

    EXERCISE = choose_exercise()  # Wybór ćwiczenia przez użytkownika

    # Przygotowanie do zapisu raportu 
    csv_file = open("training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "exercise", "reps", "state", "angle", "rpm", "feedback"
    ])

    cap = cv2.VideoCapture(0) # Uruchomienie kamery
    pe = PoseEstimator() # Tworzymy obiekt do wykrywania pozy ciała

    knee_smoother = SmoothValue(window=5) # Wygładzamy kąt kolana z ostatnich 5 klatek
    elbow_smoother = SmoothValue(window=3) # Wygładzamy kąt łokcia z ostatnich 3 klatek

    squat_counter = RepCounter(down_threshold=140, up_threshold=160) # Licznik przysiadów
    biceps_counter = RepCounter(down_threshold=120, up_threshold=155) # Licznik biceps curl

    if EXERCISE == "squat":
        exercise = SquatExercise(
            counter=squat_counter,
            smoother=knee_smoother
        )
    else:
        exercise = BicepsExercise(
            counter=biceps_counter,
            smoother=elbow_smoother
        )

    # Przechowuje stan z poprzedniej klatki (UP / DOWN),
    # i jest potrzebny do wykrycia momentu, w którym kończy się powtórzenie (DOWN -> UP)
    last_state = None

    # Główna pętla
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1) # Odbicie lustrzane

        result = pe.process(frame)
        lm = landmarks_to_np(result, frame.shape)

        if lm is not None:
            reps, state, angle_value, rpm, feedback = exercise.update(lm) # Przeliczenie danych (kąt, stan, powtórzenia, feedback) dla jednej klatki obrazu

            if EXERCISE == "biceps":
                state_label = "CURL" if state == "DOWN" else "HAND DOWN"
            else:
                state_label = state

            put_text(frame, f"REPS: {reps}   STATE: {state_label}", 30)

            put_text(frame, f"ANGLE: {angle_value:.1f}", 60)
            put_text(frame, f"RPM(30s): {rpm:.1f}", 90)
            put_text(frame, f"FEEDBACK: {feedback}", 120, scale=0.6)

            # Zapis do csv tylko po zakończeniu powtórzenia 
            if last_state == "DOWN" and state == "UP":
                csv_writer.writerow([
                    EXERCISE,
                    reps,
                    state,
                    round(angle_value, 1),
                    round(rpm, 1),
                    feedback
                ])

            # zapamiętujemy aktualny stan, żeby w następnej klatce
            # móc wykryś ruch DOWN -> UP
            last_state = state

        # Rysowanie szkieletu i okna
        frame = draw_pose(frame, result)
        cv2.imshow("Workout Assistant - MVP", frame)

        # Zamknięcie przez ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
