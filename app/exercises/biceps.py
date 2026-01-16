from rep_counter import angle_3pts

# Indeksy dla biceps curl (tylko lewa ręka)
L_SHOULDER, L_ELBOW, L_WRIST = 11, 13, 15

# Parametry techniki biceps curl
BICEPS_RESET_ANGLE = 150   # ręka wyraźnie wyprostowana (pozycja startowa)
BICEPS_BAD_ANGLE = 80      # za małe zgięcie ręki


class BicepsExercise:
    """
    Klasa obsługująca logikę biceps curl:
    - liczenie powtórzeń
    - zapamiętywanie minimalnego kąta łokcia
    - ocena jakości ugięcia (czy ręka była wystarczająco zgięta)
    """

    def __init__(self, counter, smoother):
        # Licznik powtórzeń
        self.counter = counter

        # Wygładzanie kąta łokcia
        self.elbow_smoother = smoother

        # Minimalny kąt łokcia w trakcie jednego ugięcia
        self.min_elbow_angle = 999.0

        # Ostatni feedback
        self.last_feedback = ""

        # Poprzedni stan (UP / DOWN)
        self.last_state = None

    def update(self, lm):
        """
        Aktualizacja stanu biceps curl w każdej klatce.
        Zwraca:
            reps, state, angle_value, rpm, feedback
        """

        # Tylko współrzędne (x, y) punktów
        shoulder = lm[L_SHOULDER][:2]
        elbow = lm[L_ELBOW][:2]
        wrist = lm[L_WRIST][:2]

        # Kąt w łokciu (bark - łokieć - nadgarstek)
        elbow_angle = angle_3pts(shoulder, elbow, wrist)

        # Wygładzenie kąta
        elbow_angle = self.elbow_smoother.update(elbow_angle)

        # Aktualizacja licznika ugięć ręki
        reps, state = self.counter.update(elbow_angle)
        rpm = self.counter.rpm()

        # Zapamiętujemy najmniejsze zgięcie łokcia tylko w fazie ruchu w górę
        if state == "DOWN":
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # Domyślnie brak nowego feedbacku
        feedback = self.last_feedback

        # Ocena jakości ruchu tylko po zakończeniu powtórzenia (DOWN -> UP)
        if self.last_state == "DOWN" and state == "UP":

            # Sprawdzamy, czy ugięcie było wystarczająco duże
            if self.min_elbow_angle > BICEPS_BAD_ANGLE:
                feedback = "Zegnij bardziej rękę"
            else:
                feedback = "OK"

            # Reset przed następnym ugięciem
            self.min_elbow_angle = 999.0
            self.last_feedback = feedback

        # Zapamiętujemy aktualny stan, żeby w następnej klatce
        # móc wykryć odpowiedni ruch
        self.last_state = state

        return reps, state, elbow_angle, rpm, feedback
