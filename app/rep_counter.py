import time
import numpy as np
from collections import deque

def angle_3pts(a, b, c):
    """
    Oblicza kąt ABC w stopniach.
    Punkt B jest wierzchołkiem kąta (czyli liczymy kąt w punkcie B).
    a, b, c to krotki (x, y) => (szerokość, wysokość)
    Np. a -> (xa, ya)
    """

    # Zamieniamy punkty (x, y) na tablice NumPy
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Tworzymy wektory 1. ba -> od punktu B do A 2. bc -> od B do C
    ba = a - b
    bc = c - b

    # Liczymy cosinus kąta
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6) # 1e-6 -> zabezpieczenie przed dzieleniem przez 0
    cosang = np.clip(cosang, -1.0, 1.0) # Ograniczamy wynik do zakresu [-1, 1], bo przez błędy numeryczne mógłby wyjść np. 1.00000000001

    # Zamiana cos -> kąt (radiany) -> stopnie i zwracamy kąt w stopniach
    return np.degrees(np.arccos(cosang))


class SmoothValue:
    """
    Klasa służy do wygładzania skaczących wartości np. kąta z klatki na klatkę
    Ze względu na sytuacje, gdzie w jednej klatce kąt np. kolana to 92 stopnie a w innej 88 stopni i potem 
    szum 95 stopni
    """
    def __init__(self, window=5): # window = 5 => bierzemy pod uwagę ostatnie 5 wartości
        self.buf = deque(maxlen=window) # Kolejka na maksyymalnie window elementów

    def update(self, v):
        # Funkcja wywoływana za każdym razem, gdy mamy nowy pomiar v
        self.buf.append(float(v)) # Dodajemy nową wartość do bufora
        # Jeśli bufor był pełny, to deque sam usunie najstarszą
        return sum(self.buf) / len(self.buf) # Zwracamy średnią arytmetyczną z wartości w buforze


# Liczenie powtórzeń 

class RepCounter:
    """
    Licznik powtórzeń na podstawie wartości np. wysokość bioder, kąt stawu
    """
    def __init__(self, down_threshold, up_threshold):
        self.down_th = down_threshold # Próg uznania ruchu w dół
        self.up_th = up_threshold # Próg uznania powrotu w górę

        self.state = "UP" # Pozycja startowa
        self.reps = 0 # Licznik powtórzeń
        self.rep_timestamps = [] # Lista czasów, kiedy wykonano powtórzenia (otrzebne do RPM -> repetitions per minute)

    def update(self, metric_value):
        # Jeśli byliśmy w górze i wartość spadła poniżej progu ->>> ruch w dół
        if self.state == "UP":
            if metric_value < self.down_th:
                self.state = "DOWN"

        # Jeśli byliśmy w dole i wartość wróciła powyżej progu ->>> zakończone powtórzenie
        elif self.state == "DOWN":
            if metric_value > self.up_th:
                self.state = "UP"
                self.reps += 1
                self.rep_timestamps.append(time.time()) # Zapisujemy czas wykonania powtórzenia

        return self.reps, self.state

    def rpm(self, window_sec=30):
        # Liczy powtórzenia na minutę z ostatnich window_sec sekund

        now = time.time()
        # Usuwamy stare powtórzenia spoza okna czasowego
        self.rep_timestamps = [
            t for t in self.rep_timestamps if now - t < window_sec
        ]

        # Jeśli mamy mniej niż 2 powtórzenia nie da się policzyć tempa (inaczej nie można policzyć jak szybko kolejne powtórzenia po sobie następują)
        if len(self.rep_timestamps) < 2:
            return 0.0

        # Czas pomiędzy pierwszym a ostatnim powtórzeniem
        dt = self.rep_timestamps[-1] - self.rep_timestamps[0]
        if dt <= 0: # Zabezpieczenie przed błędami
            return 0.0

        return (len(self.rep_timestamps) - 1) / dt * 60.0
