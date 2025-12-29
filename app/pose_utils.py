import cv2
import mediapipe as mp
import numpy as np

# Skróty do modułów MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseEstimator:
    def __init__(self):
        # Inicjalizacja modelu MediaPipe Pose
        self.pose = mp_pose.Pose(
            static_image_mode=False, # Tryb wideo (śledzenie w czasie)
            model_complexity=1, # Złożoność modelu (parametr mów, jak duży i dokładny model ma być użyty do wykrywania pozycji ciała)
            enable_segmentation=False, # Brak segmentacji sylwetki, czyli oddzielenia ciała od tła (brak, ponieważ nie potrzebujemy osobno danych o tle)
            min_detection_confidence=0.5, # Próg wykrycia pozycji ciała (jak bardzo model musi być pewny, że na obrazie faktycznie jest ciało)
            min_tracking_confidence=0.5, # Próg śledzenia między klatkami (jak bardzo model musi być pewny, że to nadal ta sama poza w kolejnej klatce)
        )

    def process(self, frame_bgr):
        """
        Zwraca obiekt typu mediapipe.python.solutions.pose.PoseLandmarksResult
        z informacją o wykrytej pozycji ciała.
        Zawiera głównie 33 landmarki wraz z ich pozycją, głębokością i pewnością wykrycia
        """
        # Konwersja obrazu z BGR (OpenCV) na RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Analiza klatki przez model 
        result = self.pose.process(frame_rgb)
        return result

def draw_pose(frame_bgr, result):
    # Rysuje szkielet na obrazie, jeśli wykryto pozę
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr, # obraz wejściowy
            result.pose_landmarks, # punkty na ciele z kamerki
            mp_pose.POSE_CONNECTIONS # połącznia między tymi punktami
        )
    return frame_bgr

def landmarks_to_np(result, frame_shape):
    """
    Zwraca tablicę (x, y, visibility)
    Jeśli brak ->>> None
    """
    if not result.pose_landmarks:
        return None
    
    # Wymiary obrazu
    h, w = frame_shape[:2]

    lm = []
    for p in result.pose_landmarks.landmark:
        """
        MediaPipe zwraca współrzędne landmarków znormalizowane ->> [0, 1]
        Lewy górny róg obrazu ma współrzędne (0.0, 0.0).
        Przeliczamy je na piksele, czyli jak otrzymamy:
            p.x = 0.5
            p.y = 0.25
        To znaczy (dla obraxu 640 x 480):
            x = 0.5 * 640 = 320 px
            y = 0.25 * 480 = 120 px
        """
        lm.append([p.x * w, p.y * h, p.visibility])
    return np.array(lm, dtype=np.float32)
