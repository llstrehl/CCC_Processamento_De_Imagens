"""Escala: detecta a moeda de referência e converte pixels em mm."""
import numpy as np
import cv2
import math

COIN_DIAMETER_MM = None  # A CONFIRMAR: definir a moeda usada e seu diâmetro real em mm


def detect_coin(image, min_radius=20, max_radius=300):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]

        # Encontra os círculos
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, 
    			                param1=100, param2=30,
                                minRadius=min_radius, maxRadius=max_radius)

    if circles is None:
        return None

    x, y, r = max(circles[0], key = lambda c: c[2])
    return float(x), float(y), float(r) 


def px_per_mm(coin_radius_px: float, coin_diameter_mm: float = COIN_DIAMETER_MM) -> float:
    if coin_diameter_mm is None:
        raise ValueError("COIN_DIAMETER_MM must be set to a valid value.")
    return (2 * coin_radius_px) / coin_diameter_mm


def grain_size_mm(contour, scale: float) -> float:
    """Comprimento do maior eixo do grão, em mm."""
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return max(w, h) / scale


if __name__ == "__main__":
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.circle(img, (150, 200), 100, (0, 0, 0), -1)   # "moneda" de 200 px
    print(detect_coin(img))                           # ~ (150, 200, 100)
    s = px_per_mm(100, 27)                            # 200/27 = 7.41 px/mm
    c = np.array([[[400,175]],[[450,175]],[[450,225]],[[400,225]]])  # cuadrado de 50 px
    print(grain_size_mm(c, s))                        # ~ 6.75 mm