"""Escala: detecta a moeda de referência e converte pixels em milímetros.

A moeda aparece na foto ao lado dos grãos. Como o diâmetro real dela é
conhecido, o raio medido em pixels dá a razão pixels/mm da foto inteira.
"""

import cv2
import numpy as np

from . import config


def detect_coin(image, min_radius=config.COIN_MIN_RADIUS,
                max_radius=config.COIN_MAX_RADIUS):
    """Procura a moeda na imagem e devolve ``(x, y, raio)`` em pixels.

    Devolve ``None`` quando nenhum círculo é encontrado. Entre os círculos
    detectados, assume que a moeda é o de maior raio — ela é bem maior que
    qualquer grão.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 8,
        param1=config.COIN_PARAM1, param2=config.COIN_PARAM2,
        minRadius=min_radius, maxRadius=max_radius,
    )

    if circles is None:
        return None

    x, y, r = max(circles[0], key=lambda c: c[2])
    return float(x), float(y), float(r)


def validate_coin(coin, contours, min_ratio=2.0):
    """Confere se o círculo detectado é mesmo a moeda, e não um grão.

    A moeda precisa ter área pelo menos ``min_ratio`` vezes maior que a área
    mediana dos grãos. Sem essa checagem, uma foto sem moeda faria o maior
    grão virar referência de escala e toda medida em mm sairia errada.
    """
    if coin is None or not contours:
        return False

    _, _, r = coin
    coin_area = np.pi * r * r
    median_area = float(np.median([cv2.contourArea(c) for c in contours]))
    if median_area <= 0:
        return False
    return coin_area >= min_ratio * median_area


def px_per_mm(coin_radius_px, coin_diameter_mm=config.COIN_DIAMETER_MM):
    """Quantos pixels da foto correspondem a 1 mm."""
    if not coin_diameter_mm or coin_diameter_mm <= 0:
        raise ValueError("coin_diameter_mm precisa ser um número maior que zero")
    if coin_radius_px <= 0:
        raise ValueError("coin_radius_px precisa ser maior que zero")
    return (2.0 * coin_radius_px) / coin_diameter_mm


def grain_size_mm(contour, scale):
    """Comprimento do maior eixo do grão, em milímetros."""
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return max(w, h) / scale


def grain_width_mm(contour, scale):
    """Largura (menor eixo) do grão, em milímetros."""
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    return min(w, h) / scale


if __name__ == "__main__":
    # Teste rápido sem foto real: um círculo preto de 200 px de diâmetro
    # sobre fundo branco deve ser detectado com raio ~100 px.
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.circle(img, (150, 200), 100, (0, 0, 0), -1)
    print("moeda detectada:", detect_coin(img))

    escala = px_per_mm(100, 27.0)          # 200 px / 27 mm = 7,41 px/mm
    quadrado = np.array([[[400, 175]], [[450, 175]], [[450, 225]], [[400, 225]]])
    print("tamanho do quadrado de 50 px:", round(grain_size_mm(quadrado, escala), 2), "mm")
