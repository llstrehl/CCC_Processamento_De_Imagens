"""Escala: detecta a moeda de referência e converte pixels em mm."""
import numpy as np

COIN_DIAMETER_MM = 25.0  # ajustar conforme a moeda usada


def detect_coin(image: np.ndarray):
    """Detecta a moeda (ex.: HoughCircles). Retorna (x, y, raio_px)."""
    raise NotImplementedError


def px_per_mm(coin_radius_px: float, coin_diameter_mm: float = COIN_DIAMETER_MM) -> float:
    """Fator de conversão pixels -> mm."""
    raise NotImplementedError


def grain_size_mm(contour, scale: float) -> float:
    """Tamanho do grão em mm (comprimento do maior eixo)."""
    raise NotImplementedError
