"""Segmentação de grãos: binarização, Watershed e extração de contornos."""
import numpy as np


def segment_grains(image: np.ndarray):
    """Separa grãos (inclusive encostados) via Watershed.

    Retorna: (markers, contours) com um contorno por grão.
    """
    raise NotImplementedError


def crop_grains(image: np.ndarray, contours) -> list:
    """Recorta cada grão da imagem original (entrada da CNN)."""
    raise NotImplementedError
