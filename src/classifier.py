"""Classificação de cada grão: inteiro x quebrado/defeituoso."""
import numpy as np


def classify_baseline(contour) -> str:
    """Baseline clássico (área, solidez, razão de aspecto). Retorna 'inteiro' ou 'defeituoso'."""
    raise NotImplementedError


def load_model(path: str):
    """Carrega a CNN treinada (Keras/TFLite)."""
    raise NotImplementedError


def classify_cnn(model, grain_crop: np.ndarray) -> str:
    """Classifica um recorte de grão com a CNN."""
    raise NotImplementedError
