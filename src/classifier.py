"""Classificação de cada grão: inteiro x quebrado/defeituoso."""
import numpy as np
import cv2

#Esto va en config
LABEL_INTEIRO = 0
LABEL_DEFEITUOSO = 1
UMBRAL_AREA_REL = 0.6 
UMBRAL_SOLIDEZ = 0.9  

def classify_baseline(contour, median_area: float) -> int:
    """Baseline clássico (área relativa e solidez). Retorna LABEL_INTEIRO ou LABEL_DEFEITUOSO."""
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))

    if hull_area == 0 or median_area == 0:
        return LABEL_DEFEITUOSO

    solidez = area / hull_area
    area_rel = area / median_area

    if area_rel < UMBRAL_AREA_REL or solidez < UMBRAL_SOLIDEZ:
        return LABEL_DEFEITUOSO
    return LABEL_INTEIRO
    

def load_model(path: str):
    """Carrega a CNN treinada (Keras/TFLite)."""
    raise NotImplementedError


def classify_cnn(model, grain_crop: np.ndarray) -> int:
    """Classifica um recorte de grão com a CNN."""
    raise NotImplementedError
