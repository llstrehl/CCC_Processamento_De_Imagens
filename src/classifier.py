"""Classificação de cada grão: inteiro x quebrado/defeituoso.

Dois caminhos:

* ``classify_baseline`` — critério clássico de visão computacional, baseado na
  área relativa do grão e na solidez do contorno. Não precisa de treino.
* ``CNNClassifier`` — rede convolucional treinada nos recortes dos próprios
  grãos. Quando o arquivo do modelo não existe, cai automaticamente no
  baseline, de modo que a aplicação nunca deixa de funcionar.
"""

import cv2
import numpy as np

from . import config

LABEL_INTEIRO = config.LABEL_INTEIRO
LABEL_DEFEITUOSO = config.LABEL_DEFEITUOSO


# --------------------------------------------------------------------------
# Baseline clássico
# --------------------------------------------------------------------------
def classify_baseline(contour, median_area,
                      thresh_area_rel=config.THRESH_AREA_REL,
                      thresh_solidez=config.THRESH_SOLIDEZ):
    """Classifica um grão por área relativa e solidez.

    Um grão quebrado tem área bem menor que a mediana do lote; um grão
    defeituoso ou com pedaço faltando tem contorno irregular, ou seja, área
    pequena em relação ao próprio fecho convexo (solidez baixa).
    """
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))

    if hull_area == 0 or median_area == 0:
        return LABEL_DEFEITUOSO

    solidez = area / hull_area
    area_rel = area / median_area

    if area_rel < thresh_area_rel or solidez < thresh_solidez:
        return LABEL_DEFEITUOSO
    return LABEL_INTEIRO


def classify_all_baseline(contours):
    """Aplica o baseline à lista inteira, calculando a área mediana do lote.

    Premissa deste critério: a maioria dos grãos do lote está inteira. É ela
    que torna a área mediana uma boa referência do que é um grão normal. Num
    lote em que os pedaços são maioria, a mediana passa a ser a do pedaço e
    quase nada é apontado como defeituoso — é exatamente o cenário em que a
    CNN, que olha a forma de cada grão isoladamente, faz diferença.
    """
    if not contours:
        return []
    median_area = float(np.median([cv2.contourArea(c) for c in contours]))
    return [classify_baseline(c, median_area) for c in contours]


# --------------------------------------------------------------------------
# Recorte de cada grão, usado para treinar e para rodar a CNN
# --------------------------------------------------------------------------
def crop_grain(image, contour, size=config.CROP_SIZE, margin=config.CROP_MARGIN):
    """Recorta um grão da foto e devolve um quadrado ``size`` x ``size``.

    O fundo em volta do grão é zerado pela máscara do contorno, para a rede
    aprender a forma do grão e não a cor da mesa.
    """
    x, y, w, h = cv2.boundingRect(contour)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(image.shape[1], x + w + margin)
    y1 = min(image.shape[0], y + h + margin)

    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), np.uint8)

    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    recorte = cv2.bitwise_and(image, image, mask=mask)[y0:y1, x0:x1]

    # Completa com preto até ficar quadrado, sem distorcer o grão
    altura, largura = recorte.shape[:2]
    lado = max(altura, largura)
    quadrado = np.zeros((lado, lado, 3), np.uint8)
    topo = (lado - altura) // 2
    esquerda = (lado - largura) // 2
    quadrado[topo:topo + altura, esquerda:esquerda + largura] = recorte

    return cv2.resize(quadrado, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------
def load_model(path=config.MODEL_PATH):
    """Carrega a CNN treinada. Devolve ``None`` se o arquivo não existir.

    O import do TensorFlow fica aqui dentro de propósito: ele demora alguns
    segundos e só é necessário quando existe modelo para carregar.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return None

    from tensorflow import keras
    return keras.models.load_model(path)


def classify_cnn(model, grain_crops):
    """Classifica uma lista de recortes com a CNN.

    Devolve a lista de rótulos (0 inteiro, 1 defeituoso). Faz a predição em
    lote, bem mais rápido que chamar o modelo grão a grão.
    """
    if model is None or len(grain_crops) == 0:
        return []

    # Dois cuidados aqui, ambos silenciosos quando errados — a rede não
    # reclama, só devolve a mesma resposta para todo grão:
    #
    # 1. Canais: o OpenCV trabalha em BGR, mas o treino lê os PNGs pelo
    #    utilitário do Keras, que entrega RGB. Sem converter, a rede recebe as
    #    cores trocadas em relação ao que aprendeu.
    # 2. Escala: nada de dividir por 255 aqui, porque a camada Rescaling
    #    dentro do próprio modelo já faz isso (ver src/train.py).
    rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in grain_crops]
    lote = np.stack(rgb).astype("float32")
    probabilidades = model.predict(lote, verbose=0)

    if probabilidades.shape[-1] == 1:
        return [LABEL_DEFEITUOSO if float(p) >= 0.5 else LABEL_INTEIRO
                for p in probabilidades.ravel()]
    return [int(np.argmax(p)) for p in probabilidades]


class CNNClassifier:
    """Classificador que usa a CNN quando há modelo e o baseline quando não há.

    ``modo`` informa, depois do primeiro uso, qual caminho foi seguido — é o
    que a interface mostra ao usuário.
    """

    def __init__(self, model_path=config.MODEL_PATH):
        self.model = load_model(model_path)
        self.modo = "CNN" if self.model is not None else "baseline clássico"

    def classify(self, image, contours):
        if not contours:
            return []

        if self.model is None:
            return classify_all_baseline(contours)

        recortes = [crop_grain(image, c) for c in contours]
        try:
            return classify_cnn(self.model, recortes)
        except Exception:
            # Se a inferência falhar na apresentação, o lote ainda é
            # classificado pelo baseline em vez de a aplicação quebrar.
            self.modo = "baseline clássico (a CNN falhou)"
            return classify_all_baseline(contours)


if __name__ == "__main__":
    circulo = np.zeros((100, 100), np.uint8)
    cv2.circle(circulo, (50, 50), 40, 255, -1)
    inteiro, _ = cv2.findContours(circulo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    metade = np.zeros((100, 100), np.uint8)
    cv2.ellipse(metade, (50, 50), (40, 40), 0, 0, 180, 255, -1)
    quebrado, _ = cv2.findContours(metade, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rotulos = classify_all_baseline([inteiro[0], quebrado[0]])
    print("esperado [0, 1], obtido", rotulos)
