"""Segmentação dos grãos por Watershed.

A partir de uma foto de grãos sobre uma superfície de cor contrastante,
devolve um contorno por grão, separando inclusive os que estão encostados.
"""

import cv2
import numpy as np

from . import config


def resize_to_width(image, max_width=config.MAX_WIDTH_PX, min_width=config.MIN_WIDTH_PX):
    """Reduz a imagem se ela for mais larga que ``max_width``.

    Devolve ``(imagem, fator)``, em que ``fator`` é quanto a imagem foi
    reduzida (1.0 quando não houve redução). Fotos de celular têm 3000 px ou
    mais de largura e o Watershed fica lento sem essa redução.
    Imagens pequenas (bajadas da internet) são ampliadas: com grãos de poucos pixels, a
    morfologia apagaria os grãos antes do Watershed.
    """
    height, width = image.shape[:2]
    if width > max_width:
        target, interpolation = max_width, cv2.INTER_AREA
    elif width < min_width:
        target, interpolation = min_width, cv2.INTER_CUBIC
    else:
        return image, 1.0
    factor = target / float(width)
    new_size = (target, int(round(height * factor)))
    return cv2.resize(image, new_size, interpolation=interpolation), factor


def has_light_background(gray, border=10):
    """Decide se o fundo da foto é claro olhando a borda da imagem.

    Calcula o limiar de Otsu, que separa objetos de fundo, e verifica de que
    lado desse limiar cai a moldura da foto — a borda é quase sempre
    superfície, não grão. Evita que o usuário precise informar manualmente se
    fotografou sobre fundo claro ou escuro.
    """
    height, width = gray.shape[:2]
    border = max(1, min(border, height // 4, width // 4))
    moldura = np.concatenate([
        gray[:border, :].ravel(),
        gray[-border:, :].ravel(),
        gray[:, :border].ravel(),
        gray[:, -border:].ravel(),
    ])
    limiar, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(np.median(moldura)) > float(limiar)


def build_mask(image, light_background=None):
    """Monta a máscara binária em que os grãos ficam brancos."""
    # Suaviza regiões de cor parecida preservando as bordas (reduz ruído)
    shifted = cv2.pyrMeanShiftFiltering(image, config.MEAN_SHIFT_SP, config.MEAN_SHIFT_SR)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if light_background is None:
        light_background = has_light_background(gray)

    # Fundo claro: os grãos são os pixels escuros, então inverte o limiar.
    modo = cv2.THRESH_BINARY_INV if light_background else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, 0, 255, modo | cv2.THRESH_OTSU)
    return thresh, light_background


def segment_grains(image, light_background=None, dist_factor=config.DIST_FACTOR,
                   min_area=config.MIN_AREA_PX, ignore_circle=None):
    """Separa os grãos da foto e devolve ``(contornos, markers)``.

    ``light_background`` força o tipo de fundo; ``None`` detecta sozinho.
    ``ignore_circle`` é uma tupla ``(x, y, r)`` — normalmente a moeda de
    referência — cujo contorno é descartado da lista de grãos.
    """
    thresh, _ = build_mask(image, light_background)

    # A moeda sai da máscara antes de qualquer coisa. O limiar da transformada
    # de distância é uma fração do valor MÁXIMO da imagem: deixar a moeda ali
    # elevaria esse máximo ao raio dela e apagaria todos os grãos, que são
    # muito menores.
    if ignore_circle is not None:
        x, y, r = ignore_circle
        cv2.circle(thresh, (int(round(x)), int(round(y))),
                   int(round(r * 1.1)) + 2, 0, -1)

    kernel = np.ones((config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)

    # Elimina manchas de ruído branco
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                               iterations=config.MORPH_ITERATIONS)

    # Dilata os grãos: o que ficar fora disso é fundo com certeza
    sure_bg = cv2.dilate(opening, kernel, iterations=config.MORPH_ITERATIONS)

    # Distância de cada pixel ao fundo; sobra só o centro de cada grão
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    sure_fg = _local_foreground(opening, dist_transform, dist_factor)

    # Franja onde não se sabe se é fundo ou grão
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Enumera os marcadores; 0 são os pixels desconhecidos
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Separa com -1 as fronteiras entre grãos encostados
    markers = cv2.watershed(image, markers)

    contours = []
    for label in np.unique(markers):
        if label <= 1:          # -1 bordas, 0 não segmentado, 1 fundo
            continue
        mask = np.uint8(markers == label) * 255
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            continue
        contour = max(found, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue
        if ignore_circle is not None and _inside_circle(contour, ignore_circle):
            continue
        contours.append(contour)

    return contours, markers


def _local_foreground(mask, dist_transform, dist_factor):
    """Marca o miolo de cada grão usando um limiar por mancha, não global.

    O tutorial clássico do Watershed usa ``dist_factor * dist.max()`` sobre a
    imagem inteira. Isso funciona quando todos os objetos têm o mesmo tamanho,
    mas aqui um grão quebrado tem metade da espessura de um inteiro e seria
    apagado pelo limiar — justamente o grão que o trabalho precisa contar.
    Calculando o limiar dentro de cada mancha conectada, cada grão é medido na
    própria escala, e manchas com grãos encostados continuam sendo separadas.
    """
    quantidade, componentes = cv2.connectedComponents(mask)
    sure_fg = np.zeros(mask.shape, np.uint8)

    for label in range(1, quantidade):
        regiao = componentes == label
        maximo = float(dist_transform[regiao].max())
        if maximo <= 0:
            continue
        sure_fg[regiao & (dist_transform >= dist_factor * maximo)] = 255

    return sure_fg


def _inside_circle(contour, circle, tolerance=0.6):
    """Diz se o centro do contorno cai dentro do círculo informado.

    Usado para tirar a moeda de referência da contagem de grãos: sem isso ela
    entraria no lote como se fosse mais um grão, inflando o total e a média.
    """
    x, y, r = circle
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return False
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    distance = np.hypot(cx - x, cy - y)
    return distance <= r * (1.0 + tolerance) and distance <= r * 1.6


def draw_contours(image, contours, labels=None, coin=None):
    """Desenha os contornos sobre uma cópia da imagem, colorindo por classe."""
    output = image.copy()

    if coin is not None:
        x, y, r = coin
        cv2.circle(output, (int(x), int(y)), int(r), config.COLOR_COIN, 2)
        cv2.putText(output, "moeda", (int(x - r), int(y - r - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_COIN, 1, cv2.LINE_AA)

    for index, contour in enumerate(contours):
        if labels is None:
            color = config.COLOR_INTEIRO
        else:
            color = (config.COLOR_DEFEITUOSO
                     if labels[index] == config.LABEL_DEFEITUOSO
                     else config.COLOR_INTEIRO)
        cv2.drawContours(output, [contour], -1, color, 2)

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("uso: python -m src.segmentation <caminho da foto>")
        raise SystemExit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("não consegui abrir a imagem:", sys.argv[1])
        raise SystemExit(1)

    img, _ = resize_to_width(img)
    cnts, _ = segment_grains(img)
    print(len(cnts), "grãos encontrados")
    cv2.imwrite("resultado_segmentacao.png", draw_contours(img, cnts))
    print("saída salva em resultado_segmentacao.png")
