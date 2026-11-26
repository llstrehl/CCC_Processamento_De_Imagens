"""Gerador de fotos sintéticas de grãos, usado nos testes automatizados.

Permite rodar o pipeline inteiro sem depender de fotos reais: a imagem é
construída com posições e rótulos conhecidos, então o teste sabe exatamente
quantos grãos e quantos defeituosos deveriam ser encontrados.
"""

import cv2
import numpy as np

FUNDO = 238           # superfície clara
COR_GRAO = (40, 85, 130)      # BGR, tom amarronzado de soja
COR_MOEDA = (150, 150, 150)   # cinza metálico


def make_image(n_inteiros=12, n_quebrados=4, coin_radius=100, seed=7,
               size=(900, 1200), grain_radius=22, com_moeda=True):
    """Monta uma foto sintética e devolve ``(imagem, total, defeituosos)``."""
    rng = np.random.default_rng(seed)
    altura, largura = size
    img = np.full((altura, largura, 3), FUNDO, np.uint8)
    ruido = rng.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    ocupados = []

    if com_moeda:
        centro_moeda = (coin_radius + 40, coin_radius + 40)
        cv2.circle(img, centro_moeda, coin_radius, COR_MOEDA, -1)
        cv2.circle(img, centro_moeda, coin_radius, (110, 110, 110), 3)
        ocupados.append((centro_moeda[0], centro_moeda[1], coin_radius + 20))

    def posicao_livre():
        for _ in range(500):
            x = int(rng.integers(grain_radius + 10, largura - grain_radius - 10))
            y = int(rng.integers(grain_radius + 10, altura - grain_radius - 10))
            if all((x - ox) ** 2 + (y - oy) ** 2 > (orr + grain_radius + 6) ** 2
                   for ox, oy, orr in ocupados):
                ocupados.append((x, y, grain_radius))
                return x, y
        return None

    for _ in range(n_inteiros):
        pos = posicao_livre()
        if pos is None:
            continue
        angulo = int(rng.integers(0, 180))
        eixos = (grain_radius, int(grain_radius * 0.88))
        cv2.ellipse(img, pos, eixos, angulo, 0, 360, COR_GRAO, -1)

    for _ in range(n_quebrados):
        pos = posicao_livre()
        if pos is None:
            continue
        angulo = int(rng.integers(0, 180))
        eixos = (grain_radius, int(grain_radius * 0.88))
        # Meio grão: a elipse é desenhada só até 180 graus
        cv2.ellipse(img, pos, eixos, angulo, 0, 180, COR_GRAO, -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, n_inteiros + n_quebrados, n_quebrados


if __name__ == "__main__":
    imagem, total, defeituosos = make_image()
    cv2.imwrite("amostra_sintetica.png", imagem)
    print(f"amostra_sintetica.png — {total} grãos, {defeituosos} defeituosos")
