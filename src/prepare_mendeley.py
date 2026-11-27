"""Converte o dataset público "Soybean Seeds" (Mendeley) para o formato da CNN.

Dataset: https://data.mendeley.com/datasets/v6vzvfszj6
(Lin et al., 2023 — 5513 imagens de grãos de soja individuais, 227x227 px)

As imagens do dataset trazem um grão por arquivo, ainda com o fundo. A CNN da
aplicação, porém, recebe o recorte produzido por ``classifier.crop_grain``:
fundo zerado pela máscara do contorno, quadrado e com 64x64 px.
Este script passa cada imagem pela mesma máscara e pelo mesmo recorte da aplicação.

Só duas das cinco classes do dataset são usadas:

    Intact  -> inteiro
    Broken  -> defeituoso
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from . import classifier as clf
from . import config
from . import segmentation as seg

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Palavra que identifica a pasta de cada classe dentro do dataset. A busca é
# por trecho do nome, sem diferenciar maiúsculas, porque o nome exato das
# pastas muda entre versões do zip ("Intact", "Intact soybeans", ...).
CLASSES = {
    "intact": config.LABEL_NAMES[config.LABEL_INTEIRO],
    "broken": config.LABEL_NAMES[config.LABEL_DEFEITUOSO],
}


def encontrar_pastas(raiz):
    """Localiza, dentro do dataset, a pasta de cada classe usada."""
    raiz = Path(raiz)
    encontradas = {}
    for pasta in sorted(p for p in raiz.rglob("*") if p.is_dir()):
        nome = pasta.name.lower()
        for chave, classe in CLASSES.items():
            if chave in nome and classe not in encontradas:
                encontradas[classe] = pasta
    return encontradas


def contorno_do_grao(imagem):
    """Devolve o contorno do grão de uma imagem com um grão só.

    Usa a mesma máscara da segmentação da aplicação (mean shift + Otsu, com
    detecção automática do fundo). Não precisa de Watershed: há um grão por
    imagem, então basta ficar com o maior contorno.
    """
    mascara, _ = seg.build_mask(imagem)
    kernel = np.ones((config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    contorno = max(contornos, key=cv2.contourArea)

    # Descarta máscaras sem sentido: quase vazias (grão não encontrado) ou
    # ocupando a imagem inteira (o fundo foi tomado como grão).
    fracao = cv2.contourArea(contorno) / float(imagem.shape[0] * imagem.shape[1])
    if fracao < 0.02 or fracao > 0.95:
        return None
    return contorno


def converter_pasta(pasta_origem, pasta_destino, limite=None):
    """Converte as imagens de uma classe. Devolve ``(salvas, descartadas)``."""
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(p for p in Path(pasta_origem).rglob("*")
                      if p.suffix.lower() in EXTENSOES)
    if limite:
        arquivos = arquivos[:limite]

    salvas, descartadas = 0, 0
    for arquivo in arquivos:
        imagem = cv2.imread(str(arquivo), cv2.IMREAD_COLOR)
        if imagem is None:
            descartadas += 1
            continue

        contorno = contorno_do_grao(imagem)
        if contorno is None:
            descartadas += 1
            continue

        recorte = clf.crop_grain(imagem, contorno)
        cv2.imwrite(str(pasta_destino / f"mendeley_{arquivo.stem}.png"), recorte)
        salvas += 1

    return salvas, descartadas


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="pasta do dataset descompactado")
    parser.add_argument("--saida", default=str(config.DATA_DIR / "recortes_mendeley"),
                        help="pasta onde salvar os recortes (padrão: data/recortes_mendeley)")
    parser.add_argument("--limite", type=int, default=None,
                        help="máximo de imagens por classe (útil para um teste rápido)")
    argumentos = parser.parse_args()

    pastas = encontrar_pastas(argumentos.dataset)
    faltando = [c for c in CLASSES.values() if c not in pastas]
    if faltando:
        raise SystemExit(f"Não encontrei a pasta das classes {faltando} em {argumentos.dataset}. "
                         "Confira se o zip foi descompactado.")

    for classe, origem in pastas.items():
        destino = Path(argumentos.saida) / classe
        salvas, descartadas = converter_pasta(origem, destino, argumentos.limite)
        print(f"{classe}: {salvas} recortes salvos, {descartadas} descartados "
              f"(origem: {origem.name})")

    print(f"\nConfira algumas imagens em {argumentos.saida} antes de treinar.")


if __name__ == "__main__":
    main()
