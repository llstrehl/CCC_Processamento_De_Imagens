"""Montagem do dataset de treino da CNN.

Percorre as fotos de lote, segmenta cada uma, recorta grão a grão e salva os
recortes já pré-separados pelo baseline clássico em duas pastas. O trabalho
humano vira apenas conferir e arrastar os recortes que o baseline errou, em
vez de recortar cada grão à mão.

Uso:

    python -m src.dataset data/fotos --saida data/recortes

Para fotos sem moeda (por exemplo, imagens da internet), use ``--sem-moeda``:
assim nenhum círculo falso é tomado como moeda e apaga os grãos.

    python -m src.dataset data/fotos_google --saida data/recortes_google --sem-moeda
"""

import argparse
from pathlib import Path

import cv2

from . import classifier as clf
from . import config
from . import scale as scale_mod
from . import segmentation as seg

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extrair_de_foto(caminho, pasta_saida, prefixo=None, usar_moeda=True):
    """Extrai os recortes de uma foto e devolve quantos foram salvos por classe.

    Com ``usar_moeda=False`` a detecção da moeda é pulada. Sem isso, numa foto
    sem moeda o HoughCircles pode achar um círculo falso grande, e os grãos
    dentro dele são apagados da máscara antes de a moeda ser validada.
    """
    from .pipeline import load_image

    imagem = load_image(caminho)
    imagem, _ = seg.resize_to_width(imagem)

    moeda = scale_mod.detect_coin(imagem) if usar_moeda else None
    contornos, _ = seg.segment_grains(imagem, ignore_circle=moeda)
    if not contornos:
        return {"inteiro": 0, "defeituoso": 0}

    if moeda is not None and not scale_mod.validate_coin(moeda, contornos):
        moeda = None
        contornos, _ = seg.segment_grains(imagem)

    rotulos = clf.classify_all_baseline(contornos)
    prefixo = prefixo or Path(caminho).stem
    contagem = {"inteiro": 0, "defeituoso": 0}

    for indice, (contorno, rotulo) in enumerate(zip(contornos, rotulos), start=1):
        classe = config.LABEL_NAMES[rotulo]
        destino = Path(pasta_saida) / classe
        destino.mkdir(parents=True, exist_ok=True)
        recorte = clf.crop_grain(imagem, contorno)
        cv2.imwrite(str(destino / f"{prefixo}_{indice:03d}.png"), recorte)
        contagem[classe] += 1

    return contagem


def extrair_pasta(pasta_fotos, pasta_saida, usar_moeda=True):
    """Extrai os recortes de todas as fotos de uma pasta."""
    pasta_fotos = Path(pasta_fotos)
    fotos = sorted(p for p in pasta_fotos.rglob("*") if p.suffix.lower() in EXTENSOES)

    if not fotos:
        print(f"Nenhuma foto encontrada em {pasta_fotos}")
        return {"inteiro": 0, "defeituoso": 0}

    total = {"inteiro": 0, "defeituoso": 0}
    for foto in fotos:
        contagem = extrair_de_foto(foto, pasta_saida, usar_moeda=usar_moeda)
        total["inteiro"] += contagem["inteiro"]
        total["defeituoso"] += contagem["defeituoso"]
        print(f"{foto.name}: {contagem['inteiro']} inteiros, "
              f"{contagem['defeituoso']} defeituosos")

    print(f"\nTotal: {total['inteiro']} inteiros e {total['defeituoso']} defeituosos "
          f"em {pasta_saida}")
    print("Confira as duas pastas e mova os recortes que o baseline classificou "
          "errado antes de treinar.")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fotos", help="pasta com as fotos de lote")
    parser.add_argument("--saida", default=str(config.DATA_DIR / "recortes"),
                        help="pasta onde salvar os recortes")
    parser.add_argument("--sem-moeda", action="store_true",
                        help="não procura moeda nas fotos (use em fotos sem moeda)")
    argumentos = parser.parse_args()
    extrair_pasta(argumentos.fotos, argumentos.saida,
                  usar_moeda=not argumentos.sem_moeda)


if __name__ == "__main__":
    main()