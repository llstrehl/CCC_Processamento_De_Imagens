"""Fluxo completo da análise de um lote de grãos.

Liga as quatro etapas numa chamada só: segmentação, escala pela moeda,
classificação e resumo. É o que a interface gráfica e a linha de comando
usam, para que ambas produzam exatamente o mesmo resultado.
"""

import cv2
import numpy as np

from . import classifier as clf
from . import config
from . import scale as scale_mod
from . import segmentation as seg
from . import summary as summary_mod


class AnalysisError(Exception):
    """Erro previsível da análise, com mensagem pronta para mostrar ao usuário."""


def load_image(path):
    """Lê a foto do disco, aceitando caminhos com acento e espaço."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise AnalysisError(f"Não consegui ler o arquivo: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise AnalysisError(f"O arquivo não é uma imagem válida: {path}")
    return image


def analyze(image, coin_name=config.DEFAULT_COIN, classifier=None,
            light_background=None, use_coin=True):
    """Analisa uma foto de grãos já carregada em memória.

    Devolve um dicionário com a imagem anotada, os contornos, os rótulos, os
    tamanhos em mm, o resumo do lote e o texto formatado.
    """
    if image is None:
        raise AnalysisError("Nenhuma imagem foi carregada.")

    image, _ = seg.resize_to_width(image)

    # 1. Moeda de referência -------------------------------------------------
    coin = scale_mod.detect_coin(image) if use_coin else None

    # 2. Segmentação ---------------------------------------------------------
    contours, _ = seg.segment_grains(image, light_background=light_background,
                                     ignore_circle=coin)

    if not contours:
        raise AnalysisError(
            "Nenhum grão foi encontrado. Verifique se a foto tem contraste "
            "entre os grãos e a superfície e se está em foco."
        )

    # A moeda só vale como referência se for bem maior que os grãos
    escala_ok = use_coin and scale_mod.validate_coin(coin, contours)
    if not escala_ok:
        coin = None

    # 3. Escala e tamanhos ---------------------------------------------------
    if escala_ok:
        px_mm = scale_mod.px_per_mm(coin[2], config.COINS_MM[coin_name])
        sizes_mm = [scale_mod.grain_size_mm(c, px_mm) for c in contours]
    else:
        px_mm = None
        sizes_mm = [0.0] * len(contours)

    # 4. Classificação -------------------------------------------------------
    if classifier is None:
        classifier = clf.CNNClassifier()
    labels = classifier.classify(image, contours)

    # 5. Resumo --------------------------------------------------------------
    resumo = summary_mod.batch_summary(labels, sizes_mm)
    texto = summary_mod.format_summary(resumo, escala_ok=escala_ok,
                                       modo=classifier.modo)

    anotada = seg.draw_contours(image, contours, labels=labels, coin=coin)

    return {
        "image": image,
        "annotated": anotada,
        "contours": contours,
        "labels": labels,
        "sizes_mm": sizes_mm,
        "coin": coin,
        "px_per_mm": px_mm,
        "escala_ok": escala_ok,
        "summary": resumo,
        "summary_text": texto,
        "modo": classifier.modo,
    }


def analyze_file(path, **kwargs):
    """Mesma análise, começando pelo caminho do arquivo."""
    return analyze(load_image(path), **kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("uso: python -m src.pipeline <caminho da foto>")
        raise SystemExit(1)

    try:
        resultado = analyze_file(sys.argv[1])
    except AnalysisError as erro:
        print("erro:", erro)
        raise SystemExit(1)

    print(resultado["summary_text"])
    cv2.imwrite("resultado.png", resultado["annotated"])
    print("imagem anotada salva em resultado.png")
