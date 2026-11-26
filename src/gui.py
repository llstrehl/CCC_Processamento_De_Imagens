"""Interface gráfica do contador e classificador de grãos.

Fluxo da tela: o usuário escolhe a foto, confere qual moeda está na imagem,
clica em Analisar e vê a foto com os grãos marcados em verde (inteiros) e
vermelho (defeituosos), junto com o resumo do lote. Os resultados podem ser
salvos como imagem anotada e planilha CSV.
"""

import csv
import io
from pathlib import Path

import cv2
import FreeSimpleGUI as sg
import numpy as np
from PIL import Image

from . import classifier as clf
from . import config
from . import pipeline

TEMA = "SystemDefault1"
PREVIEW_MAX = (760, 560)
TIPOS_IMAGEM = (
    ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
    ("Todos os arquivos", "*.*"),
)


# --------------------------------------------------------------------------
# Conversão da imagem do OpenCV para o formato que o FreeSimpleGUI exibe
# --------------------------------------------------------------------------
def to_png_bytes(image_bgr, max_size=PREVIEW_MAX):
    """Converte uma imagem BGR do OpenCV em PNG redimensionado, em bytes."""
    if image_bgr is None:
        return None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail(max_size)

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def build_layout():
    coluna_controles = [
        [sg.Text("Foto do lote", font=("Segoe UI", 10, "bold"))],
        [
            sg.Input(key="-ARQUIVO-", size=(34, 1), enable_events=True, readonly=True),
            sg.FileBrowse("Abrir...", file_types=TIPOS_IMAGEM, target="-ARQUIVO-"),
        ],
        [sg.HorizontalSeparator()],
        [sg.Text("Moeda de referência na foto")],
        [
            sg.Combo(list(config.COINS_MM.keys()), default_value=config.DEFAULT_COIN,
                     key="-MOEDA-", readonly=True, enable_events=True, size=(14, 1)),
            sg.Text("", key="-MM-", size=(12, 1)),
        ],
        [sg.Checkbox("A foto tem moeda de referência", default=True, key="-USAR-MOEDA-")],
        [sg.HorizontalSeparator()],
        [sg.Button("Analisar", key="-ANALISAR-", size=(14, 1), button_color=("white", "#1f6f3f")),
         sg.Button("Limpar", key="-LIMPAR-", size=(10, 1))],
        [sg.HorizontalSeparator()],
        [sg.Text("Resumo do lote", font=("Segoe UI", 10, "bold"))],
        [sg.Multiline("", key="-RESUMO-", size=(40, 9), disabled=True,
                      no_scrollbar=True, background_color="#f4f4f4")],
        [sg.Button("Salvar imagem anotada", key="-SALVAR-IMG-", size=(24, 1), disabled=True)],
        [sg.Button("Salvar medidas em CSV", key="-SALVAR-CSV-", size=(24, 1), disabled=True)],
    ]

    coluna_imagem = [
        [sg.Text("", key="-TITULO-IMG-", font=("Segoe UI", 10, "bold"))],
        [sg.Image(key="-IMAGEM-", size=PREVIEW_MAX, background_color="#dddddd")],
    ]

    return [
        [
            sg.Column(coluna_controles, vertical_alignment="top"),
            sg.VerticalSeparator(),
            sg.Column(coluna_imagem, vertical_alignment="top"),
        ],
        [sg.StatusBar("Escolha uma foto para começar.", key="-STATUS-", size=(100, 1))],
    ]


# --------------------------------------------------------------------------
# Ações
# --------------------------------------------------------------------------
def salvar_csv(caminho, resultado):
    """Grava uma linha por grão com classe, comprimento e largura em mm."""
    from . import scale as scale_mod

    px_mm = resultado["px_per_mm"]
    with open(caminho, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(["grao", "classe", "comprimento_mm", "largura_mm", "area_px"])
        for indice, contorno in enumerate(resultado["contours"], start=1):
            classe = config.LABEL_NAMES[resultado["labels"][indice - 1]]
            if px_mm:
                comprimento = f"{scale_mod.grain_size_mm(contorno, px_mm):.2f}"
                largura = f"{scale_mod.grain_width_mm(contorno, px_mm):.2f}"
            else:
                comprimento = largura = ""
            escritor.writerow([indice, classe, comprimento, largura,
                               int(cv2.contourArea(contorno))])


def main():
    """Abre a janela principal."""
    sg.theme(TEMA)
    janela = sg.Window("Contador e Classificador de Grãos — PIVC 2026/2",
                       build_layout(), finalize=True, resizable=True)

    janela["-MM-"].update(f"{config.COINS_MM[config.DEFAULT_COIN]:.0f} mm")

    classificador = clf.CNNClassifier()
    janela["-STATUS-"].update(
        f"Pronto. Classificação por {classificador.modo}. Escolha uma foto."
    )

    resultado = None

    while True:
        evento, valores = janela.read()

        if evento in (sg.WIN_CLOSED, "Sair"):
            break

        if evento == "-MOEDA-":
            janela["-MM-"].update(f"{config.COINS_MM[valores['-MOEDA-']]:.0f} mm")

        elif evento == "-ARQUIVO-":
            caminho = valores["-ARQUIVO-"]
            if caminho:
                try:
                    imagem = pipeline.load_image(caminho)
                except pipeline.AnalysisError as erro:
                    sg.popup_error(str(erro), title="Erro ao abrir a foto")
                    continue
                janela["-IMAGEM-"].update(data=to_png_bytes(imagem))
                janela["-TITULO-IMG-"].update(Path(caminho).name)
                janela["-RESUMO-"].update("")
                janela["-STATUS-"].update("Foto carregada. Clique em Analisar.")
                janela["-SALVAR-IMG-"].update(disabled=True)
                janela["-SALVAR-CSV-"].update(disabled=True)
                resultado = None

        elif evento == "-ANALISAR-":
            caminho = valores["-ARQUIVO-"]
            if not caminho:
                sg.popup("Escolha uma foto antes de analisar.", title="Nenhuma foto")
                continue

            janela["-STATUS-"].update("Analisando...")
            janela.refresh()

            try:
                resultado = pipeline.analyze_file(
                    caminho,
                    coin_name=valores["-MOEDA-"],
                    classifier=classificador,
                    use_coin=valores["-USAR-MOEDA-"],
                )
            except pipeline.AnalysisError as erro:
                janela["-STATUS-"].update("A análise não foi concluída.")
                sg.popup_error(str(erro), title="Não foi possível analisar")
                continue

            janela["-IMAGEM-"].update(data=to_png_bytes(resultado["annotated"]))
            janela["-RESUMO-"].update(resultado["summary_text"])
            janela["-SALVAR-IMG-"].update(disabled=False)
            janela["-SALVAR-CSV-"].update(disabled=False)

            if resultado["escala_ok"]:
                janela["-STATUS-"].update(
                    f"{resultado['summary']['total']} grãos analisados — "
                    f"escala {resultado['px_per_mm']:.2f} px/mm."
                )
            else:
                janela["-STATUS-"].update(
                    f"{resultado['summary']['total']} grãos analisados — "
                    "sem moeda na foto, os tamanhos em mm não foram calculados."
                )

        elif evento == "-LIMPAR-":
            resultado = None
            janela["-ARQUIVO-"].update("")
            janela["-IMAGEM-"].update(data=None)
            janela["-TITULO-IMG-"].update("")
            janela["-RESUMO-"].update("")
            janela["-SALVAR-IMG-"].update(disabled=True)
            janela["-SALVAR-CSV-"].update(disabled=True)
            janela["-STATUS-"].update("Escolha uma foto para começar.")

        elif evento == "-SALVAR-IMG-" and resultado is not None:
            destino = sg.popup_get_file(
                "Salvar imagem anotada", save_as=True, no_window=True,
                default_extension=".png", file_types=(("PNG", "*.png"),),
            )
            if destino:
                cv2.imwrite(destino, resultado["annotated"])
                janela["-STATUS-"].update(f"Imagem salva em {destino}")

        elif evento == "-SALVAR-CSV-" and resultado is not None:
            destino = sg.popup_get_file(
                "Salvar medidas", save_as=True, no_window=True,
                default_extension=".csv", file_types=(("CSV", "*.csv"),),
            )
            if destino:
                salvar_csv(destino, resultado)
                janela["-STATUS-"].update(f"Medidas salvas em {destino}")

    janela.close()


if __name__ == "__main__":
    main()
