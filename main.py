"""Ponto de entrada da aplicação.

    python main.py                 abre a interface gráfica
    python main.py foto.jpg        analisa a foto pelo terminal
"""

import sys


def main():
    if len(sys.argv) > 1:
        from src import pipeline

        try:
            resultado = pipeline.analyze_file(sys.argv[1])
        except pipeline.AnalysisError as erro:
            print("erro:", erro)
            return 1

        print(resultado["summary_text"])

        import cv2
        cv2.imwrite("resultado.png", resultado["annotated"])
        print("imagem anotada salva em resultado.png")
        return 0

    from src import gui
    gui.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
