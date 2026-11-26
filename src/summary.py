"""Resumo do lote: total, % de defeituosos e tamanho médio."""

from . import config


def batch_summary(labels, sizes_mm):
    """Resume o lote analisado.

    Devolve um dicionário com total de grãos, quantidade e percentual de
    defeituosos, tamanho médio, mínimo e máximo em milímetros.
    """
    if len(labels) != len(sizes_mm):
        raise ValueError("labels e sizes_mm devem ter o mesmo tamanho")

    total = len(labels)
    defeituosos = sum(1 for label in labels if label == config.LABEL_DEFEITUOSO)
    inteiros = total - defeituosos

    validos = [s for s in sizes_mm if s is not None]

    return {
        "total": total,
        "inteiros": inteiros,
        "defeituosos": defeituosos,
        "pct_defeituosos": (defeituosos / total * 100) if total else 0.0,
        "tamanho_medio_mm": (sum(validos) / len(validos)) if validos else 0.0,
        "tamanho_min_mm": min(validos) if validos else 0.0,
        "tamanho_max_mm": max(validos) if validos else 0.0,
    }


def format_summary(resumo, escala_ok=True, modo=""):
    """Transforma o dicionário do resumo no texto exibido na interface."""
    linhas = [
        f"Total de grãos: {resumo['total']}",
        f"Inteiros: {resumo['inteiros']}",
        f"Defeituosos: {resumo['defeituosos']} ({resumo['pct_defeituosos']:.1f}%)",
    ]

    if escala_ok:
        linhas += [
            f"Tamanho médio: {resumo['tamanho_medio_mm']:.2f} mm",
            f"Menor grão: {resumo['tamanho_min_mm']:.2f} mm",
            f"Maior grão: {resumo['tamanho_max_mm']:.2f} mm",
        ]
    else:
        linhas.append("Tamanho: moeda de referência não encontrada na foto")

    if modo:
        linhas.append(f"Classificação por: {modo}")

    return "\n".join(linhas)


if __name__ == "__main__":
    print(batch_summary([0, 0, 1, 0], [6.0, 7.0, 3.5, 6.5]))
    print(batch_summary([], []))
