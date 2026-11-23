"""Resumo do lote: total, % defeituosos e tamanho médio."""


def batch_summary(labels: list, sizes_mm: list) -> dict:
    """Retorna {'total': int, 'pct_defeituosos': float, 'tamanho_medio_mm': float}."""
    
    if len(labels) != len(sizes_mm):
        raise ValueError("labels e sizes_mm devem ter o mesmo tamanho")
    
    total = len(labels)
    defeituosos = sum(1 for label in labels if label == 1)
    pct_defeituosos = (defeituosos / total * 100) if total > 0 else 0
    tamanho_medio_mm = sum(sizes_mm) / total if total > 0 else 0

    return {
        'total': total,
        'pct_defeituosos': pct_defeituosos,
        'tamanho_medio_mm': tamanho_medio_mm
    }


if __name__ == "__main__":
    print(batch_summary([0, 0, 1, 0], [6.0, 7.0, 3.5, 6.5]))
    # esperado: total 4, 25.0 %, media 5.75
    print(batch_summary([], []))   # ceros, sin error