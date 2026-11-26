"""Testes automatizados do contador e classificador de grãos.

As fotos usadas aqui são geradas por ``tests/synthetic.py``, então o teste
sabe de antemão quantos grãos existem e quantos deveriam ser classificados
como defeituosos. Rodar com:

    pytest -v
"""

import cv2
import numpy as np
import pytest

from src import classifier as clf
from src import config
from src import pipeline
from src import scale as scale_mod
from src import segmentation as seg
from src import summary as summary_mod
from tests.synthetic import make_image


# --------------------------------------------------------------------------
# Segmentação
# --------------------------------------------------------------------------
def test_conta_todos_os_graos():
    imagem, total, _ = make_image(n_inteiros=12, n_quebrados=4, seed=7)
    resultado = pipeline.analyze(imagem)
    assert resultado["summary"]["total"] == total


def test_separa_inteiros_de_defeituosos():
    imagem, _, defeituosos = make_image(n_inteiros=12, n_quebrados=4, seed=7)
    resultado = pipeline.analyze(imagem)
    assert resultado["summary"]["defeituosos"] == defeituosos


def test_moeda_nao_entra_na_contagem():
    """A moeda é um objeto escuro grande: sem o descarte, viraria um grão."""
    com_moeda, total, _ = make_image(n_inteiros=10, n_quebrados=3, seed=19)
    assert pipeline.analyze(com_moeda)["summary"]["total"] == total


def test_detecta_fundo_claro_e_escuro():
    imagem, _, _ = make_image(seed=7)
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    assert seg.has_light_background(gray) is True
    assert seg.has_light_background(255 - gray) is False


def test_graos_quebrados_sobrevivem_ao_limiar_local():
    """Grão quebrado tem metade da espessura e sumiria com limiar global.

    Aqui o lote é propositalmente dominado por pedaços: o que se verifica é a
    segmentação, ou seja, que nenhum grão fino seja descartado. A classificação
    desse lote é checada em outro teste, porque o critério clássico parte de
    outra premissa (ver abaixo).
    """
    imagem, total, _ = make_image(n_inteiros=4, n_quebrados=8, seed=31)
    assert pipeline.analyze(imagem)["summary"]["total"] == total


def test_baseline_assume_maioria_de_graos_inteiros():
    """Documenta a premissa do critério clássico, para ela não surpreender.

    O baseline compara cada grão com a área mediana do lote. Num lote em que
    os pedaços são maioria, a mediana passa a ser a do pedaço e eles deixam de
    destoar — o resultado sai quase sem defeituosos. É o caso em que vale a
    pena treinar a CNN, que olha a forma de cada grão e não depende do lote.
    """
    maioria_inteira, _, defeituosos = make_image(n_inteiros=12, n_quebrados=4, seed=31)
    assert pipeline.analyze(maioria_inteira)["summary"]["defeituosos"] == defeituosos

    maioria_quebrada, _, muitos = make_image(n_inteiros=4, n_quebrados=8, seed=31)
    assert pipeline.analyze(maioria_quebrada)["summary"]["defeituosos"] < muitos


def test_reduz_foto_grande():
    grande = np.zeros((2000, 4000, 3), np.uint8)
    reduzida, fator = seg.resize_to_width(grande)
    assert reduzida.shape[1] == config.MAX_WIDTH_PX
    assert fator < 1.0


# --------------------------------------------------------------------------
# Escala
# --------------------------------------------------------------------------
def test_escala_bate_com_a_moeda():
    """Moeda de 100 px de raio e 27 mm de diâmetro dá 7,41 px/mm."""
    assert scale_mod.px_per_mm(100, 27.0) == pytest.approx(200 / 27.0)


def test_tamanho_do_grao_em_milimetros():
    escala = scale_mod.px_per_mm(100, 27.0)
    quadrado = np.array([[[400, 175]], [[450, 175]], [[450, 225]], [[400, 225]]])
    assert scale_mod.grain_size_mm(quadrado, escala) == pytest.approx(50 / escala)


def test_escala_recusada_sem_moeda():
    """Sem moeda, nenhum grão pode ser promovido a referência de escala."""
    imagem, _, _ = make_image(com_moeda=False, seed=7)
    resultado = pipeline.analyze(imagem)
    assert resultado["escala_ok"] is False
    assert resultado["px_per_mm"] is None


def test_tamanho_medio_plausivel_para_soja():
    """Grão de 44 px com escala de ~7,4 px/mm tem de dar algo perto de 6 mm."""
    imagem, _, _ = make_image(n_inteiros=14, n_quebrados=0, seed=7)
    resultado = pipeline.analyze(imagem)
    assert 4.0 < resultado["summary"]["tamanho_medio_mm"] < 8.0


def test_px_per_mm_rejeita_valores_invalidos():
    with pytest.raises(ValueError):
        scale_mod.px_per_mm(100, 0)
    with pytest.raises(ValueError):
        scale_mod.px_per_mm(0, 27.0)


# --------------------------------------------------------------------------
# Classificação
# --------------------------------------------------------------------------
def _contorno_circulo(raio, ate_graus=360):
    canvas = np.zeros((120, 120), np.uint8)
    cv2.ellipse(canvas, (60, 60), (raio, raio), 0, 0, ate_graus, 255, -1)
    contornos, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos[0]


def test_baseline_marca_o_grao_menor_do_lote():
    """O baseline compara cada grão com a mediana do lote.

    Por isso o teste monta um lote realista — vários inteiros e um pedaço —
    em vez de comparar dois contornos soltos: com apenas duas amostras a
    mediana cai no meio do caminho e nenhum dos dois destoa.
    """
    inteiros = [_contorno_circulo(40) for _ in range(6)]
    quebrado = _contorno_circulo(40, ate_graus=180)

    rotulos = clf.classify_all_baseline(inteiros + [quebrado])

    assert rotulos[:-1] == [config.LABEL_INTEIRO] * 6
    assert rotulos[-1] == config.LABEL_DEFEITUOSO


def test_baseline_marca_contorno_irregular():
    """Grão com pedaço faltando tem solidez baixa mesmo com área parecida."""
    inteiros = [_contorno_circulo(40) for _ in range(5)]

    mordido = np.zeros((120, 120), np.uint8)
    cv2.circle(mordido, (60, 60), 40, 255, -1)
    cv2.circle(mordido, (60, 20), 22, 0, -1)
    contornos, _ = cv2.findContours(mordido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rotulos = clf.classify_all_baseline(inteiros + [contornos[0]])
    assert rotulos[-1] == config.LABEL_DEFEITUOSO


def test_recorte_tem_o_tamanho_da_entrada_da_rede():
    imagem, _, _ = make_image(n_inteiros=6, n_quebrados=0, seed=7)
    contornos, _ = seg.segment_grains(imagem)
    recorte = clf.crop_grain(imagem, contornos[0])
    assert recorte.shape == (config.CROP_SIZE, config.CROP_SIZE, 3)


def test_classificador_cai_no_baseline_sem_modelo():
    classificador = clf.CNNClassifier(model_path="models/modelo_que_nao_existe.keras")
    assert classificador.model is None
    imagem, total, _ = make_image(n_inteiros=8, n_quebrados=2, seed=7)
    resultado = pipeline.analyze(imagem, classifier=classificador)
    assert resultado["summary"]["total"] == total
    assert "baseline" in resultado["modo"]


# --------------------------------------------------------------------------
# Resumo
# --------------------------------------------------------------------------
def test_resumo_calcula_percentual():
    resumo = summary_mod.batch_summary([0, 0, 1, 0], [6.0, 7.0, 3.5, 6.5])
    assert resumo["total"] == 4
    assert resumo["defeituosos"] == 1
    assert resumo["pct_defeituosos"] == pytest.approx(25.0)
    assert resumo["tamanho_medio_mm"] == pytest.approx(5.75)


def test_resumo_de_lote_vazio_nao_quebra():
    resumo = summary_mod.batch_summary([], [])
    assert resumo["total"] == 0
    assert resumo["pct_defeituosos"] == 0.0


def test_resumo_recusa_listas_de_tamanhos_diferentes():
    with pytest.raises(ValueError):
        summary_mod.batch_summary([0, 1], [5.0])


# --------------------------------------------------------------------------
# Erros tratados
# --------------------------------------------------------------------------
def test_foto_sem_graos_da_erro_explicado():
    vazia = np.full((600, 800, 3), 240, np.uint8)
    with pytest.raises(pipeline.AnalysisError):
        pipeline.analyze(vazia)


def test_arquivo_inexistente_da_erro_explicado():
    with pytest.raises((pipeline.AnalysisError, FileNotFoundError, OSError)):
        pipeline.analyze_file("nao_existe_essa_foto.jpg")
