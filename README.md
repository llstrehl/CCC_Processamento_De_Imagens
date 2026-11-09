# Contador e Classificador de Grãos

Trabalho prático de Processamento de Imagens e Visão Computacional (PIVC 2026/2).

## Objetivo
Aplicação com interface gráfica que, a partir de uma foto de grãos sobre superfície clara:
1. Conta os grãos (separando os encostados com Watershed).
2. Mede o tamanho médio em mm usando uma moeda como referência de escala.
3. Classifica cada grão como inteiro ou quebrado/defeituoso (CNN).
4. Gera um resumo do lote: total, % de defeituosos e tamanho médio.

## Stack
- Python 3.10+
- OpenCV (segmentação, contornos, medição)
- TensorFlow/Keras (classificador CNN, treinado no Colab)
- GUI: Tkinter

## Estrutura
```
src/segmentation.py   Watershed e contornos
src/scale.py          detecção da moeda e conversão px -> mm
src/classifier.py     classificação inteiro x defeituoso
src/summary.py        resumo do lote
src/gui.py            interface gráfica
notebooks/            treinamento no Colab
data/                 dataset (não versionado)
models/               modelos treinados (não versionados)
docs/                 artigo, relatório, vídeo
```

## Como rodar
```
pip install -r requirements.txt
python -m src.gui
```

## Condições da foto
Vista de cima, fundo claro e liso, sem sombras fortes, moeda visível no enquadramento.

## Equipe e divisão de tarefas
_A definir pelo grupo._

## Prazos
Entrega: 30/11.
