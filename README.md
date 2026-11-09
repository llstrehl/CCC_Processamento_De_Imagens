# Contador e Classificador de Grãos

Trabalho prático de Processamento de Imagens e Visão Computacional (PIVC 2026/2).

## Objetivo
Aplicação com interface gráfica que, a partir de uma foto de grãos sobre superfície clara:
1. Conta os grãos (separando os encostados com Watershed).
2. Mede o tamanho médio em mm usando uma moeda como referência de escala.
3. Classifica cada grão como inteiro ou quebrado/defeituoso (CNN).
4. Gera um resumo do lote: total, % de defeituosos e tamanho médio.

## Stack
Definido pelo grupo:
- OpenCV (segmentação, contornos, medição)
- TensorFlow (classificador CNN)
- Google Colab (apenas para treinar o modelo)
- Interface gráfica rodando localmente (exigência do enunciado)

A confirmar:
- Biblioteca da GUI
- Versão do Python
- Dependências extras 
- Tipo de grão
- Dataset 

## Estrutura (proposta)
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
A confirmar 

## Condições da foto

## Prazos
Entrega: 30/11.
