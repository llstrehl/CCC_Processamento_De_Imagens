# Contador e Classificador de Grãos

Trabalho prático da disciplina de Processamento de Imagens e Visão Computacional
(PIVC 2026/2 — UPF). Grupo: Álan, Luiza e Francisco.

Artigo de referência: https://doi.org/10.1016/j.atech.2025.101388

## O que a aplicação faz

A partir de uma foto de grãos de soja espalhados sobre uma superfície clara, com
uma moeda ao lado servindo de régua, a aplicação:

1. **Conta os grãos**, separando com Watershed os que estão encostados uns nos outros.
2. **Mede cada grão em milímetros**, usando o diâmetro conhecido da moeda para
   converter pixels em medida real.
3. **Classifica cada grão** como inteiro ou quebrado/defeituoso, por uma CNN
   treinada nos recortes dos próprios grãos.
4. **Resume o lote**: total de grãos, percentual de defeituosos e tamanho médio,
   mínimo e máximo.

Os grãos inteiros aparecem contornados em verde, os defeituosos em vermelho e a
moeda em azul. O resultado pode ser salvo como imagem anotada e como planilha CSV
com uma linha por grão.

## Como rodar

Requer **Python 3.12**. Na pasta do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # Linux e macOS

pip install -r requirements.txt
python main.py
```

Para analisar uma foto direto pelo terminal, sem abrir a interface:

```bash
python main.py caminho/da/foto.jpg
```

## Como tirar a foto

As condições abaixo não são exigência da biblioteca, são o que faz a segmentação
acertar. Fora delas o programa continua rodando, mas a contagem erra mais.

- Superfície lisa e clara, de cor bem diferente da dos grãos. Uma folha A4 branca
  resolve.
- Grãos espalhados, de preferência sem montinhos. Encostados tudo bem: o Watershed
  separa. Empilhados um em cima do outro, não.
- Uma moeda brasileira no mesmo plano dos grãos, inteira dentro do quadro e sem
  encostar em nenhum grão. A de R$ 1,00 é a mais fácil de detectar por ser a maior.
- Câmera paralela à mesa, olhando de cima. Foto inclinada distorce a escala.
- Luz distribuída, sem flash direto e sem a sombra de quem fotografa sobre a cena.

Sem moeda na foto, basta desmarcar a opção na interface: a contagem e a
classificação continuam funcionando, só as medidas em milímetros deixam de sair.

## Treinar a CNN

O modelo treinado já está no repositório (`models/grain_cnn.keras`): basta clonar
e rodar a aplicação, que carrega a CNN sozinha. Quando o arquivo não existe, a
aplicação não quebra: ela cai no critério clássico, que separa os grãos por área
relativa e solidez do contorno. A barra de status mostra qual dos dois está em uso.

### Dados usados no treino

- **Dataset público Soybean Seeds** ([Mendeley](https://data.mendeley.com/datasets/v6vzvfszj6),
  Lin et al., 2023): grãos individuais fotografados com câmera industrial. Usamos
  as classes *Intact* (inteiro) e *Broken* (defeituoso), cerca de 2.200 recortes.
- **Imagens de soja da internet**, sobre fundo branco e escolhidas à mão: cerca de
  150 recortes, conferidos manualmente.

As imagens do dataset passam pelo mesmo recorte da aplicação (máscara do contorno,
fundo preto, 64x64) com `src/prepare_mendeley.py`, para que a rede treine com o
mesmo tipo de imagem que vê depois:

```bash
python -m src.prepare_mendeley "pasta/do/dataset" --saida data/recortes_mendeley
```

### Resultados

| Modelo | Treinado com | Validação | Teste com imagens externas |
| --- | --- | --- | --- |
| Só dataset | Soybean Seeds | 93,4% | 51,7% (marcou todos os grãos como defeituosos) |
| Dataset + internet (em uso) | Soybean Seeds + imagens externas | 95,9% | 100% (29 grãos) |

O modelo treinado só com o dataset não generalizava: as imagens são todas da mesma
câmera e da mesma luz. Somar imagens de outras fontes resolveu. O teste externo é
pequeno, e a validação definitiva fica para fotos tiradas pelo grupo.

### Treinar de novo

**1. Gerar os recortes** a partir de fotos de lote em `data/fotos`:

```bash
python -m src.dataset data/fotos --saida data/recortes
```

Cada grão vira uma imagem de 64x64, pré-separada nas pastas `inteiro` e
`defeituoso` pelo critério clássico; depois é só conferir e mover os que ficaram
errados. Para fotos sem moeda (por exemplo, imagens da internet), use `--sem-moeda`:
sem essa opção, um círculo falso pode ser tomado como moeda e apagar os grãos.

**2. Treinar**, localmente ou no Colab pelo notebook `notebooks/treino_cnn.ipynb`:

```bash
python -m src.train --dados data/recortes --saida models/grain_cnn.keras
```

Para o Colab, compacte a pasta `recortes` pelo Explorador do Windows (botão
direito → Enviar para → Pasta compactada). O zip criado pelo `Compress-Archive` do
PowerShell não é lido pelo notebook.
## Estrutura

| Arquivo | O que faz |
| --- | --- |
| `main.py` | Ponto de entrada: abre a interface ou analisa uma foto pelo terminal |
| `src/config.py` | Parâmetros do projeto: moeda, limiares, caminhos, cores |
| `src/segmentation.py` | Watershed, contornos e desenho dos resultados |
| `src/scale.py` | Detecção da moeda e conversão de pixels para milímetros |
| `src/classifier.py` | Critério clássico, recorte dos grãos e inferência da CNN |
| `src/summary.py` | Resumo do lote |
| `src/pipeline.py` | Liga as quatro etapas num fluxo só |
| `src/gui.py` | Interface gráfica em FreeSimpleGUI |
| `src/dataset.py` | Extrai os recortes de treino a partir das fotos |
| `src/train.py` | Treino da CNN |
| `notebooks/treino_cnn.ipynb` | Mesmo treino, para rodar no Colab |
| `tests/` | Testes automatizados |

## Decisões técnicas que valem explicação

**Limiar local na transformada de distância.** O tutorial clássico do Watershed
marca o miolo dos objetos com `0,52 × máximo da imagem inteira`. Isso funciona
quando todos os objetos têm o mesmo tamanho, mas aqui atrapalha duas vezes: a
moeda, muito maior que os grãos, eleva o máximo e apaga todo o resto; e o grão
quebrado, que tem metade da espessura do inteiro, cai abaixo do limiar e some —
justamente o grão que o trabalho precisa contar. A moeda é removida da máscara
antes do cálculo, e o limiar passou a ser calculado dentro de cada mancha
conectada, de modo que cada grão é medido na própria escala.

**Validação da moeda.** O círculo de maior raio encontrado só é aceito como moeda
se a área dele for pelo menos o dobro da área mediana dos grãos. Sem essa
checagem, uma foto sem moeda faria o maior grão virar a referência de escala e
todas as medidas em milímetros sairiam erradas sem nenhum aviso.

**Detecção automática do tipo de fundo.** O programa calcula o limiar de Otsu e
verifica de que lado dele cai a moldura da foto, que é quase sempre superfície e
não grão. Assim a mesma aplicação funciona com grãos claros sobre mesa escura e
com grãos escuros sobre papel branco, sem o usuário precisar informar nada.

**Premissa do critério clássico.** O baseline compara a área de cada grão com a
área mediana do lote, o que só faz sentido enquanto a maioria dos grãos estiver
inteira. Num lote dominado por pedaços, a mediana vira a do pedaço e quase nada é
apontado como defeituoso. É justamente aí que a CNN ganha: ela olha a forma de
cada grão isoladamente e não depende da composição do lote. Há um teste
documentando esse limite.

**Ordem dos canais na CNN.** O OpenCV trabalha em BGR e o utilitário do Keras que
carrega as imagens de treino entrega RGB. Passar BGR direto para a rede não dá
erro nenhum — ela simplesmente devolve a mesma resposta para todo grão. A
conversão é feita em `classifier.classify_cnn`.

## Testes

```bash
pytest -v
```

São 21 testes cobrindo contagem, descarte da moeda, conversão para milímetros,
recusa de escala quando não há moeda, os dois critérios de classificação, o
fallback sem modelo e as mensagens de erro. As fotos usadas nos testes são
geradas por `tests/synthetic.py` com quantidade de grãos conhecida, então o
resultado esperado é exato e não depende de nenhum arquivo externo.

## Vídeo demonstrativo

(adicionar aqui o link do vídeo antes da apresentação)
