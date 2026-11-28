"""Configurações centrais do projeto.

Todos os parâmetros ajustáveis ficam aqui para que segmentação, escala,
classificação e interface usem os mesmos valores.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "grain_cnn.keras"

# --------------------------------------------------------------------------
# Moeda de referência para converter pixels em milímetros
# Diâmetros oficiais da segunda família do Real (Banco Central do Brasil):
# https://www.bcb.gov.br/dinheirobrasileiro/segunda-familia-moedas.html
# --------------------------------------------------------------------------
COINS_MM = {
    "R$ 1,00": 27.0,
    "R$ 0,50": 23.0,
    "R$ 0,25": 25.0,
    "R$ 0,10": 20.0,
    "R$ 0,05": 22.0,
}
DEFAULT_COIN = "R$ 1,00"
COIN_DIAMETER_MM = COINS_MM[DEFAULT_COIN]

# --------------------------------------------------------------------------
# Segmentação (Watershed)
# --------------------------------------------------------------------------
MEAN_SHIFT_SP = 5          # raio espacial do pyrMeanShiftFiltering
MEAN_SHIFT_SR = 50         # raio de cor do pyrMeanShiftFiltering
DIST_FACTOR = 0.52         # fração do máximo da transformada de distância
MIN_AREA_PX = 50           # área mínima, em pixels, para o contorno valer como grão
MORPH_KERNEL = 3           # lado do kernel das operações morfológicas
MORPH_ITERATIONS = 3

# Redimensionamento: fotos de celular são grandes demais para o Watershed
# rodar rápido. A imagem é reduzida para esta largura antes do processamento.
MAX_WIDTH_PX = 1280
MIN_WIDTH_PX = 800

# --------------------------------------------------------------------------
# Detecção da moeda (HoughCircles)
# --------------------------------------------------------------------------
COIN_MIN_RADIUS = 20
COIN_MAX_RADIUS = 300
COIN_PARAM1 = 100
COIN_PARAM2 = 30

# --------------------------------------------------------------------------
# Classificação
# --------------------------------------------------------------------------
LABEL_INTEIRO = 0
LABEL_DEFEITUOSO = 1
LABEL_NAMES = {LABEL_INTEIRO: "inteiro", LABEL_DEFEITUOSO: "defeituoso"}

THRESH_AREA_REL = 0.60     # abaixo desta fração da área mediana, o grão é quebrado
THRESH_SOLIDEZ = 0.90      # abaixo desta solidez, o contorno é irregular

# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------
CROP_SIZE = 64             # lado, em pixels, do recorte quadrado de cada grão
CROP_MARGIN = 4            # folga em pixels ao redor do grão no recorte
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 60            # a rede é pequena: cada época leva poucos segundos
CNN_PATIENCE = 12          # paciência do EarlyStopping, em épocas
CNN_SEED = 42

# --------------------------------------------------------------------------
# Cores usadas no desenho dos resultados (BGR, padrão do OpenCV)
# --------------------------------------------------------------------------
COLOR_INTEIRO = (0, 200, 0)        # verde
COLOR_DEFEITUOSO = (0, 0, 255)     # vermelho
COLOR_COIN = (255, 170, 0)         # azul claro
