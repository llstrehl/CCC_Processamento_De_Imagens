"""Treino da CNN que classifica cada grão em inteiro ou defeituoso.

Espera os recortes organizados em duas pastas, uma por classe:

    data/recortes/inteiro/*.png
    data/recortes/defeituoso/*.png

Use ``python -m src.dataset`` para gerar esses recortes a partir das fotos.

Uso:

    python -m src.train --dados data/recortes --saida models/grain_cnn.keras

O mesmo código roda no Colab pelo notebook ``notebooks/treino_cnn.ipynb``.
"""

import argparse
from pathlib import Path

from . import config


def build_model(input_shape=(config.CROP_SIZE, config.CROP_SIZE, 3)):
    """Monta a CNN.

    A rede é pequena de propósito: o dataset de um trabalho de disciplina tem
    algumas centenas de recortes, e uma rede grande decoraria as imagens em
    vez de aprender a diferença entre grão inteiro e grão quebrado.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    aumento = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
    ], name="aumento_de_dados")

    modelo = keras.Sequential([
        layers.Input(shape=input_shape),
        aumento,
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="grain_cnn")

    modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
    return modelo


def load_datasets(pasta_dados, batch_size=config.CNN_BATCH_SIZE,
                  validation_split=0.2, seed=config.CNN_SEED):
    """Carrega os recortes das pastas e devolve ``(treino, validação)``.

    ``class_names`` é fixado para que o rótulo 1 seja sempre "defeituoso",
    igual ao que ``config.LABEL_DEFEITUOSO`` define no resto do projeto.
    """
    from tensorflow import keras

    comum = dict(
        labels="inferred",
        label_mode="binary",
        class_names=[config.LABEL_NAMES[config.LABEL_INTEIRO],
                     config.LABEL_NAMES[config.LABEL_DEFEITUOSO]],
        image_size=(config.CROP_SIZE, config.CROP_SIZE),
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
    )

    treino = keras.utils.image_dataset_from_directory(pasta_dados, subset="training", **comum)
    validacao = keras.utils.image_dataset_from_directory(pasta_dados, subset="validation", **comum)
    return treino, validacao


def train(pasta_dados=None, saida=None, epochs=config.CNN_EPOCHS):
    """Treina a rede e salva o modelo. Devolve o histórico do treino."""
    import tensorflow as tf
    from tensorflow import keras

    pasta_dados = Path(pasta_dados or (config.DATA_DIR / "recortes"))
    saida = Path(saida or config.MODEL_PATH)
    saida.parent.mkdir(parents=True, exist_ok=True)

    if not pasta_dados.exists():
        raise SystemExit(
            f"A pasta {pasta_dados} não existe. Rode antes:\n"
            f"  python -m src.dataset data/fotos --saida {pasta_dados}"
        )

    treino, validacao = load_datasets(pasta_dados)
    treino = treino.cache().shuffle(256).prefetch(tf.data.AUTOTUNE)
    validacao = validacao.cache().prefetch(tf.data.AUTOTUNE)

    modelo = build_model()
    modelo.summary()

    paradas = [
        # A acurácia costuma ficar parada nas primeiras épocas, enquanto a rede
        # ainda prevê sempre a classe majoritária. Uma paciência curta
        # interromperia o treino justamente aí, antes de ela aprender a forma
        # do grão — por isso a espera é longa.
        keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                      patience=config.CNN_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]

    historico = modelo.fit(treino, validation_data=validacao,
                           epochs=epochs, callbacks=paradas)

    modelo.save(saida)
    perda, acuracia = modelo.evaluate(validacao, verbose=0)
    print(f"\nAcurácia na validação: {acuracia * 100:.1f}%  (perda {perda:.3f})")
    print(f"Modelo salvo em {saida}")
    return historico


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", default=None, help="pasta com inteiro/ e defeituoso/")
    parser.add_argument("--saida", default=None, help="caminho do modelo .keras")
    parser.add_argument("--epocas", type=int, default=config.CNN_EPOCHS)
    argumentos = parser.parse_args()
    train(argumentos.dados, argumentos.saida, argumentos.epocas)


if __name__ == "__main__":
    main()
