import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns

from tensorflow.keras.models import (
    Sequential,
    load_model)

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.optimizers import Adam

# =========================================
# CAMINHO DO DATASET
# =========================================

base_path = r"C:\Users\solan\Downloads\EcoSepMobileNetV2\img"
# =========================================
# CONFIGURAÇÕES
# =========================================

IMG_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 60

LEARNING_RATE = 0.0001

RANDOM_STATE = 42

MODEL_PATH = "mobilenet_ecosep.keras"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================================
# CLASSES
# =========================================

classes = [ "PET", "PEAD"]

# =========================================
# LISTAS
# =========================================

imagens = []
labels = []

# =========================================
# CARREGAMENTO DAS IMAGENS
# =========================================

for classe in classes:

    pasta_classe = os.path.join(base_path, classe)

    for arquivo in os.listdir(pasta_classe):

        caminho_imagem = os.path.join(pasta_classe, arquivo)

        try:

            # Abre imagem
            img = Image.open(caminho_imagem).convert("RGB")

            # Redimensiona
            img = img.resize((IMG_SIZE, IMG_SIZE))

            # Converte para numpy
            img_array = np.array(img)

            # Salva
            imagens.append(img_array)
            labels.append(classe)

        except Exception as e:
            print(f"Erro ao carregar {arquivo}: {e}")

# =========================================
# CONVERSÃO PARA NUMPY
# =========================================

imagens = np.array(imagens)
labels = np.array(labels)


# =========================================
# BALANCEAMENTO AUTOMÁTICO
# =========================================

print("\n========== BALANCEAMENTO ==========")

# Descobre quantidade mínima
menor_quantidade = min([
    np.sum(labels == classe)
    for classe in classes
])

print(f"\nMenor classe possui {menor_quantidade} imagens")

imagens_balanceadas = []
labels_balanceados = []

for classe in classes:

    # Pega índices da classe
    indices_classe = np.where(labels == classe)[0]

    # Embaralha
    np.random.shuffle(indices_classe)

    # Limita pela menor quantidade
    indices_classe = indices_classe[:menor_quantidade]

    # Salva imagens balanceadas
    imagens_balanceadas.extend(imagens[indices_classe])

    # Salva labels balanceados
    labels_balanceados.extend(labels[indices_classe])

# Converte novamente
imagens = np.array(imagens_balanceadas)
labels = np.array(labels_balanceados)

# =========================================
# EMBARALHAMENTO FINAL
# =========================================

indices = np.arange(len(imagens))

np.random.shuffle(indices)

imagens = imagens[indices]
labels = labels[indices]

# =========================================
# PRINT DAS CLASSES
# =========================================

print("\nQuantidade após balanceamento:\n")

for classe in classes:

    quantidade = np.sum(labels == classe)

    print(f"{classe}: {quantidade}")

# =========================================
# NORMALIZAÇÃO
# =========================================

imagens = preprocess_input(imagens.astype(np.float32))

# =========================================
# CONVERSÃO DOS LABELS
# =========================================

mapa_labels = {
    "PET": 0,
    "PEAD": 1
}

labels = np.array([
    mapa_labels[label]
    for label in labels
])

# =========================================
# BALANCEAMENTO DO DATASET
# =========================================

contagem_classes = []

for i, classe in enumerate(classes):

    quantidade = np.sum(labels == i)

    contagem_classes.append(quantidade)

    print(f"{classe}: {quantidade} imagens")

# Gráfico de barras
plt.figure(figsize=(8,5))

plt.bar(
    classes,
    contagem_classes
)

plt.title("Balanceamento do Dataset")

plt.xlabel("Classes")
plt.ylabel("Quantidade de Imagens")

plt.show()

# =========================================
# TRAIN / VALIDATION / TEST
# 70 / 15 / 15
# =========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    imagens,
    labels,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

# =========================================
# PRINTS
# =========================================

print("\n========== DATASET ==========")

print("Treino:", X_train.shape)
print("Validação:", X_val.shape)
print("Teste:", X_test.shape)

# =========================================
# CRIAÇÃO DA MOBILENETV2
# =========================================

print("\nCriando MobileNetV2...")

base_model = MobileNetV2(

    weights="imagenet",
    include_top=False,
    pooling=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)

)

# Congela todas as camadas da MobileNet
base_model.trainable = False

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.4),

    Dense(
        2,
        activation="softmax"
    )

])

   
# =========================================
# COMPILAÇÃO
# =========================================

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# RESUMO
# =========================================

print("\n========== MODELO ==========")

model.summary()


# =========================================
# CALLBACKS
# =========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
# =========================================
# TREINAMENTO
# =========================================
inicio = time.time()

print("\nTreinando a cabeça da rede...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=10,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]

)
# =========================================
# FINE TUNING
# =========================================

print("\nIniciando Fine Tuning...")

# Descongela apenas as últimas camadas
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(

    optimizer=Adam(

        learning_rate=1e-5

    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

history_fine = model.fit(

    X_train,

    y_train,

    validation_data=(X_val, y_val),

    epochs=20,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]

)

fim = time.time()

print(f"\nTempo de treinamento: {(fim-inicio):.2f} segundos")

# =========================================
# AVALIAÇÃO
# =========================================
model = load_model(MODEL_PATH)

loss, acc = model.evaluate(X_test, y_test)

print(f"\nAcurácia no teste: {acc:.4f}")

# =========================================
# MATRIZ DE CONFUSÃO
# =========================================

# Faz previsões
y_pred = model.predict(X_test)

# Pega a classe com maior probabilidade
y_pred_classes = np.argmax(y_pred, axis=1)

# Cria matriz de confusão
matriz = confusion_matrix(y_test, y_pred_classes)

# Plot
plt.figure(figsize=(8,6))

sns.heatmap(
    matriz,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")

plt.title("Matriz de Confusão")

plt.show()

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        y_pred_classes,

        target_names=classes

    )

)

print(f"Precision: {precision_score(y_test, y_pred_classes):.4f}")

print(f"Recall: {recall_score(y_test, y_pred_classes):.4f}")

print(f"F1-Score: {f1_score(y_test, y_pred_classes):.4f}")

print("\nModelo salvo com sucesso!")

# =========================================
# GRÁFICO DE ACURÁCIA
# =========================================



plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.plot(history_fine.history["accuracy"])
plt.plot(history_fine.history["val_accuracy"])

plt.title("Acurácia do Modelo")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend([

    "Treino (Head)",

    "Validação (Head)",

    "Treino (Fine)",

    "Validação (Fine)"

])

plt.show()

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.plot(history_fine.history["loss"])
plt.plot(history_fine.history["val_loss"])

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend([

    "Treino (Head)",

    "Validação (Head)",

    "Treino (Fine)",

    "Validação (Fine)"

])

plt.show()