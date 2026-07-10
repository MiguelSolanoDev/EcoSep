import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split

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
    load_model
)

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
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

base_path = r"C:\Users\solan\OneDrive\Área de Trabalho\EcoSep\IA\DataSet"

# =========================================
# CONFIGURAÇÕES
# =========================================

IMG_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 60

LEARNING_RATE = 0.0001

RANDOM_STATE = 42

MODEL_PATH = "modelo_ecosep.keras"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================================
# CLASSES
# =========================================

classes = [ "PETPEAD", "Outros"]

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

# =========================================
# NORMALIZAÇÃO
# =========================================

imagens = imagens / 255.0

# =========================================
# CONVERSÃO DOS LABELS
# =========================================

mapa_labels = {
    "PETPEAD": 0,
    "Outros": 1
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
# CARREGAR OU CRIAR MODELO
# =========================================

if os.path.exists(MODEL_PATH):

    print("\nCarregando modelo existente...")

    model = load_model(MODEL_PATH)

else:

    print("\nCriando novo modelo...")

    model = Sequential([

        # Primeira convolução
        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),

        BatchNormalization(),

        MaxPooling2D((2,2)),

        # Segunda convolução
        Conv2D(
            64,
            (3,3),
            activation='relu'
        ),

        BatchNormalization(),

        MaxPooling2D((2,2)),

        # Terceira convolução
        Conv2D(
            128,
            (3,3),
            activation='relu'
        ),

        BatchNormalization(),

        MaxPooling2D((2,2)),

        # Flatten
        GlobalAveragePooling2D(),

        # Dense
        Dense(128, activation='relu'),

        BatchNormalization(),

        # Evita overfitting
        Dropout(0.5),

        # Saída
        Dense(2, activation='softmax')
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
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
   callbacks=[
    early_stop,
    reduce_lr,
    checkpoint
    ]
)
print(f"\nEpochs treinadas: {len(history.history['loss'])}")
fim = time.time()

print(f"\nTempo de treinamento: {(fim-inicio):.2f} segundos")

# =========================================
# AVALIAÇÃO
# =========================================

loss, acc = model.evaluate(X_test, y_test)

print(f"\nAcurácia no teste: {acc:.4f}")

print(f"Loss final: {loss:.4f}")

inicio = time.time()

y_pred = model.predict(X_test)

fim = time.time()

tempo_total = fim - inicio
tempo_imagem = tempo_total / len(X_test)

print(f"\nTempo total de inferência: {tempo_total:.4f} s")
print(f"Tempo médio por imagem: {tempo_imagem*1000:.2f} ms")
# =========================================
# MATRIZ DE CONFUSÃO
# =========================================

# Pega a classe com maior probabilidade
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred_classes,
    target_names=classes
))

# Matriz
matriz = confusion_matrix(
    y_test,
    y_pred_classes
)

precision = precision_score(
    y_test,
    y_pred_classes,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred_classes,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred_classes,
    average='weighted'
)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

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

# =========================================
# SALVAR MODELO
# =========================================

model.save(MODEL_PATH)

print("\nModelo salvo com sucesso!")

# =========================================
# GRÁFICO DE ACURÁCIA
# =========================================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Acurácia do Modelo')

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend([
    'Treino',
    'Validação'
])

print("\n========== DATASET ==========")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Imagem: {IMG_SIZE}x{IMG_SIZE}")
print(f"Épocas máximas: {EPOCHS}")

plt.show()