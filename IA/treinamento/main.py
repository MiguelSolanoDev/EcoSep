import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import seaborn as sns

from tensorflow.keras.models import (
    Sequential,
    load_model
)

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import EarlyStopping

# =========================================
# CAMINHO DO DATASET
# =========================================

base_path = r"C:\Users\0081824\Downloads\EcoSepCNN_0527(1)\EcoSepCNN_0527\EcoSepCNN_2705\EcoSepCNN\img"

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
            img = img.resize((224, 224))

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

imagens = imagens / 255.0

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
    random_state=42,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
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

if os.path.exists("modelo_ecosep.keras"):

    print("\nCarregando modelo existente...")

    model = load_model("modelo_ecosep.keras")

else:

    print("\nCriando novo modelo...")

    model = Sequential([

        # Primeira convolução
        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(224,224,3)
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
        Flatten(),

        # Dense
        Dense(256, activation='relu'),

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
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# RESUMO
# =========================================

print("\n========== MODELO ==========")

model.summary()

# =========================================
# EARLY STOPPING
# =========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# =========================================
# TREINAMENTO
# =========================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# =========================================
# AVALIAÇÃO
# =========================================

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

# =========================================
# SALVAR MODELO
# =========================================

model.save("modelo_ecosep.keras")

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

plt.show()