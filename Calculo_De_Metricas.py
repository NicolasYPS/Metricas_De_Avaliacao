import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# Matriz de confusão
confusion_matrix_normalized = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.98, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.99, 0.02, 0.00, 0.00, 0.00, 0.01],
    [0.00, 0.00, 0.00, 0.02, 0.00, 0.98, 0.01, 0.00, 0.00, 0.00],
    [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.99, 0.01, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.99, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.98]
])


# Assumindo 100 amostras por classe e 10 classes
samples_per_class = 100
total_samples = samples_per_class * 10

# Convertendo para matriz de contagens absolutas
confusion_matrix = np.round(confusion_matrix_normalized * samples_per_class).astype(int)

# Para cálculos de métricas, precisamos de vetores y_true e y_pred
y_true = []
y_pred = []

for i in range(10):  # Para cada classe real
    for j in range(10):  # Para cada classe predita
        count = confusion_matrix[i, j]
        y_true.extend([i] * count)
        y_pred.extend([j] * count)

# Calculando métricas globais
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Calculando métricas por classe
class_metrics = []
for i in range(10):
    # Para cada classe i, consideramos como "positivo"
    # e todas as outras como "negativo"
    tp = confusion_matrix[i, i]
    fn = np.sum(confusion_matrix[i, :]) - tp
    fp = np.sum(confusion_matrix[:, i]) - tp
    tn = total_samples - tp - fn - fp
    
    class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
    
    class_metrics.append({
        'class': i,
        'precision': class_precision,
        'recall': class_recall,
        'f1': class_f1
    })

# Exibindo os resultados
print("===== Matriz de Confusão (Contagens Absolutas) =====")
print(confusion_matrix)
print("\n===== Valores Normalizados (da Imagem) =====")
print(confusion_matrix_normalized)
print("\n===== Métricas Globais =====")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão (weighted): {precision:.4f}")
print(f"Sensibilidade/Recall (weighted): {recall:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")

print("\n===== Métricas por Classe =====")
for metric in class_metrics:
    print(f"Classe {metric['class']}: Precisão = {metric['precision']:.4f}, Recall = {metric['recall']:.4f}, F1 = {metric['f1']:.4f}")

# Plotando a matriz de confusão 
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix_normalized, annot=True, cmap='Blues', fmt='.2f', 
            cbar_kws={'label': 'Proporção'})
plt.title('Matriz de Confusão Normalizada', fontsize=16)
plt.xlabel('Classe Predita', fontsize=14)
plt.ylabel('Classe Real', fontsize=14)
plt.xticks(range(10), [str(i) for i in range(10)], fontsize=12)
plt.yticks(range(10), [str(i) for i in range(10)], fontsize=12, rotation=0)

# Ajusta o layout para não cortar os rótulos
plt.tight_layout()
plt.show()

# Calculando a curva ROC para cada classe (One-vs-Rest)
plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

for i in range(10):
    # Criar vetores binários para a classe i
    y_true_binary = [1 if label == i else 0 for label in y_true]
    y_score = []
    
    for j in range(len(y_true)):
        y_score.append(confusion_matrix_normalized[y_true[j], y_pred[j]])
    
    # Calculando FPR, TPR e thresholds
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plotando a curva ROC para esta classe
    plt.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'Classe {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)', fontsize=12)
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)', fontsize=12)
plt.title('Curvas ROC para cada classe (One-vs-Rest)', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()

# Exibindo informações adicionais sobre a curva ROC
print("\n===== Curvas ROC por Classe =====")
for i in range(10):
    y_true_binary = [1 if label == i else 0 for label in y_true]
    y_score = []
    
    for j in range(len(y_true)):
        y_score.append(confusion_matrix_normalized[y_true[j], y_pred[j]])
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    print(f"Classe {i}: AUC = {roc_auc:.4f}")