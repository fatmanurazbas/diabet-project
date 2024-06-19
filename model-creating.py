import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle

# Veriyi yükleme ve dataframe'e dönüştürme
dataset = pd.read_csv('diabetes-data.csv')

# Veriyi ayırma
X = dataset.drop(columns='Outcome', axis=1)
y = dataset['Outcome']

# Veriyi standartlaştırma
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)

X = scaled_data
y = dataset['Outcome']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Modeli eğitme
model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Modeli kaydetme
with open('svm_model.pkl', 'wb') as file:
    pickle.dump((scaler, model), file)

# Eğitim verisi ile tahmin
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test verisi ile tahmin
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Değerlendirme metrikleri
f1 = f1_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# ROC eğrisi için veri
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

# ROC eğrisini çizme
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Sonuçları yazdırma
print('Train Accuracy: ', train_accuracy)
print('Test Accuracy: ', test_accuracy)
print('F1 Score: ', f1)
print('Recall: ', recall)
print('ROC AUC: ', roc_auc)
