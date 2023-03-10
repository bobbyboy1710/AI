import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Зчитуємо дані з файлу та створюємо DataFrame
data = pd.read_csv('data_metrics.csv')

# Визначаємо змінні X та y для моделі
# Вибираємо усі змінні, окрім цільової змінної
X = data.drop('actual_label', axis=1)
y = data['actual_label']

# Розділяємо дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Створюємо модель машини опорних векторів
svm_model = SVC(kernel='linear')

# Тренуємо модель на тренувальному наборі даних
svm_model.fit(X_train, y_train)

# Діагностичні метрики класифікації
y_pred = svm_model.predict(X_test)

# Рахуємо accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
