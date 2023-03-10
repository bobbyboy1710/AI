import numpy as np
from sklearn import preprocessing
input_data = np.array([[4.1, -5.9, -3.5], [-1.9, 4.6, 3.9], [-4.2, 6.8, 6.3], [3.9, 3.4, 1.2]])
data_binarized = preprocessing.Binarizer(threshold=3.2).transform(input_data)
data_scaled = preprocessing.scale(input_data)
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')



print('\n Бінаризація: \n', data_binarized)
print('\n ДО: ')
print('Mean =', input_data.mean(axis=0))
print("Виключення середнього =", input_data.std(axis=0))
print("\nПІСЛЯ: ")
print("Mean =", data_scaled.mean(axis=0))
print("Виключення середнього: ", data_scaled.std(axis=0))
print("\nМасштабування:\n", data_scaled_minmax)
print("\nL1 нормалізація:\n", data_normalized_l1)
print("\nL2 нормалізація:\n", data_normalized_l2)
