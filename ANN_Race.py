import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/RA_AP_F.csv')
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values

dataset2 = pd.read_csv('D:/RA_AP_Test.csv')

X_test = dataset2.iloc[:, :-1].values
y_test = dataset2.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(42, activation = 'relu', input_dim = 20))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 42, activation = 'relu'))

# Adding the third hidden layer
ann.add(tf.keras.layers.Dense(units = 42, activation = 'relu'))

# Adding the output layer

ann.add(tf.keras.layers.Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
ann.fit(X_train, y_train, batch_size = 90, epochs = 2000)

y_pred = ann.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
predy = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

import math
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

rmse = math.sqrt(mse)
print('MSE is', mse)
print('RMSE is', rmse)

df = pd.DataFrame(predy).T
df.to_excel(excel_writer = "D:/RA_AP_Test.xlsx")
