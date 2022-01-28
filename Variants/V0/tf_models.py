import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

# Getting all of the data

data = pd.read_csv('excel_files/data_col_cleaned.csv')
titles = ['evaluations consumed', 'evaluations available', 'threshold-0']
for i in range(1,100):
    x = 'threshold-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = 'stdev-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = 'avfit-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '0%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '10%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '20%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '30%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '40%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '50%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '60%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '70%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '80%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '90%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '100%-' + str(i)
    titles.append(x)
titles.append('best_gamma')
results = {}
data = data.sample(frac=1).reset_index(drop=True)
X_all = data[titles[:-1]]
Y_all = data['best_gamma']
import pickle

filename = 'models/x_train.p'
file = open(filename, 'rb')
x_train = pickle.load(file)
file.close()
filename = 'models/y_train.p'
file = open(filename, 'rb')
y_train = pickle.load(file)
file.close()
filename = 'models/x_test.p'
file = open(filename, 'rb')
x_test = pickle.load(file)
file.close()
filename = 'models/y_test.p'
file = open(filename, 'rb')
y_test = pickle.load(file)
file.close()

y_test = to_categorical(y_test, 9)
y_train = to_categorical(y_train, 9)

def create_model1():
    model1 = keras.Sequential()
    model1.add(layers.Dense(64, input_shape=(1402,)))
    model1.add(layers.Dense(32))
    model1.add(layers.Dense(9))
    optimizer = keras.optimizers.SGD(lr = 0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy']
    model1.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model1

model1 = create_model1()
model1.fit(x_train, y_train, epochs = 1000)

evaluation = model1.evaluate(x_test, y_test)
ev = pd.Series(evaluation, index = model1.metrics_names)
print(ev)

model2 = keras.Sequential()
model2.add(layers.Dense(64, input_shape=(1402,), activation='sigmoid'))
model2.add(layers.Dense(32))
model2.add(layers.Dense(9))
optimizer = keras.optimizers.Adam(lr = 0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']
model2.compile(loss=loss, metrics=metrics, optimizer=optimizer)

model2.fit(x_train, y_train, epochs = 1000)

evaluation = model2.evaluate(x_test, y_test)
ev = pd.Series(evaluation, index = model2.metrics_names)
print(ev)

model3 = keras.Sequential()
model3.add(layers.Dense(64, input_shape=(x_train.shape[1],), activation='relu'))
model3.add(layers.Dropout(0.1))
model3.add(layers.Dense(32))
model3.add(layers.Dropout(0.1))
model3.add(layers.Dense(9, activation='sigmoid'))
optimizer = keras.optimizers.Adam(lr = 0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']
model3.compile(loss=loss, metrics=metrics, optimizer=optimizer)

model3.fit(x_train, y_train, epochs = 1000)
evaluation = model3.evaluate(x_test, y_test)
ev = pd.Series(evaluation, index = model3.metrics_names)
print(ev)

model4 = keras.Sequential()
model4.add(layers.Dense(256, input_shape=(x_train.shape[1],), activation='relu'))
model4.add(layers.Dense(128))
model4.add(layers.Dense(9, activation='softmax'))
optimizer = keras.optimizers.Adam(lr = 0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
metrics = ['accuracy']
model4.compile(loss=loss, metrics=metrics, optimizer=optimizer)

model4.fit(x_train, y_train, epochs = 1000)
evaluation = model4.evaluate(x_test, y_test)
ev = pd.Series(evaluation, index = model4.metrics_names)
print(ev)

model1.save('models/model19')
model2.save('models/model20')
model3.save('models/model21')
model4.save('models/model22')