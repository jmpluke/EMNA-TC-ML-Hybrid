'''
    Training, scoring, and saving Keras models
'''
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import pickle

filename = 'v1_models/x_train_v1.p'
file = open(filename, 'rb')
x_train = pickle.load(file)
file.close()
filename = 'v1_models/y_train_v1.p'
file = open(filename, 'rb')
y_train = pickle.load(file)
file.close()
filename = 'v1_models/x_test_v1.p'
file = open(filename, 'rb')
x_test = pickle.load(file)
file.close()
filename = 'v1_models/y_test_v1.p'
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

model1.save('v1_models/model19')
model2.save('v1_models/model20')
model3.save('v1_models/model21')
model4.save('v1_models/model22')