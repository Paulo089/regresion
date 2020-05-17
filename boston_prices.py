import tensorflow as tf
import numpy as np
import keras
from keras.datasets import boston_housing
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from keras.applications import VGG16
import matplotlib.pylab as plt
batch_size = 10
epochs = 50

(xt, yt), (xtest, ytest) = boston_housing.load_data()

mean = xt.mean(axis=0)
xt -= mean
std = xt.std(axis=0)
xt /= std
print(xt.shape[1])
xtest -= mean
xtest /= std

entradas = Input(shape=(13,))

x=Dense(64,activation='relu')(entradas)
x=Dense(64,activation='relu')(x)
x=Dense(1,activation='linear')(x)

modelo = Model(inputs=entradas, outputs=x)
modelo.summary()
Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)  # SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

modelo.compile(loss=keras.losses.mse, optimizer=Adam,metrics=['mse'])

history = modelo.fit(xt,yt,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(xtest,ytest))
puntuacion = modelo.evaluate(xtest,ytest,verbose=1)
print(puntuacion)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdidas del Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()
