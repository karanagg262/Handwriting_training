import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activaion
from keras.optimizers import SGD
from keras.utils import np_utils

np.random seed(1007)

(xtrain, ytrain),(xtest,ytest) = mnist.load_data()

#each image is 28*28pixels = 784 i.e 784 neurons, one pixel for each
xtrain = xtrain.reshape(60000,784)
xtest = xtest.reshape(10000,784)
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')

#to change value of each pixel in the range[1-10]
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.shape[0],'train samples')
print(xtest.shape[0],test samples)

#since there are 10 numbers, 1 neuron each
ytrain = np_utils.to_categorical(y_train, 10)
ytest = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(10,input_shape=(784,)))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=SGD(),metrices=['accuracy'])
history = model.fit(xtrain,ytrain,batch_size=128,epochs=200,verbose=1,validation_split=0.2)

score=model.evaluate(xtest,ytest,verbose=1)
print("Test Score:",score[0])
print("Test accuracy:",score[1])
