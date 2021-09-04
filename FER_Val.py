from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred.shape)
    conf_Mat=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat
X=np.genfromtxt('/kaggle/input/datanfer/CleanedFERNFinal/FERFinalN.csv', delimiter=' ', dtype='float32')
X_test=X[X.shape[0]-2000:, :]
X=X[:X.shape[0]-2000, :]
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=5)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=5)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model=keras.models.load_model('../input/models/CleanedFER.h5')
model.evaluate(X_test, y_test)
conf=ConfusionMatrix(X_test, y_test, model)
print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 5, axis=1))
