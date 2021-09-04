#For CKPlus
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred.shape)
    conf_Mat=np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat

#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
X=np.genfromtxt('/kaggle/input/ckplusfinal/CKPlusFinalN.csv', delimiter=' ', dtype='float32')
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
#X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
#y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
#X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
def genModel():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7 , activation='softmax'))
    #print(model.summary())
    return model
history=[]
es=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=40,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
for k in range(5):
    dbyf=X.shape[0]/5.0
    X_test=X[int(k*dbyf):int((k+1)*dbyf), :, :, :]
    Y_test=Y[int(k*dbyf):int((k+1)*dbyf), :]
    X_train=np.concatenate((X[0:int(k*dbyf), :, :, :], X[int((k+1)*dbyf):, :, :, :]), axis=0)
    Y_train=np.concatenate((Y[0:int(k*dbyf), :], Y[int((k+1)*dbyf):, :]), axis=0)
    model=genModel()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    history.append(model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=220, batch_size=500, shuffle=True, callbacks=[es]))
    if k==0:
        conf=ConfusionMatrix(X_test, Y_test, model)
    else:
        conf=conf+ConfusionMatrix(X_test, Y_test, model)
    model.save('/kaggle/working/model1.h5')
print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 7, axis=1))
#print(history.history.keys())
for  i in range(5):
    print(i)
    plt.plot(history[i].history['val_accuracy'])
    plt.plot(history[i].history['accuracy'])
    plt.show()
    plt.plot(history[i].history['val_loss'])
    plt.plot(history[i].history['loss'])
    plt.show()
print(max(history[0].history['val_accuracy']))
print(max(history[1].history['val_accuracy']))
print(max(history[2].history['val_accuracy']))
print(max(history[3].history['val_accuracy']))
print(max(history[4].history['val_accuracy']))
#print(max(history[5].history['val_accuracy']))
#adam=tf.keras.optimizers.Adam(learning_rate=0.1)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=15,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=1,
    min_lr=0.00001
)
opt = SGD(lr=0.001, momentum=0.9, decay=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#ct_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])