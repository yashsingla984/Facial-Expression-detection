#Cleaned FER
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred)
    conf_Mat=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat

#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
X=np.genfromtxt('/kaggle/input/datanfer/CleanedFERNFinal/FERFinalN.csv', delimiter=' ', dtype='float32')
#np.random.shuffle(X)
X_test=X[X.shape[0]-2000:, :]
X=X[:X.shape[0]-2000, :]
#X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=5)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=5)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model = Sequential()
p=0.2
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(p))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(p))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(p))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(p))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(p))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p))
model.add(Dense(64, activation='relu'))
model.add(Dropout(p))
model.add(Dense(5 , activation='softmax'))
print(model.summary())
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
try:
    model.fit(X, Y, validation_data=(X_test, y_test), epochs=1000, batch_size=500, shuffle=True)
    conf=ConfusionMatrix(X_test, y_test, model)
    print(conf)
    print(np.sum(conf))
    print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 5, axis=1)) 
    model.save('/kaggle/working/model1.h5')
except:
    conf=ConfusionMatrix(X_test, y_test, model)
    print(conf)
    print(np.sum(conf))
    print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 5, axis=1)) 
    model.save('/kaggle/working/model1.h5')