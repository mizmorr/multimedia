# библиотека для вывода изображений
import matplotlib.pyplot as plt
import cv2 as cv
# -- Импорт для построения модели: --
# импорт слоев
import keras
from keras import layers,utils
from keras.layers import Dense, Flatten
# импорт модели
from keras.models import Sequential
# импорт оптимайзера
from keras.optimizers import Adam
# print(tf.config.list_physical_devices('GPU'))
# Импортируем набор данных MNIST
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/X_train.max()
X_test = X_test/X_test.max()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train[0].shape)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
            optimizer = Adam(learning_rate=0.00024),
             metrics = ['binary_accuracy'])


stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)
history = model.fit(X_train, y_train, batch_size=500, verbose=1,
                    epochs= 50, validation_split = 0.2, callbacks=[stop])
pred = model.predict(X_test)
print(pred[0])
cv.imshow("pred",X_test[0])
cv.waitKey(0)
cv.destroyAllWindows()
