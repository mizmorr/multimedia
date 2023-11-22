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
# print(tf.config.list_physical_devices('GPU'))
def get_num(a):
    for i in range(0,len(a)):
        if a[i]>0.5: return i


# Импортируем набор данных MNIST

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/X_train.max()
X_test = X_test/X_test.max()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# model = Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#         layers.MaxPooling2D((2, 2)),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10, activation='softmax')

# ])

model = Sequential([
    layers.Conv2D(filters=64,input_shape=(28,28,1),kernel_size = (3,3),activation = 'relu'),
    layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(512,activation='relu'),
    layers.Dense(10,activation='softmax'),

])

# model.compile(loss='binary_crossentropy',
#             optimizer = Adam(learning_rate=0.00024),
#              metrics = ['binary_accuracy'])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)

es = keras.callbacks.EarlyStopping(
        monitor="val_acc", # metrics to monitor
        patience=10, # how many epochs before stop
        verbose=1,
        mode="max", # we need the maximum accuracy.
        restore_best_weights=True, #
     )

rp = keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc",
        factor=0.2,
        patience=3,
        verbose=1,
        mode="max",
        min_lr=0.00001,
     )
history = model.fit(X_train, y_train, batch_size=128, verbose=1,
                    epochs= 100, validation_split = 0.2, callbacks=[stop])
pred = model.predict(X_test)
print(pred[1])
print(get_num[pred[1]])
cv.imshow("pred",X_test[1])
cv.waitKey(0)
cv.destroyAllWindows()
