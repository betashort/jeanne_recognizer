import numpy as np
#データの読み込みと前処理
from keras.utils import np_utils
from keras.datasets import mnist
#kerasでCNN構築
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
#時間計測
import time

np.random.seed(1)

#データの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#訓練データ
X_train = X_train.reshape(60000, 28, 28, 1)
X_train = X_train.astype('float32')#型を変更
X_train /= 255 #0から1.0の範囲に変換

#正解ラベル
correct = 10
y_train = np_utils.to_categorical(y_train, correct)

#テストデータ
X_test = X_test.reshape(10000, 28, 28, 1)
X_test = X_test.astype('float32')
X_test /= 255

y_test = np_utils.to_categorical(y_test, correct)



'''
CNNの構築
'''
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=(28, 28, 1), padding='same', activation = 'relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

#model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

'''
学習
'''
#計測開始
startTime = time.time()

history = model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('Test Loss:{0:.3f}'.format(score[0]))
print("time:{0:.3f}sec".format(time.time() - startTime))