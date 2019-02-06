from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils

def napravi_mrezu():
    mreza = Sequential()
    mreza.add(Dense(512, input_shape=(784,), activation='tanh'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(512, activation='tanh'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(512, activation='tanh'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(512, activation='tanh'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(512, activation='tanh'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(10, activation='softmax'))

    return mreza


def treniraj_mrezu(mreza, X_train, y_train, X_test, y_test):
    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    mreza.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    mreza.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1, validation_data=(X_test, y_test))

    return mreza