import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

## LOAD DATA
# mnist_reader default function
(X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()
# reshape to HxWxChannels, normalize pixel values to floats 0.0 to 1.0
X_train = X_train.reshape((-1, 28, 28, 1)) / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)) / 255.0
# one-hot labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

## CONSTRUCT MODEL
model = Sequential()
# 2D convolutional layer, 28 filters, 3x3 window size, ReLU activation
model.add(Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# 2x2 max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# 2D convolutional layer, 56 filters, 3x3 window size, ReLU activation
model.add(Conv2D(56, kernel_size=(3, 3), activation='relu'))
# Flattening for fully-connected layers ?
model.add(Flatten())
# Fully-connected layer, 56 nodes, ReLU activation
model.add(Dense(56, activation='relu'))
# Fully-connected OUTPUT layer, 56 nodes, softmax activation
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## SPLIT DATA FOR VALIDATION
X_train, X_val = X_train[:-12000], X_train[-12000:]
y_train, y_val = y_train[:-12000], y_train[-12000:]

## TRAIN
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

## EVALUATE
model.summary()

## PLOT
plt.ion()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fashion MNIST Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('accuracy_plot.png')

## EVALUATE
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

## SAVE
model.save('./saves/model.keras')
