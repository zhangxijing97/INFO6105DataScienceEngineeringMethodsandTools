from keras.models import Sequential
import numpy as np
import keras
from keras.layers import *
import matplotlib.pyplot as plt

#parameters
num_classes = 10

#MNIST

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#visualize 25 training samples
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Create NN Model
nn = Sequential()
nn.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
nn.add(Dropout(0.2))
nn.add(Dense(num_classes, activation="softmax"))

# Set Optimizer, Loss Function, and metric
nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
nn.fit(x_train, y_train, batch_size=128, epochs=5 )
print(nn.summary())

# Accuracy
loss, accuracy= nn.evaluate(x_test, y_test )
print("Test accuracy:", accuracy)


#visualize 25 predictions on test data
y_test_predicted = nn.predict(x_test)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    plt.xlabel(np.argmax(y_test_predicted[i]))
plt.show()



#CIFAR 10

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) =  keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#visualize 25 training samples
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Create NN Model
nn = Sequential()
nn.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Dropout(0.2))

nn.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Flatten())
nn.add(Dense(num_classes, activation="softmax"))

# Set Optimizer, Loss Function, and metric
nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
nn.fit(x_train, y_train, batch_size=64, epochs=7 )

# Accuracy
loss, accuracy= nn.evaluate(x_test, y_test )
print("Test accuracy:", accuracy)


#visualize 25 predictions on test data
y_test_predicted = nn.predict(x_test)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    plt.xlabel(class_names[np.argmax(y_test_predicted[i])])
plt.show()