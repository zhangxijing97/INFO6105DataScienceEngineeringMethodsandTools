import numpy as np
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.xception import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

num_classes = 10

#CIFAR 10

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) =  keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',  'dog', 'frog', 'horse', 'ship', 'truck']

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

# Xception as feature extractor
feature_extractor = Xception( weights='imagenet', # Load weights pre-trained on ImageNet
                              include_top=False,  # Do not include the ImageNet classifier at the top
                              pooling='max')
feature_extractor.trainable = False

# Create model
my_new_model = Sequential()
my_new_model.add(feature_extractor)
my_new_model.add(Dense(800, activation='relu'))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.summary()

# Set Optimizer, Loss Function, and metric
my_new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
my_new_model.fit(x_train, y_train, batch_size=64, epochs=2 )

# Accuracy
loss, accuracy= my_new_model.evaluate(x_test, y_test )
print("Test accuracy:", accuracy)

#visualize 25 predictions on test data
y_test_predicted = my_new_model.predict(x_test)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    plt.xlabel(class_names[np.argmax(y_test_predicted[i])])
plt.show()
