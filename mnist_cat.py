# Load libraries
import sys

import tensorflow as tf

data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_images  = training_images / 255.0
val_images = val_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(20, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20, validation_data=(val_images, val_labels))

# get metrics for validation data, accuracy lower than test since model has not
# seen this data and may not be fully generalised, can also predict images and
# compare them to their actual label
model.evaluate(val_images, val_labels)

classifications = model.predict(val_images)
print(classifications[0])
print(val_labels[0])

# code is the same but layers are named prior to adding to the Sequential
# this means you can inspect the learned params later

data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_images  = training_images / 255.0
val_images = val_images / 255.0
layer_1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
layer_2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                                                        layer_1,
                                                                        layer_2])

model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20)

model.evaluate(val_images, val_labels)

classifications = model.predict(val_images)
print(classifications[0])
print(val_labels[0])

# inspect weights

print(layer_1.get_weights()[0].size)

# this statement above should print 15680 because there are 20 neurons in the first layer
# images are 28x28 = 784 / / 784 x 20 = 15680
# therefore instead of y=Mx+c, we have y=M1X1+M2X2+M3X3+....+M784X784+C in every neuron!
# Every pixel has a weight in every neuron. Those weights are multiplied by the pixel value, summed up, and given a bias.

print(layer_1.get_weights()[1].size)

print(layer_2.get_weights()[0].size)

print(layer_2.get_weights()[1].size)
