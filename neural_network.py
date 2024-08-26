# Load in TensorFlow library
import tensorflow as tf
import numpy as np

# Define my custom Neural Network model

class MyModel(tf.keras.Model):
    """Model Class, instance of which gives a model, this model happens to be a neural network.

    :param tf: tensorflow python module.
    :type tf: Module
    """

    def __init__(self):

        super(MyModel, self).__init__()

        # define Neural Network layer types

        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu')

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')

        self.dense2 = tf.keras.layers.Dense(10)

    #run my Neural Network model by evaluating each layer on my input data

    def call(self, x):
        """Call the model by evaluating each layer on input data.

        :param x: input data
        :type x: could be many things
        :return: instance of model
        :rtype: model
        """

        x = self.conv(x)

        x = self.flatten(x)

        x = self.dense1(x)

        x = self.dense2(x)

        return x

# Create an instance of the model
# model = MyModel()

# define a neural network with one neuron
# for more information on TF functions see: https://www.tensorflow.org/api_docs
my_layer_1 = tf.keras.layers.Dense(units=2, input_shape=[1])
my_layer_2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([my_layer_1, my_layer_2])

# use stochastic gradient descent for optimization and
# the mean squared error loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# define some training data (xs as inputs and ys as outputs)
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# fit the model to the data (aka train the model)
model.fit(xs, ys, epochs=500)
input_data = np.array([[10.0]])
print(model.predict(input_data))

# print weights
print(my_layer_1.get_weights())
print(my_layer_2.get_weights())


# Use 2 weight values and bias then apply
value_to_predict = 10.0

layer1_w1 = (my_layer_1.get_weights()[0][0][0])

layer1_w2 = (my_layer_1.get_weights()[0][0][1])

layer1_b1 = (my_layer_1.get_weights()[1][0])

layer1_b2 = (my_layer_1.get_weights()[1][1])




layer2_w1 = (my_layer_2.get_weights()[0][0])

layer2_w2 = (my_layer_2.get_weights()[0][1])

layer2_b = (my_layer_2.get_weights()[1][0])

 

neuron1_output = (layer1_w1 * value_to_predict) + layer1_b1

neuron2_output = (layer1_w2 * value_to_predict) + layer1_b2

 

neuron3_output = (layer2_w1 * neuron1_output) + (layer2_w2 * neuron2_output) + layer2_b

 

print(neuron3_output)
