# For Numpy
import matplotlib.pyplot as plt
import numpy as np
import pprint
import re
import sys
# For TensorFlow Lite (also uses some of the above)
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import tensorflow as tf
from tensorflow import keras
import pathlib
import pprint
import re
import sys

weights = np.random.randn(256, 256)

def quantizeAndReconstruct(weights):
    """
    @param W: np.ndarray

    This function computes the scale value to map fp32 values to int8. The function returns a weight matrix in fp32, that is representable
    using 8-bits.
    """
    # Compute the range of the weight.
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    range = max_weight - min_weight

    max_int8 = 2**8
    
    # Compute the scale
    scale = range / max_int8

    # Compute the midpoint
    midpoint = np.mean([max_weight, min_weight])

    # Next, we need to map the real fp32 values to the integers. For this, we make use of the computed scale. By diving the weight 
    # matrix with the scale, the weight matrix has a range between (-128, 127). Now, we can simply round the full precision numbers
    # to the closest integers.
    centered_weights = weights - midpoint
    quantized_weights = np.rint(centered_weights / scale)

    # Now, we can reconstruct the values back to fp32.
    reconstructed_weights = scale * quantized_weights + midpoint
    return reconstructed_weights

reconstructed_weights = quantizeAndReconstruct(weights)
print("Original weight matrix\n", weights)
print("Weight Matrix after reconstruction\n", reconstructed_weights)
errors = reconstructed_weights-weights
max_error = np.max(errors)
print("Max Error  : ", max_error)
reconstructed_weights.shape


# We can use np.unique to check the number of unique floating point numbers in the weight matrix.
np.unique(quantizeAndReconstruct(weights)).shape

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_data=(test_images, test_labels)
)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("/content/mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)


# Convert the model using DEFAULT optimizations: https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/python/lite.py#L91-L130
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir / "mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)


%%bash

cd /content/
git clone https://github.com/google/flatbuffers
cd flatbuffers
git checkout 0dba63909fb2959994fec11c704c5d5ea45e8d83
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make
cp flatc /usr/local/bin/
cd /content/
git clone --depth 1 https://github.com/tensorflow/tensorflow
flatc --python --gen-object-api tensorflow/tensorflow/lite/schema/schema_v3.fbs
pip install flatbuffers

# To allow us to import the Python files we've just generated we need to update the path env variable
sys.path.append("/content/tflite/")
import Model

def CamelCaseToSnakeCase(camel_case_input):
  """Converts an identifier in CamelCase to snake_case."""
  s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def FlatbufferToDict(fb, attribute_name=None):
  """Converts a hierarchy of FB objects into a nested dict."""
  if hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        snake_name = CamelCaseToSnakeCase(attribute_name)
        result[snake_name] = FlatbufferToDict(attribute, snake_name)
    return result
  elif isinstance(fb, str):
    return fb
  elif attribute_name == "name" and fb is not None:
    result = ""
    for entry in fb:
      result += chr(FlatbufferToDict(entry))
    return result
  elif hasattr(fb, "__len__"):
    result = []
    for entry in fb:
      result.append(FlatbufferToDict(entry))
    return result
  else:
    return fb

def CreateDictFromFlatbuffer(buffer_data):
  model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
  model = Model.ModelT.InitFromObj(model_obj)
  return FlatbufferToDict(model)

MODEL_ARCHIVE_NAME = 'inception_v3_2015_2017_11_10.zip'
MODEL_ARCHIVE_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/' + MODEL_ARCHIVE_NAME
MODEL_FILE_NAME = 'inceptionv3_non_slim_2015.tflite'
!curl -o {MODEL_ARCHIVE_NAME} {MODEL_ARCHIVE_URL}
!unzip {MODEL_ARCHIVE_NAME}
with open(MODEL_FILE_NAME, 'rb') as file:
 model_data = file.read()

model_dict = CreateDictFromFlatbuffer(model_data)


pprint.pprint(model_dict['subgraphs'][0]['tensors'])

param_bytes = bytearray(model_dict['buffers'][212]['data'])
params = np.frombuffer(param_bytes, dtype=np.float32)

params.min()
params.max()

plt.figure(figsize=(8,8))
plt.hist(params, 100)

 Text Classification
!wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite

# Post Estimation
!wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite


TEXT_CLASSIFICATION_MODEL_FILE_NAME = "text_classification_v2.tflite"
POSE_ESTIMATION_MODEL_FILE_NAME = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

with open(TEXT_CLASSIFICATION_MODEL_FILE_NAME, 'rb') as file:
  text_model_data = file.read()

with open(POSE_ESTIMATION_MODEL_FILE_NAME, 'rb') as file:
  pose_model_data = file.read()


def aggregate_all_weights(buffers):
    weights = []
    for i in range(len(buffers)):
        raw_data = buffers[i]['data']
        if raw_data is not None:
            param_bytes = bytearray(raw_data)
            params = np.frombuffer(param_bytes, dtype=np.float32)
            weights.extend(params.flatten().tolist())

    weights = np.asarray(weights)
    weights = weights[weights<50]
    weights = weights[weights>-50]

    return weights

model_dict_temp = CreateDictFromFlatbuffer(text_model_data)
weights = aggregate_all_weights(model_dict_temp['buffers'])

plt.figure(figsize=(8,8))
plt.hist(weights, 256, log=True)


model_dict_temp = CreateDictFromFlatbuffer(pose_model_data)
weights = aggregate_all_weights(model_dict_temp['buffers'][:-1])

plt.figure(figsize=(8,8))
plt.hist(weights, 256, log=True)
