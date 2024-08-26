!wget https://github.com/tensorflow/tensorflow/archive/v2.4.1.zip
!unzip v2.4.1.zip &> 0
!mv tensorflow-2.4.1/ tensorflow/

#!pip uninstall tensorflow -y
#!pip install tensorflow-gpu==1.15

import tensorflow.compat.v1 as tf
#assert(tf.VERSION == "1.15.0") #Make sure you install Tensorflow 1.15.0 and restart the runtime or this will fail!
# Now that the runtime is set import things!
import sys
# We add this path so we can import the speech processing modules.
sys.path.append("/content/tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models
import numpy as np
import pickle


WANTED_WORDS = # YOUR CODE HERE #

TRAINING_STEPS =  # YOUR CODE HERE #

LEARNING_RATE = # YOUR CODE HERE #

MODEL_ARCHITECTURE = 'tiny_conv'

# Calculate the total number of steps, which is used to identify the checkpoint
# file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Print the configuration to confirm it
print("Training these words: %s" % WANTED_WORDS)
print("Training steps in each stage: %s" % TRAINING_STEPS)
print("Learning rate in each stage: %s" % LEARNING_RATE)
print("Total number of training steps: %s" % TOTAL_STEPS)


# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20

# Constants used during training only
VERBOSITY = 'DEBUG'
EVAL_STEP_INTERVAL = '1000'
SAVE_STEP_INTERVAL = '1000'

# Constants for training directories and filepaths
DATASET_DIR =  'dataset/'
LOGS_DIR = 'logs/'
TRAIN_DIR = 'train/' # for training checkpoints and other files.

# Constants for inference directories and filepaths
import os
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
  os.mkdir(MODELS_DIR)
MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

# Constants for Quantization
QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN

# Constants for audio process during Quantization and Evaluation
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# URL for the dataset and train/val/test split
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10


# Calculate the correct flattened input data shape for later use in model conversion
# since the model takes a flattened version of the spectrogram. The shape is number of 
# overlapping windows times the number of frequency bins. For the default settings we have
# 40 bins (as set above) times 49 windows (as calculated below) so the shape is (1,1960)
def window_counter(total_samples, window_size, stride):
  '''helper function to count the number of full-length overlapping windows'''
  window_count = 0
  sample_index = 0
  while True:
    window = range(sample_index,sample_index+stride)
    if window.stop < total_samples:
      window_count += 1
    else:
      break
    
    sample_index += stride
  return window_count

OVERLAPPING_WINDOWS = window_counter(CLIP_DURATION_MS, int(WINDOW_SIZE_MS), WINDOW_STRIDE)
FLATTENED_SPECTROGRAM_SHAPE = (1, OVERLAPPING_WINDOWS * FEATURE_BIN_COUNT)



%load_ext tensorboard
%tensorboard --logdir {LOGS_DIR}

!python tensorflow/tensorflow/examples/speech_commands/train.py \
--data_dir={DATASET_DIR} \
--wanted_words={WANTED_WORDS} \
--silence_percentage={SILENT_PERCENTAGE} \
--unknown_percentage={UNKNOWN_PERCENTAGE} \
--preprocess={PREPROCESS} \
--window_stride={WINDOW_STRIDE} \
--model_architecture={MODEL_ARCHITECTURE} \
--how_many_training_steps={TRAINING_STEPS} \
--learning_rate={LEARNING_RATE} \
--train_dir={TRAIN_DIR} \
--summaries_dir={LOGS_DIR} \
--verbosity={VERBOSITY} \
--eval_step_interval={EVAL_STEP_INTERVAL} \
--save_step_interval={SAVE_STEP_INTERVAL}

!rm -rf {SAVED_MODEL}
!python tensorflow/tensorflow/examples/speech_commands/freeze.py \
--wanted_words=$WANTED_WORDS \
--window_stride_ms=$WINDOW_STRIDE \
--preprocess=$PREPROCESS \
--model_architecture=$MODEL_ARCHITECTURE \
--start_checkpoint=$TRAIN_DIR$MODEL_ARCHITECTURE'.ckpt-'{TOTAL_STEPS} \
--save_format=saved_model \
--output_file={SAVED_MODEL}

model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
    SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
    WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, LOGS_DIR)

with tf.Session() as sess:
  # with tf.compat.v1.Session() as sess: #replaces the above line for use with TF2.x
  float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  float_tflite_model = float_converter.convert()
  float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
  print("Float model is %d bytes" % float_tflite_model_size)

  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_input_type = tf.lite.constants.INT8
  # converter.inference_input_type = tf.compat.v1.lite.constants.INT8 #replaces the above line for use with TF2.x   
  converter.inference_output_type = tf.lite.constants.INT8
  # converter.inference_output_type = tf.compat.v1.lite.constants.INT8 #replaces the above line for use with TF2.x
  def representative_dataset_gen():
    for i in range(100):
      data, _ = audio_processor.get_data(1, i*1, model_settings,
                                         BACKGROUND_FREQUENCY, 
                                         BACKGROUND_VOLUME_RANGE,
                                         TIME_SHIFT_MS,
                                         'testing',
                                         sess)
      flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(FLATTENED_SPECTROGRAM_SHAPE)
      yield [flattened_data]
  converter.representative_dataset = representative_dataset_gen
  tflite_model = converter.convert()
  tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
  print("Quantized model is %d bytes" % tflite_model_size)

  # Helper function to run inference
def run_tflite_inference_testSet(tflite_model_path, model_type="Float"):
  #
  # Load test data
  #
  np.random.seed(0) # set random seed for reproducible test results.
  with tf.Session() as sess:
  # with tf.compat.v1.Session() as sess: #replaces the above line for use with TF2.x
    test_data, test_labels = audio_processor.get_data(
        -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
        TIME_SHIFT_MS, 'testing', sess)
  test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

  #
  # Initialize the interpreter
  #
  interpreter = tf.lite.Interpreter(tflite_model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]
  
  #
  # For quantized models, manually quantize the input data from float to integer
  #
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  #
  # Evaluate the predictions
  #
  correct_predictions = 0
  for i in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()
    correct_predictions += (top_prediction == test_labels[i])

  print('%s model accuracy is %f%% (Number of test samples=%d)' % (
      model_type, (correct_predictions * 100) / len(test_data), len(test_data)))

# Compute float model accuracy
run_tflite_inference_testSet(FLOAT_MODEL_TFLITE)

# Compute quantized model accuracy
run_tflite_inference_testSet(MODEL_TFLITE, model_type='Quantized')

!pip install ffmpeg-python &> 0
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
!pip install librosa
import librosa
import scipy.io.wavfile
!git clone https://github.com/petewarden/extract_loudest_section.git
!make -C extract_loudest_section/
print("Packages Imported, Extract_Loudest_Section Built")

# Helper function to run inference (on a single input this time)
# Note: this also includes additional manual pre-processing
TF_SESS = tf.compat.v1.InteractiveSession()
def run_tflite_inference_singleFile(tflite_model_path, custom_audio, sr_custom_audio, model_type="Float"):
  #
  # Preprocess the sample to get the features we pass to the model
  #
  # First re-sample to the needed rate (and convert to mono if needed)
  custom_audio_resampled = librosa.resample(y = librosa.to_mono(np.float64(custom_audio)), orig_sr = sr_custom_audio, target_sr = SAMPLE_RATE)
  # Then extract the loudest one second
  scipy.io.wavfile.write('custom_audio.wav', SAMPLE_RATE, np.int16(custom_audio_resampled))
  !/tmp/extract_loudest_section/gen/bin/extract_loudest_section custom_audio.wav ./trimmed
  # Finally pass it through the TFLiteMicro preprocessor to produce the 
  # spectrogram/MFCC input that the model expects
  custom_model_settings = models.prepare_model_settings(
      0, SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
      WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
  custom_audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0,
                                                    model_settings, None)
  custom_audio_preprocessed = custom_audio_processor.get_features_for_wav(
                                        'trimmed/custom_audio.wav', model_settings, TF_SESS)
  # Reshape the output into a 1,1960 matrix as that is what the model expects
  custom_audio_input = custom_audio_preprocessed[0].flatten()
  test_data = np.reshape(custom_audio_input,(1,len(custom_audio_input)))

  #
  # Initialize the interpreter
  #
  interpreter = tf.lite.Interpreter(tflite_model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  #
  # For quantized models, manually quantize the input data from float to integer
  #
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  #
  # Run the interpreter
  #
  interpreter.set_tensor(input_details["index"], test_data)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details["index"])[0]
  top_prediction = output.argmax()

  #
  # Translate the output
  #
  top_prediction_str = ''
  if top_prediction >= 2:
    top_prediction_str = WANTED_WORDS.split(',')[top_prediction-2]
  elif top_prediction == 0:
    top_prediction_str = 'silence'
  else:
    top_prediction_str = 'unknown'

  print('%s model guessed the value to be %s' % (model_type, top_prediction_str))


AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };            
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {            
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data); 
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});
      
</script>
"""

def get_audio():
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  
  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav', ac='1')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)
  
  riff_chunk_size = len(output) - 8
  # Break up the chunk size into four bytes, held in b.
  q = riff_chunk_size
  b = []
  for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)

  # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
  riff = output[:4] + bytes(b) + output[8:]

  sr, audio = wav_read(io.BytesIO(riff))

  return audio, sr
print("Chrome Audio Recorder Defined")

custom_audio, sr_custom_audio = get_audio()
print("DONE")

# Then test the model
run_tflite_inference_singleFile(MODEL_TFLITE, custom_audio, sr_custom_audio, model_type="Quantized")

# answers: 
WANTED_WORDS = "yes,no"

TRAINING_STEPS = "12000,3000"

LEARNING_RATE = "0.001,0.0001"

more_reading = https://colab.research.google.com/github/tinyMLx/colabs/blob/master/3-5-18-TrainingKeywordSpotting.ipynb
