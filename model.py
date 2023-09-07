# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


# List of classes of expressions to be recognized

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

# Initialize recognition function

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        self.preds = None
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model._make_predict_function()

# Predicting emotion in realtime

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
