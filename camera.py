# Importing necessary libraries
import cv2
from model import FacialExpressionModel
import numpy as np
import matplotlib.pyplot as plt

# Loading model and haarcascade file

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
emotions = {}


# Real Time Video Camera class

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            if pred in emotions:
                emotions[pred] += 1
            else:
                emotions[pred] = 1

            generate_graph()
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()


# Generate graph of emotions recognized

def generate_graph():
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(emotions.keys(), emotions.values(), width=0.4, color=['maroon', 'green', 'blue', 'cyan', 'black', 'red',
                                                                  'yellow'])

    x = np.arange(1, 10)
    plt.ylim([0, 100])

    plt.xlabel("Emotions")
    plt.ylabel('Emotions Frequency')
    plt.title("Emotions Frequency Chart")
    plt.savefig('static/graph.png')
