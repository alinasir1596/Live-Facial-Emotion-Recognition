# Facial Emotion Recognition Flask Endpoint

This repository contains code for a Flask-based endpoint that performs real-time facial emotion recognition using a trained deep learning model. The endpoint captures video from your camera, detects faces, and predicts the emotion displayed on each detected face. Additionally, it generates a real-time graph showing the frequency of recognized emotions.

## Prerequisites

Before running the code, make sure you have the following prerequisites installed on your system:

- Python 3.x
- Flask
- OpenCV (cv2)
- TensorFlow
- Numpy
- Matplotlib

You can install these dependencies using `pip`:

```bash
pip install Flask opencv-python tensorflow numpy matplotlib
```

## Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/facial-emotion-recognition.git
   cd facial-emotion-recognition
   ```

2. Run the Flask application by executing `main.py`:

   ```bash
   python main.py
   ```

3. Open a web browser and navigate to `http://localhost:5000` to access the application.

## Endpoint Routes

- `/`: Displays the homepage with a brief description of the application.
- `/index`: Shows the main page where real-time emotion recognition takes place.
- `/video_feed`: Provides a video feed of the camera with emotion recognition overlays.

## Files

- `main.py`: The main Flask application script.
- `model.py`: Contains the code for loading a pre-trained emotion recognition model.
- `camera.py`: Handles video capture, face detection, emotion prediction, and graph generation.

## Emotions Recognized

The model can recognize the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Additional Notes

- The pre-trained model (`model.json` and `model_weights.h5`) should be present in the same directory as `model.py`. You may need to replace them with your own pre-trained model if you have one.

- The Haar Cascade classifier file (`haarcascade_frontalface_default.xml`) should also be in the same directory as `camera.py` to enable face detection.

- The application generates a dynamic graph (`static/graph.png`) that shows the frequency of recognized emotions in real-time.

## Acknowledgments

This project uses the following libraries and technologies:

- Flask: A micro web framework for Python.
- OpenCV: An open-source computer vision library.
- TensorFlow: An open-source machine learning framework.
- Numpy: A library for numerical computations in Python.
- Matplotlib: A library for creating static, animated, and interactive visualizations in Python.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, report issues, or provide feedback to enhance this facial emotion recognition Flask endpoint. Happy coding!
