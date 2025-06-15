# 🤟 Sign Language to Speech Translator 🗣️  
Translate sign language gestures (using both hands) into real-time speech using computer vision and deep learning.

Built with ❤️ using **TensorFlow**, **MediaPipe**, **OpenCV**, and **pyttsx3**.

---

## 📁 Project Structure

```
SIGN_LANGUAGE_TO_SPEECH/
├── accuracy/
│   ├── accuracy_loss.png            # Training accuracy/loss plot
│   ├── test.txt                     # Test log or label list
│   └── train.txt                    # Training log or label list
├── data/
│   └── gesture_landmarks_both_hands.csv  # Collected hand gesture landmarks
├── model/
│   ├── gesture_dl_model.keras      # Trained DL model
│   └── label_encoder.pkl           # Label encoder for gestures
├── total_signs/
│   ├── display_signs.py            # Script to display supported signs
│   └── toal_signs.txt              # Text file with sign labels
├── app.py                          # Main sign-to-speech inference script
├── collect_data.py                 # Data collection script for gestures
├── train_dl_model.py               # Model training script
├── LICENSE                         # License file
└── README.md                       # Project documentation
```

---

## 🛠️ How It Works

1. **Data Collection (`collect_data.py`)**  
   Capture gesture landmark coordinates for both hands using MediaPipe and store them in a CSV file.

2. **Model Training (`train_dl_model.py`)**  
   Trains a deep learning classifier (Dense NN) on the gesture landmark data and saves the model and label encoder.

3. **Real-time Inference (`app.py`)**  
   Detects gestures live from the webcam, classifies them using the trained model, and speaks out the recognized gesture using `pyttsx3`.

---

## 📦 Requirements

Install the required Python libraries:

```bash
pip install tensorflow mediapipe opencv-python numpy pandas matplotlib pyttsx3 scikit-learn
```

---

## 🚀 Quick Start

### 🔹 1. Collect Gesture Data

```bash
python collect_data.py
```
Follow the prompts to record samples for a gesture. Make sure both hands are visible.

### 🔹 2. Train the Model

```bash
python train_dl_model.py
```
This trains the model and saves it under the `model/` directory.

### 🔹 3. Run the Translator

```bash
python app.py
```
The app waits for a gesture, records for 2 seconds, predicts the gesture, and converts it to speech.

---

## 📊 Visualization

![Training Accuracy and Loss](accuracy/accuracy_loss.png)

---

## 📂 Add/Update Supported Gestures

1. Add more samples using `collect_data.py`.
2. Retrain the model with `train_dl_model.py`.
3. Keep `toal_signs.txt` updated with the list of gestures.

---

## 🗣️ Example Use Cases

- Accessibility aid for speech-impaired individuals  
- Educational tool for learning sign language  
- Real-time communication bridge using gesture recognition

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♀️ Author

**Divya Khunt**  
ML & DL Enthusiast | B.Tech in Engineering

---

## 🌟 Show your support

If you like this project, leave a ⭐ on [GitHub](#) and share it with others!
