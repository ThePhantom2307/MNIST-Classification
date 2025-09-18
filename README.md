
# Handwritten Digit Recognition (MNIST)

This project implements a simple feedforward neural network (MLP) to recognize handwritten digits from the **MNIST dataset**.

It includes both a training script and a Tkinter-based GUI application where you can draw digits and get real-time predictions.

- The project is using the MLP neural network library that I've made you can find it on:
	- Github: https://github.com/ThePhantom2307/MLP-Neural-Network
	- PyPi: https://pypi.org/project/NeuralNetworkMLP/

## Project Structure

```
├── main.py # Training script (loads MNIST, trains, saves model)
├── predict.py # GUI app for digit recognition
├── requirements.txt # Python dependencies
├── model.json # Saved trained model
```

## Features

- Train a fully connected neural network (MLP) on MNIST.
- The trained model achieves ~97% accuracy (This accuracy is based on the **Training Details**).
- Save and load the model (`model.json`).
- GUI app where you can:
	- Draw digits on a canvas.
	- Predict the digit with probability distribution.
	- Clear and redraw easily.

## Training Details

The model was trained on MNIST with the following setup:
-  **Input size:** 784 (28x28 flattened images)
-  **Hidden layers:** 512 and 256 neurons (ReLU activation)
-  **Output layer:** 10 neurons (Softmax activation)
-  **Batch size:** 32
-  **Epochs:** 75
-  **Learning rate:** 0.002

### Results
- Validation accuracy: **96.94%**
- Test accuracy: **96.93%**

The trained model is saved to `model.json`.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script:

```bash
python main.py
```

This will:
- Load the MNIST dataset.
- Train the neural network.
- Save the trained model to `model.json`.

## Running the Digit Recognizer GUI

Once the model is trained (or `model.json` is present), launch the GUI:

```bash
python predict.py
```

### Usage

- Draw a digit (0–9) on the canvas with your mouse.
- Click **Predict** to see the model’s prediction and probabilities.
- Click **Clear** to reset the canvas.

## Requirements

The project depends on:
-  `numpy>=1.21.0`
-  `keras>=2.4.0`
-  `tensorflow>=2.4.0`
-  `NeuralNetworkMLP>=1.1.2`
-  `Pillow` (for image processing in the GUI)
-  `tkinter` (usually included with Python)

Install them via:

```bash
pip install -r requirements.txt
```

## Notes

- The trained model (`model.json`) is generated after training and should be included if you want to run predictions without retraining.
