import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from NeuralNetworkMLP.NeuralNetwork import NeuralNetwork

model = NeuralNetwork(modelFile="model.json")

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas_size = 280
        self.image_size = 28

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Predict", command=self.predict_digit).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT)

        self.prediction_label = tk.Label(root, text="Draw a digit and click Predict")
        self.prediction_label.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        radius = 8
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.prediction_label.config(text="Canvas cleared")

    def preprocess_image(self):
        img_resized = self.image.resize((self.image_size, self.image_size))
        img_inverted = ImageOps.invert(img_resized)
        img_array = np.array(img_inverted).astype(np.float32) / 255.0
        return img_array.reshape(1, -1)

    def predict_digit(self):
        input_data = self.preprocess_image()
        outputs, _ = model.predict(input_data)
        probabilities = outputs[-1].flatten()
        predicted_digit = np.argmax(probabilities)
        prob_str = "\n".join([f"{i}: {p:.2%}" for i, p in enumerate(probabilities)])

        self.prediction_label.config(
            text=f"Prediction: {predicted_digit}\n\nProbabilities:\n{prob_str}"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
