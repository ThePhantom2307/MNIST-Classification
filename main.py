import numpy as np
from keras.datasets import mnist
from NeuralNetworkMLP.NeuralNetwork import NeuralNetwork, RELU, SOFTMAX

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (x_train.astype(np.float32) / 255.0).reshape(-1, 28 * 28)
x_test = (x_test.astype(np.float32) / 255.0).reshape(-1, 28 * 28)

x_val, y_val = x_train[-10000:], y_train[-10000:]
x_train, y_train = x_train[:-10000], y_train[:-10000]

num_classes = 10
eye = np.eye(num_classes, dtype=np.float32)
y_train_oh = eye[y_train]
y_val_oh = eye[y_val]
y_test_oh = eye[y_test]

print(f"[INFO] Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

model = NeuralNetwork(
    inputLayerNeurons=28 * 28,
    hiddenLayersNeurons=[512, 256],
    outputLayerNeurons=10,
    activationFunctions=[RELU, RELU, SOFTMAX]
)

model.setEpochs(75)
model.setBatchSize(32)
model.setLearningRate(2e-3)

model.train(
    trainingData=x_train,
    trainingLabels=y_train_oh,
    useThreshold=False,
    plotErrorsVsEpochs=False
)

val_acts, _ = model.predict(x_val)
val_probs = val_acts[-1]
val_pred = np.argmax(val_probs, axis=1)
val_acc = (val_pred == y_val).mean() * 100.0
print(f"[RESULT] Validation accuracy: {val_acc:.2f}%")

print("[INFO] Evaluating on test set...")
model.evaluation(x_test, y_test_oh)

model.saveModel("model.json")
print('[INFO] Saved model to "model.json"')
