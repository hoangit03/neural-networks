import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x * 1e-3 , x)

def relu_derivative(x):
    return np.where(x > 0, 1, 1e-3)

def tanh(x):
  return 2 / (1 + np.exp(-2 * x)) - 1

def tanh_derivative(x):
    return 1 - tanh(x)**2

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.rand(784, 128)
        self.b1 = np.zeros((1, 128))

        self.w2 = np.random.rand(128, 32)
        self.b2 = np.zeros((1, 32))

        self.w_output = np.random.rand(32, 10)
        self.b_output = np.zeros((1, 10))

    def forward(self, input_data):
        self.layer1 = tanh(input_data @ self.w1 + self.b1)
        self.layer2 = tanh(self.layer1 @ self.w2 + self.b2)
        self.output = sigmoid(self.layer2 @ self.w_output + self.b_output)
        return self.output

    def backpropagate(self, input_data, error, learning_rate):
        output_delta = error * sigmoid_derivative(self.output)
        # print(self.output, sigmoid_derivative(self.output))

        layer2_error = output_delta @ self.w_output.T
        layer2_delta = layer2_error * tanh_derivative(self.layer2)

        layer1_error = layer2_delta @ self.w2.T
        layer1_delta = layer1_error * tanh_derivative(self.layer1)

        self.w_output += self.layer2.T @ output_delta * learning_rate
        self.b_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.w2 += self.layer1.T @ layer2_delta * learning_rate
        self.b2 += np.sum(layer2_delta, axis=0, keepdims=True) * learning_rate

        self.w1 += input_data.T @ layer1_delta * learning_rate
        self.b1 += np.sum(layer1_delta, axis=0, keepdims=True) * learning_rate


    def train(self, input_data, target_output, test_data, test_label, learning_rate=0.01, epochs=1000, ):
        self.loss = []
        N = len(target_output)
        for epoch in range(epochs):
            self.output = self.forward(input_data)
            error = target_output - self.output
            loss = (error**2).sum()**0.5 / N
            self.loss.append(loss)
            self.backpropagate(input_data, error, learning_rate)

            # Đánh giá
            output_test = self.forward(test_data)
            predictions = np.argmax(output_test, axis=1)
            accuracy = accuracy_score(test_label, predictions)
            if epoch % 100 == 99:
              print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
