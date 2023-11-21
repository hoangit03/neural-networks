import numpy as np
from sklearn.metrics import accuracy_score

np.seterr(over='ignore') 

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.rand(784, 256) * np.sqrt(2/784)
        self.b1 = np.zeros((1, 256))

        self.w2 = np.random.rand(256, 128) * np.sqrt(2/256)
        self.b2 = np.zeros((1, 128))

        self.w_output = np.random.rand(128, 24) * np.sqrt(2/128)  
        self.b_output = np.zeros((1, 24))

    def _tanh(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def _tanh_deriv(self, x):
        return 1 - self._tanh(x) ** 2

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu(self,x):
        return np.maximum(0, x)

    def _relu_deriv(self,x):
        return np.where(x > 0, 1, 0)

    def forward(self, input_data):
        self.layer1 = self._tanh(np.dot(input_data , self.w1) + self.b1)
        self.layer2 = self._tanh(np.dot(self.layer1 , self.w2) + self.b2)
        self.output = self._sigmoid(np.dot(self.layer2 , self.w_output) + self.b_output)
        return self.output

    def backpropagate(self, input_data, error, learning_rate):
        output_delta = error * self._sigmoid_deriv(self.output)

        layer2_error = np.dot(output_delta , self.w_output.T)
        layer2_delta = layer2_error * self._tanh_deriv(self.layer2)
        layer1_error = np.dot(layer2_delta , self.w2.T)
        layer1_delta = layer1_error * self._tanh_deriv(self.layer1)

        self.w_output += np.dot(self.layer2.T , output_delta) * learning_rate
        self.b_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.w2 += np.dot(self.layer1.T , layer2_delta) * learning_rate
        self.b2 += np.sum(layer2_delta, axis=0, keepdims=True) * learning_rate
        self.w1 += np.dot(input_data.T , layer1_delta) * learning_rate
        self.b1 += np.sum(layer1_delta, axis=0, keepdims=True) * learning_rate

    def train(self, input_data, target_output, test_data, test_label, learning_rate=0.001, epochs=1000):
        self.loss = []
        N = len(target_output)
        for epoch in range(epochs):
            self.output = self.forward(input_data)
            error = target_output - self.output
            loss = (error**2).sum()**0.5 / N
            self.loss.append(loss)
            self.backpropagate(input_data, error, learning_rate)

            output_test = self.forward(test_data)
            predictions = np.argmax(output_test, axis=1)
            accuracy = accuracy_score(test_label, predictions)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
