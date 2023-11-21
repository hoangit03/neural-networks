import numpy as np
import matplotlib.pyplot as plt

from data import X_train, Y_train_one_hot, X_test,Y_test
from neural_networks import NeuralNetwork

# Khởi tạo mô hình neural network
model = NeuralNetwork()

# Normalize dữ liệu đầu vào
X_train_normalized = X_train 
X_test_normalized = X_test 

# Huấn luyện mô hình

model.train(X_train_normalized, Y_train_one_hot, X_test_normalized, Y_test, learning_rate=0.001, epochs=1000)


plt.plot(range(len(model.loss)), model.loss, color='blue', alpha=0.6)
plt.title('Loss Function')
plt.show()

