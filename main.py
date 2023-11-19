import numpy as np

# # Hàm kích hoạt - Rectified Linear Unit (ReLU)
# def relu(x):
#     return np.maximum(0, x)

# # Hàm softmax
# def softmax(x):
#     exp_x = np.exp(x - np.max(x))  # Tránh overflow
#     return exp_x / exp_x.sum(axis=0, keepdims=True)

# # Phân loại
# def forward_propagation(X, W1, b1, W2, b2):
#     # Hidden layer
#     Z1 = np.dot(W1, X) + b1
#     A1 = relu(Z1)

#     # Output layer
#     Z2 = np.dot(W2, A1) + b2
#     A2 = softmax(Z2)

#     return Z1, A1, Z2, A2

# # Hàm loss function - Cross-Entropy Loss
# def compute_loss(Y, Y_hat):
#     m = Y.shape[1]
#     loss = -1 / m * np.sum(Y * np.log(Y_hat + 1e-8))  # Thêm epsilon để tránh log(0)
#     return loss

# # Trọng số gradient descent
# def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate):
#     m = X.shape[1]

#     dZ2 = A2 - Y
#     dW2 = 1 / m * np.dot(dZ2, A1.T)
#     db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
#     dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
#     dW1 = 1 / m * np.dot(dZ1, X.T)
#     db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

#     # Gradient descent
#     W1 -= learning_rate * dW1
#     b1 -= learning_rate * db1
#     W2 -= learning_rate * dW2
#     b2 -= learning_rate * db2

#     return W1, b1, W2, b2

# # Khởi tạo các tham số mô hình
# np.random.seed(0)
# W1 = np.random.randn(100, 28*28) * 0.01  # Trọng số của hidden layer
# b1 = np.zeros((100, 1))  # Bias của hidden layer
# W2 = np.random.randn(10, 100) * 0.01  # Trọng số của output layer
# b2 = np.zeros((10, 1))  # Bias của output layer

# # Đầu vào
# X = np.random.rand(28*28, 1)

# # One-hot encode label cho ví dụ
# Y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T  # Ví dụ chỉ có một lớp đúng, được biểu diễn bằng one-hot encoding

# # Huấn luyện
# learning_rate = 0.01
# for i in range(1000):
#     Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
#     loss = compute_loss(Y, A2)
#     W1, b1, W2, b2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate)
#     if i % 100 == 0:
#         print(f"Loss after iteration {i}: {loss}")

# # Dự đoán
# def predict(X, W1, b1, W2, b2):
#     _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
#     return np.argmax(A2, axis=0)

# # Ví dụ dự đoán
# X_test = np.random.rand(28*28, 1)
# prediction = predict(X_test, W1, b1, W2, b2)
# print(f"Dự đoán: {prediction[0]}")
from data import X_train, Y_train

print(np.random.rand(784, 128))