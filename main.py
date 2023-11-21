import numpy as np
import matplotlib.pyplot as plt

from data import X_train, y_one_hot, X_test,Y_test
from neural_networks import NeuralNetwork

net = NeuralNetwork()




net.train(X_train, y_one_hot,X_test,Y_test)

plt.plot(range(len(net.loss)), net.loss, color='blue', alpha=0.6)
plt.title('Loss Function')
plt.show()

# fig, ax = plt.subplots(5, 5)
# index = 0
# print(X_train)
# for i in range(5):
#   for j in range(5):
#     ax[i,j].imshow(X_train[index])
#     ax[i,j].set_axis_off()
#     index+=1
# plt.show()