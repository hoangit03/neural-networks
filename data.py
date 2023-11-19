import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train = []
Y_train = []
X_test = []
Y_test = []
folder_path = 'image' 



for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        height, width, channels = image.shape
        pixel = []
        label = filename.split("_")[0]
        Y_train.append(label)

        for i in range(height):
            for j in range(width):
                pixel.append(image[i][j][0])
        X_train.append(pixel)
X_train = np.array(X_train)  
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)




