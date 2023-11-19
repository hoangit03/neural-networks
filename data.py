import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

folder_path = 'image' 

for filename in os.listdir(folder_path):
    X_train = []
    Y_train = []
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        height, width, channels = image.shape
        # image = np.zeros((height, width), dtype=np.uint8)
        pixel = []
        label = filename.split("_")[0]
        Y_train.append(label)

        for i in range(height):
            for j in range(width):
                pixel.append(image[i][j][0])
        X_train.append(pixel)
       
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(np.array(X_train))
print(X_train)

       