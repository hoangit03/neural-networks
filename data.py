import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler



X_train = []
Y_train = []
X_test = []
Y_test = []
folder_path = 'image' 
folder_test = 'test'

label_file = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
for label in label_file:
    print(folder_path+"/"+label)
def get_Dataset(folder,X_train,Y_train):
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  
            file_path = os.path.join(folder, filename)
            image = cv2.imread(file_path)
            height, width, channels = image.shape
            pixel = []
            label = label_file.index(filename.split("_")[0])
            Y_train.append(label)
            for i in range(height):
                for j in range(width):
                    pixel.append(image[i][j][0])
            X_train.append(pixel)
    return X_train, Y_train




X_train, Y_train = get_Dataset(folder_path,X_train,Y_train)

X_train = np.array(X_train)  
# y_one_hot = np.zeros((len(Y_train), 24))
# for i in range(len(Y_train)):
#     y_one_hot[i, Y_train[i]] = 1
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)






