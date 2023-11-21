import os
import cv2
import numpy as np




X_train = []
Y_train = []
X_test = []
Y_test = []
folder_path = 'image' 
folder_test = 'data_test'


label_file = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
def get_Dataset(folder,X_train,Y_train):
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  
            file_path = os.path.join(folder, filename)
            image = cv2.imread(file_path)
            kernel = np.ones((2, 2), np.uint8)
            dilated_img = cv2.dilate(image, kernel, iterations=1)
            height, width, channels = dilated_img.shape
            pixel = []
            label = label_file.index(filename.split("_")[0])
            Y_train.append(label)
            for i in range(height):
                for j in range(width):
                    pixel.append(image[i][j][0])
            X_train.append(pixel)
    return X_train, Y_train

for label in label_file:
    X_test, Y_test = get_Dataset(folder_test+"/"+label,X_test,Y_test)

for label in label_file:
    X_train, Y_train = get_Dataset(folder_path+"/"+label,X_train,Y_train)

X_test = np.array(X_test) 
X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
X_test[X_test > 0] = 1
X_train[X_train > 0] = 1


permutation_test = np.random.permutation(len(X_test))
permutation_train = np.random.permutation(len(X_train))

X_train = X_train[permutation_train]
Y_train = Y_train[permutation_train]
X_test = X_test[permutation_test]
Y_test = Y_test[permutation_test]


y_one_hot = np.zeros((len(Y_train), 24))
for i in range(len(Y_train)):
    y_one_hot[i, Y_train[i]] = 1
