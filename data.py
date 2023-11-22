import os
import cv2
import numpy as np
from sklearn.utils import shuffle

X_train = []
Y_train = []
X_test = []
Y_test = []
folder_path = 'image' 
folder_test = 'data_test'

label_file = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]

def process_image(image):
    alpha = 1.7  # Điều chỉnh độ tương phản theo nhu cầu
    beta = 40 # Điều chỉnh độ sáng theo nhu cầu
    contrast_enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    kernel = np.ones((3, 3), np.uint8)
    ret, thresh_img = cv2.threshold(contrast_enhanced_image, 127, 255, cv2.THRESH_BINARY)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
    height, width, channels = dilated_img.shape
    pixel = []
    for i in range(height):
        for j in range(width):
            pixel.append(image[i][j][0])
    return pixel


def get_dataset(folder, X, Y):
    for label in label_file:
        for filename in os.listdir(folder + "/" + label):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(folder + "/" + label, filename)
                image = cv2.imread(file_path)
                processed_image = process_image(image)
                X.append(processed_image)
                Y.append(label_file.index(label))
    return X, Y

X_train, Y_train = get_dataset(folder_path, X_train, Y_train)
X_test, Y_test = get_dataset(folder_test, X_test, Y_test)

# Chuyển đổi sang numpy array
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Chuẩn hóa giá trị pixel về 0 hoặc 1
X_train[X_train > 0] = 1
X_test[X_test > 0] = 1

# Trộn dữ liệu
X_train, Y_train = shuffle(X_train, Y_train)
X_test, Y_test = shuffle(X_test, Y_test)

# One-hot encode nhãn
Y_train_one_hot = np.zeros((len(Y_train), 24))
for i in range(len(Y_train)):
    Y_train_one_hot[i, Y_train[i]] = 1
