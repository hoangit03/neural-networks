import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import imgaug.augmenters as iaa


folder_path = 'image' 
filename = 'phi/phi_10.png'
file_path = os.path.join(folder_path, filename)
# Đọc hình ảnh từ file (đảm bảo rằng bạn đã cài đặt OpenCV trước)
image= cv2.imread(file_path, 0)  # Đọc ảnh dưới dạng ảnh xám

if image is not None:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Cải thiện độ tương phản
alpha = 1.7  # Điều chỉnh độ tương phản theo nhu cầu
beta = 40 # Điều chỉnh độ sáng theo nhu cầu
contrast_enhanced_image = cv2.convertScaleAbs(blurred_image, alpha=alpha, beta=beta)

# Làm sáng ảnh
# brightness_adjusted_image = cv2.addWeighted(contrast_enhanced_image, 1.3, contrast_enhanced_image, 0, 40)

# # Làm mịn ảnh
# smoothed_image = cv2.fastNlMeansDenoisingColored(brightness_adjusted_image,None,10,10,7,21)

# Hiển thị ảnh ban đầu và ảnh đã xử lý
new_width = image.shape[1] * 20
new_height = image.shape[0] * 20


kernel = np.ones((2, 2), np.uint8) 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))




# dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)



# contrasted_image = clahe.apply(smoothed_image)
# print(np.mean(dilated_img,axis=2))

# resized_img = cv2.resize(dilated_img, (new_width, new_height))
# cv2.imshow('Original Image', resized_img)
# # cv2.imshow('Processed Image', smoothed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
height, width = image.shape[:2]
M_left = cv2.getRotationMatrix2D((width / 2, height / 2), 20, 1)
rotated_left = cv2.warpAffine(contrast_enhanced_image, M_left, (width, height))

# Tạo ma trận quay cho xoay 20 độ về bên phải
M_right = cv2.getRotationMatrix2D((width / 2, height / 2), -20, 1)
rotated_right = cv2.warpAffine(contrast_enhanced_image, M_right, (width, height))


ret, thresh_img = cv2.threshold(rotated_right, 127, 255, cv2.THRESH_BINARY)


# dilated_img = cv2.dilate(image, kernel, iterations=1)
dilated_img1 = cv2.dilate(thresh_img, kernel, iterations=1)
dilated_img2 = cv2.dilate(rotated_left, kernel, iterations=1)
dilated_img3 = cv2.dilate(rotated_right, kernel, iterations=1)

# Tăng độ tương phản sử dụng phép biến đổi histogram
# resized_img = cv2.resize(dilated_img, (new_width, new_height))/

dilated_img = []
dilated_img.extend([dilated_img1,dilated_img2,dilated_img3])
dilated_img = np.array(dilated_img)
new_arr = np.mean(dilated_img, axis=3)
print(new_arr[0])

# cv2.imshow('Original Image', resized_img)

# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Thực hiện resize ảnh về kích thước cố định
# resized_image = cv2.resize(gray_image, (100, 100))
# # Áp dụng Gaussian Blur để làm mịn ảnh
# blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
# # Áp dụng ngưỡng để chuyển ảnh về ảnh nhị phân
# _, threshold_img = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow(" dfg",threshold_img)


# # Hiển thị ảnh gốc và ảnh đã xử lý
# # resized_img = cv2.resize(dilated_img, (new_width, new_height))
# # resized_img1 = cv2.resize(thresh_img, (new_width, new_height))
# resized_img2 = cv2.resize(img, (new_width, new_height))
# # Hiển thị ảnh mới
# # cv2.imshow('Resized Image', resized_img2)
# # cv2.imshow('Resized Image', contrasted_image)
# # cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()