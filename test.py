import cv2
import numpy as np
import os


folder_path = 'image' 
filename = 'alpha/alpha_1.png'
file_path = os.path.join(folder_path, filename)
# Đọc hình ảnh từ file (đảm bảo rằng bạn đã cài đặt OpenCV trước)
img = cv2.imread(file_path, 0)  # Đọc ảnh dưới dạng ảnh xám

# Tăng độ tương phản sử dụng phép biến đổi histogram
kernel = np.ones((2, 2), np.uint8)  # Đây là kernel có kích thước 5x5, bạn có thể điều chỉnh kích thước nếu cần thiết
# Dilation để tăng độ dày của nét vẽ lên 2 pixel
dilated_img = cv2.dilate(img, kernel, iterations=1)

new_width = img.shape[1] * 20
new_height = img.shape[0] * 20
# Hiển thị ảnh gốc và ảnh đã xử lý
resized_img = cv2.resize(dilated_img, (new_width, new_height))
# Hiển thị ảnh mới
cv2.imshow('Resized Image', resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()