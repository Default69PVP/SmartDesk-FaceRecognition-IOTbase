import cv2
import os
import numpy as np

data_folder = "datasets"

# Tạo đối tượng nhận dạng khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()
training_count = 0

face_data = []
labels = []

# Lấy từng folder ra
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)

    # Kiểm tra đường dẫn xem có phải thư mục không hay là tệp
    if os.path.isdir(folder_path):
        label = int(folder_name)  # ID là tên thư mục

        # Lấy từng hình trong thư mục
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Đổi sang grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Thêm dữ liệu vào mảng face_data
            face_data.append(image)
            labels.append(label)

            training_count += 1

# Training
recognizer.train(face_data, np.array(labels))

recognizer.save("Trainer.yml")
cv2.destroyAllWindows()

print('Số ảnh:',training_count)
print("Training xong.")