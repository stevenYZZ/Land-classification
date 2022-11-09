import os
from PIL import Image
from pathlib import Path
import cv2
import math
from pathlib import Path
import shutil



input_path = Path("../building-map-generator-datasets/AerialImageDataset")
output_path = Path("../building-map-generator-datasets/Inria_256")/"train"/"images"

test_path = input_path/"test"/"images"
train_path = input_path/"train"/"images"
print(len(os.listdir(train_path)))
print(len(os.listdir(output_path)))

ground_truth_path = input_path/"train"/"gt"
# try_path1 = test_path/os.listdir(test_path)[0]
# try_path2 = train_path/os.listdir(train_path)[0]
# try_path3 = ground_truth_path/os.listdir(ground_truth_path)[0]

# img1 = cv2.imread(str(try_path1))
# img2 = cv2.imread(str(try_path2))
# img3 = cv2.imread(str(try_path3), 0)
# print(type(img1))
# print(type(img2))
# print(type(img3))
# test_path2 = output_path/"test"/"images"
# print(len(os.listdir(test_path)))
# print(len(os.listdir(test_path2)))

# if os.path.exists(output_path):
    # shutil.rmtree(output_path)
# os.makedirs(output_path)
# a = [1:100, 1:100]
# print(a)

# print(img2.dtype)
# print(img3.dtype)



