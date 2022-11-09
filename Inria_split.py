import os
# from PIL import Image
from pathlib import Path
import cv2
import math
from tqdm import tqdm

import shutil


input_path = Path("../building-map-generator-datasets/AerialImageDataset")
output_path = Path("../building-map-generator-datasets/Inria_256")



sub_image_size = 256


def split(sub_image_size, input_dir, output_dir, isGrey = False):
    """
        划分储存在input_dir的Infia数据集图像, 并保存在output_dir
        划分方式为，去除原图像多余的边缘，中间部分分割成不重叠的子图
        Parameters:
            sub_image_size: the size of sub image
            input_dir: the input path
            output_dir: the output path
            isGrey: using True if the image is Grey image, default False(RGB)
        Returns:
            no return
    """
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    a = []

    with tqdm( total = len(os.listdir(input_dir))) as pbar:
        for name in os.listdir(input_dir):
            img = cv2.imread(str(input_dir/name), 0 if isGrey else 1)
        
            height = img.shape[0]
            width = img.shape[1]
            row_num = math.floor(height / sub_image_size)
            col_num = math.floor(width / sub_image_size)
            start_y = math.floor((height - row_num * sub_image_size) / 2)
            start_x = math.floor((width - col_num * sub_image_size) / 2)
            for i in range(row_num):
                for j in range(col_num):
                    if isGrey :
                        new_img = img[start_y  + i * sub_image_size : start_y + (i + 1) * sub_image_size,
                                    start_x  + j * sub_image_size : start_x + (j + 1) * sub_image_size ]
                    else:
                        new_img = img[start_y  + i * sub_image_size : start_y + (i + 1) * sub_image_size,
                                    start_x  + j * sub_image_size : start_x + (j + 1) * sub_image_size,
                                    : ]
                    new_name = name[:-4] + str(i) + "_" + str(j) + ".png"
                    # a.append(new_img)
                    # print(start_y)
                    # print(start_y + i * sub_image_size)
                    # print(new_img.shape)
                    
                    cv2.imwrite(str(output_dir/new_name), new_img)
            pbar.update(1)
    # print(len(a))



if __name__ == "__main__":

    in_test_path = input_path/"test"/"images"
    in_train_path = input_path/"train"/"images"
    in_ground_truth_path = input_path/"train"/"gt"

    out_test_path = output_path/"test"/"images"
    out_train_path = output_path/"train"/"images"
    out_ground_truth_path = output_path/"train"/"gt"
    
    # split(sub_image_size, in_train_path, out_train_path)
    split(sub_image_size, in_ground_truth_path, out_ground_truth_path, 1)
    split(sub_image_size, in_train_path, out_train_path, 0)







# try_path1 = test_path/os.listdir(test_path)[0]
# try_path2 = train_path/os.listdir(train_path)[0]
# try_path3 = ground_truth_path/os.listdir(ground_truth_path)[0]

# img1 = cv2.imread(str(try_path1))
# img2 = cv2.imread(str(try_path2))
# img3 = cv2.imread(str(try_path3), 0)
# print(type(img1))
# print(type(img2))
# print(type(img3))
# print(img2.dtype)
# print(img3.dtype)

# print(img1.shape)
# print(img2.shape)
# print(img3.shape)

# print(img1.sum())
# print(img2.sum())
# print(img3.sum())

# print((img2[:,:,0] == img2[:, :, 1]).any())
# print(img3.max())



# output_path = Path("./data_all_256_origin/predict_datasetRotate/68_12.png")
# img = Image.open(output_path)
# # min_ = 2555
# # max_ = -1
# c = []
# for i in range(img.size[0]):
#     for j in range(img.size[1]):
#         cur = img.getpixel((i, j))
#         # max_ = max(cur, max_)
#         # min_ = min(cur, min_)\
#         if not cur in c:
#             c.append(cur)
# print(c)
# print(img.size)


