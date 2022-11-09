import os
import shutil
from pathlib import Path
from PIL import Image

import time
from tqdm import tqdm

input_dir = Path("./data_all_256")
output_dir = Path("./data_all_256_rotate")


# 将目录的文件复制到指定目录
# 同时每个文件旋转90，180，270度后，保存到目标目录
def copy_demo(src_dir, dst_dir):
    """
    复制src_dir目录下的所有内容到dst_dir目录
    :param src_dir: 源文件目录
    :param dst_dir: 目标目录
    :return:
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if os.path.exists(src_dir):
        total = len(os.listdir(src_dir))
        count = 0
        pbar = tqdm(total = total, desc = f'{count} / {total}', unit = 'img')
        for file in os.listdir(src_dir):
            # pbar.update(images.shape[0])
            pbar.set_postfix({"正在处理的元素为": file})
            count += 1
            file_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            if os.path.isfile(os.path.join(src_dir, file)):
                shutil.copy(file_path, dst_path)
                img = Image.open(file_path)
                img1 = img.transpose(Image.ROTATE_90)
                img2 = img1.transpose(Image.ROTATE_90)
                img3 = img2.transpose(Image.ROTATE_90)
                img1.save(dst_path[0:-4] + "_1.png")
                img2.save(dst_path[0:-4] + "_2.png")
                img3.save(dst_path[0:-4] + "_3.png")
            else:
                copy_demo(file_path, dst_path)
    


if __name__ == "__main__":
    copy_demo(input_dir, output_dir)
    # path = "./data_all_256_rotate/"
    # print( len(os.listdir(path + "label/train")) )
    # print( len(os.listdir(path + "label/val")) )
    # print( len(os.listdir(path + "data/train")) )
    # print( len(os.listdir(path + "data/val")) )
    # Image.open(path + "data/train/10_29_3.png")
    # print( len(os.listdir(path + "data/train")) + len(os.listdir(path + "data/val")) )
    


# #将源文件夹复制到新地址
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# if os.path.exists(input_dir):
#     for i in os.listdir(input_dir):
#         if not os.path.exists(output_dir/i):
#             os.makedirs(output_dir/i)
#         for file in i:
#             file_path = os.path.join(src_dir, file)
#         #在新地址上每个文件旋转后复制一份 共复制3次
        
#         dst_path = os.path.join(dst_dir, file)
#         if os.path.isfile(os.path.join(src_dir, file)):
#             copyfile(file_path, dst_path)
#         else:
#             copy_demo(file_path, dst_path)
# # os.makedirs(output_dir)
# shutil.copy(input_dir, output_dir)





