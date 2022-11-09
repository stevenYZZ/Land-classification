# 划分数据集单图像或多时像数据集，数据集划分到flower_datas中，训练验证比例为8：2
import os
from shutil import copy
import random
from tqdm import tqdm
# from util import files

def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path) # 这个可以删除单个文件，不能删除文件夹
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            # if file_name != 'wibot.log':
            tf = os.path.join(dir_path, file_name)
            del_files(tf)

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


is_multi = 0    # 划分数据类型 0为单图像，1为多时像
seed = 4396     # 随机种子
number = 10000   # 取样数量

input_file_path = './data2_all_256_origin/'        # 输入文件路径
output_file_path = './data2_all_256/' # 输出文件路径

mkfile(output_file_path)
del_files(output_file_path)

# 划分比例，训练集 : 验证集 : 测试集 = 9 : 1 : 0
train_rate = 0.9  # 训练集比例
val_rate = 0.1  # 验证集比例
test_rate = 1 - train_rate - val_rate  # 测试集比例


# 源文件夹
input_file_path_data = input_file_path + 'data/'
input_file_path_label = input_file_path + 'label/'

# 新文件夹
train_path_data = output_file_path + 'data/train/'
val_path_data = output_file_path + 'data/val/'
test_path_data = output_file_path + 'data/test/'

train_path_label = output_file_path + 'label/train/'
val_path_label = output_file_path + 'label/val/'
test_path_label = output_file_path + 'label/test/'

# 创建所需文件夹
mkfile(train_path_data)
mkfile(val_path_data)
mkfile(test_path_data)
mkfile(train_path_label)
mkfile(val_path_label)
mkfile(test_path_label)


# 开始划分
random.seed(seed)
image_label = [cla for cla in os.listdir(input_file_path + 'label')]  # 获取全部label图像
total_num = min(len(image_label), number)
print("total num: %d" % (total_num))

# 划分数据集index

total_list = range(len(image_label))

val_index = random.sample(total_list, k=int(
    total_num * val_rate))  # 从total_list列表中随机抽取k个
new_total_list = [n for i, n in enumerate(
    total_list) if i not in val_index]  # 从total_list中剔除val_index
test_index = random.sample(new_total_list, k=int(
    total_num * test_rate))  # 从new_total_list列表中随机抽取k个
train_index = random.sample([n for i, n in enumerate(new_total_list) if n not in test_index], k=int(
    total_num * train_rate))   # 从new_total_list中剔除test_index，取训练集部分


with tqdm(total = len(image_label)) as pbar:
    for index, image in enumerate(image_label):
        pbar.update(1)
        # 划分验证集
        if index in val_index:
            data_image_path = input_file_path_data + image
            copy(data_image_path, val_path_data)

            label_image_path = input_file_path_label + image
            copy(label_image_path, val_path_label)
        # 划分测试集
        elif index in test_index:
            data_image_path = input_file_path_data + image
            copy(data_image_path, test_path_data)

            label_image_path = input_file_path_label + image
            copy(label_image_path, test_path_label)
        # 划分训练集
        elif index in train_index:
            data_image_path = input_file_path_data + image
            copy(data_image_path, train_path_data)

            label_image_path = input_file_path_label + image
            copy(label_image_path, train_path_label)

# 划分数据集file
# 多时像数据划分
# if (is_multi):
#     for index, image in enumerate(image_clear):
#         # 划分验证集
#         if index in val_index:
#             clear_image_path = file_path_clear + image
#             copy(clear_image_path, val_path_clear)  # 将选中的图像复制到新路径
#             for i in range(3):
#                 cloudy_image_path = file_path_cloudy + \
#                     image[:-4] + '_' + str(i) + '.jpg'
#                 copy(cloudy_image_path, val_path_cloudy)
#         # 划分测试集
#         elif index in test_index:
#             clear_image_path = file_path_clear + image
#             copy(clear_image_path, test_path_clear)
#             for i in range(3):
#                 cloudy_image_path = file_path_cloudy + \
#                     image[:-4] + '_' + str(i) + '.jpg'
#                 copy(cloudy_image_path, test_path_cloudy)
#         # 划分训练集
#         else:
#             clear_image_path = file_path_clear + image
#             copy(clear_image_path, train_path_clear)
#             for i in range(3):
#                 cloudy_image_path = file_path_cloudy + \
#                     image[:-4] + '_' + str(i) + '.jpg'
#                 copy(cloudy_image_path, train_path_cloudy)
# # 单图像数据划分
# else:


print("[label dir]\ttest: %d, val:%d, train:%d" % (len(os.listdir(test_path_label)), len(
    os.listdir(val_path_label)), len(os.listdir(train_path_label))))
print("[data dir]\ttest: %d, val:%d, train:%d" % (len(os.listdir(test_path_data)), len(
    os.listdir(val_path_data)), len(os.listdir(train_path_data))))
print("processing done!")
