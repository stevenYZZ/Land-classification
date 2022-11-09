import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import math

# 创建dataset
class BasicDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, scale = 1):
        # super().__init__
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(data_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    
    @staticmethod
    def preprocess(pil_img, scale, is_label):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_label else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_label:
            if img_ndarray.ndim == 2:
                # print(img_ndarray.shape)
                img_ndarray = img_ndarray[np.newaxis, ...]
                # print(img_ndarray.shape)
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            # img_ndarray = img_ndarray / 255
        # if(is_label):
        #     for i in img_ndarray:
        #         for j in i:

        #             if(j == 0):
        #                 a = 1
        #             else:
        #                 b = 1
        
        
        if not is_label:    
            img_ndarray = img_ndarray / 255
        else:
            img_ndarray = img_ndarray / 255
            img_ndarray = img_ndarray.astype(int) 
        #     print(img_ndarray.dtype)
        #     img_ndarray = np.int64(img_ndarray > 0)
        #     print(img_ndarray.sum())
        # if(is_label):
        #     print(img_ndarray.sum())
        #     if((img_ndarray == np.zeros(img_ndarray.shape, dtype=img_ndarray.dtype)).all()):
        #         print("1")

        return img_ndarray


    # 重构 len 函数
    def __len__(self):
        return len(self.ids)
    

    # 重构 getitem 函数
    def __getitem__(self, idx):
        name = self.ids[idx]
        label_file = list(self.label_dir.glob(name + '.*'))
        data_file = list(self.data_dir.glob(name + '.*'))

        assert len(data_file) == 1, f'Either no data or multiple datas found for the ID {name}: {data_file}'
        assert len(label_file) == 1, f'Either no label or multiple label found for the ID {name}: {label_file}'
        
        data = Image.open(data_file[0])
        label = Image.open(label_file[0])

        assert data.size == label.size, \
            f'data and label {name} should be the same size, but are {data.size} and {label.size}'

        data = self.preprocess(data, self.scale, is_label=False)
        label = self.preprocess(label, self.scale, is_label=True)
        # print(label.sum())
        # print(label.dtype)
        # print(label.shape)
        # x = torch.as_tensor(label.copy())
        # print(x.sum())
        # print(x.dtype)
        # print(x.shape)
        # y = torch.as_tensor(label.copy()).bool()
        # print(y.sum())
        # print(y.dtype)
        # print(y.shape)
        return {
            'data': torch.as_tensor(data.copy()).float().contiguous(),
            'label': torch.as_tensor(label.copy()).long().contiguous()
        }


# class splitDataset(Dataset):
#     def __init__(self, data_dir: str, label_dir: str, sub_image_size, train_size, scale = 1):
#         self.data_dir = Path(data_dir)
#         self.label_dir = Path(label_dir)
#         self.sub_image_size = sub_image_size
#         assert 0 < scale <= 1, 'Scale must be between 0 and 1'
#         self.scale = scale
#         self.ids = []
#         files = listdir(label_dir)
#         assert len(files) <= train_size, f"train_size must less than {len(files)} images"

#         for i in range(train_size):
#             img = cv2.imread(str(label_dir/files[i]), 0)

#             height = img.shape[0]
#             width = img.shape[1]
#             row_num = math.floor(height / sub_image_size)
#             col_num = math.floor(width / sub_image_size)
#             start_y = math.floor((height - row_num * sub_image_size) / 2)
#             start_x = math.floor((width - col_num * sub_image_size) / 2)
#             for i in range(row_num):
#                 for j in range(col_num):

#                     if isGrey :
#                         new_img = img[start_y : start_y + (i + 1) * sub_image_size,
#                                     start_x : start_x + (j + 1) * sub_image_size ]

#         # self.ids = [splitext(file)[0] for file in listdir(data_dir) if not file.startswith('.')]
#         if not self.ids:
#             raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        
#         logging.info(f'Creating dataset with {len(self.ids)} examples')


