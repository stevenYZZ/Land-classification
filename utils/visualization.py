import cv2
import numpy as np
from skimage.color import label2rgb

color_map_16 = [
    [128, 128, 128],  # 物体灰色
    [255, 0, 0],  # 红色
    [0, 255, 0],  # 绿色
    [0, 0, 255],  # 蓝色
    [255, 255, 0],  # 黄色
    [255, 0, 255],  # 洋红
    [0, 255, 255],  # 青色
    [255, 128, 0],  # 橙色
    [128, 255, 0],  # 鲜绿色
    [0, 128, 255],  # 天蓝色
    [128, 0, 255],  # 紫色
    [255, 0, 128],  # 粉红色
    [0, 255, 128],  # 青绿色
    [128, 0, 128],  # 紫红色
    [128, 128, 0],  # 橄榄色
    [0, 128, 128],  # 青灰色
]

color_map_17 = [
    [0, 0, 0],  # 背景黑色
    [128, 128, 128],  # 物体灰色
    [255, 0, 0],  # 红色
    [0, 255, 0],  # 绿色
    [0, 0, 255],  # 蓝色
    [255, 255, 0],  # 黄色
    [255, 0, 255],  # 洋红
    [0, 255, 255],  # 青色
    [255, 128, 0],  # 橙色
    [128, 255, 0],  # 鲜绿色
    [0, 128, 255],  # 天蓝色
    [128, 0, 255],  # 紫色
    [255, 0, 128],  # 粉红色
    [0, 255, 128],  # 青绿色
    [128, 0, 128],  # 紫红色
    [128, 128, 0],  # 橄榄色
    [0, 128, 128],  # 青灰色
]
def RGBImage(y_pred):
    '''
        输入预测结果，输出RGB图像
        y_pred: 2D array, shape = (145,145)
        return: 3D array, shape = (145,145,3)
    '''
    image_label_overlay = label2rgb(y_pred, colors=color_map_16, bg_label=0)
    # image_label_overlay = label2rgb(y_pred)
    image_label_overlay = np.uint8(image_label_overlay * 255)
    image_label_overlay = np.uint8(image_label_overlay * 255)
    return image_label_overlay  

