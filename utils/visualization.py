import cv2
import numpy as np
from skimage.color import label2rgb


def RGBImage(y_pred):
    '''
        输入预测结果，输出RGB图像
        y_pred: 2D array, shape = (145,145)
        return: 3D array, shape = (145,145,3)
    '''
    # color_map = np.array([[26, 82, 118], [138, 5, 190], [248, 0, 137], [0, 184, 148], [253, 229, 16],
    #                   [195, 98, 17], [23, 191, 99], [255, 80, 80], [138, 43, 226], [230, 126, 34],
    #                   [0, 206, 209], [255, 182, 193], [75, 0, 130], [176, 224, 230], [255, 69, 0], [210, 105, 30]])
    image_label_overlay = label2rgb(y_pred)
    image_label_overlay = np.uint8(image_label_overlay * 255)
    return image_label_overlay  

