import cv2
import numpy as np
from skimage.color import label2rgb


def RGBImage(y_pred):
    '''
        输入预测结果，输出RGB图像
        y_pred: 2D array, shape = (145,145)
        return: 3D array, shape = (145,145,3)
    '''
    # color_map = np.array([[0, 0, 0],  # Background, black
    #                   [128, 0, 0],  # Aeroplane, maroon
    #                   [0, 128, 0],  # Bicycle, green
    #                   [128, 128, 0],  # Bird, olive
    #                   [0, 0, 128],  # Boat, navy
    #                   [128, 0, 128],  # Bottle, purple
    #                   [0, 128, 128],  # Bus, teal
    #                   [128, 128, 128],  # Car, gray
    #                   [64, 0, 0],  # Cat, dark maroon
    #                   [192, 0, 0],  # Chair, bright maroon
    #                   [64, 128, 0],  # Cow, dark green
    #                   [192, 128, 0],  # Dining table, bright green
    #                   [64, 0, 128],  # Dog, dark purple
    #                   [192, 0, 128],  # Horse, bright purple
    #                   [64, 128, 128],  # Motorbike, dark teal
    #                   [192, 128, 128],  # Person, bright teal
    #                   [0, 64, 0],  # Potted plant, dark olive
    #                   [128, 64, 0],  # Sheep, maroon green
    #                   [0, 192, 0],  # Sofa, bright green
    #                   [128, 192, 0],  # Train, olive green
    #                   [0, 64, 128],  # TV, dark navy
    #                   ])

    # image_label_overlay = label2rgb(y_pred, colors=color_map)
    image_label_overlay = label2rgb(y_pred)
    image_label_overlay = np.uint8(image_label_overlay * 255)
    return image_label_overlay  

