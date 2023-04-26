import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(dataset, path_dset='./datasets', num_components = None, preprocessing="standard"):
    """
        加载数据集
        param：
        dataset：str，数据集名称，可选值为 'indian_pines' 和 'salinas'
        path_dset：str，数据集所在路径，默认为'../HSI-datasets'

        return：
        包含数据和标签的字典，键分别为'data'和'label'
    """
    # 获取数据集所在目录的路径
    data_dir = os.path.join(os.getcwd(), path_dset)

    # 根据数据集名称确定数据集文件名和对应的键名
    if dataset == 'indian_pines':
        data_path = os.path.join(data_dir, 'Indian_pines_corrected.mat')
        label_path = os.path.join(data_dir, 'Indian_pines_gt.mat')
        data_key = 'indian_pines_corrected'
        label_key = 'indian_pines_gt'
    elif dataset == 'salinas':
        data_path = os.path.join(data_dir, 'Salinas_corrected.mat')
        label_path = os.path.join(data_dir, 'Salinas_gt.mat')
        data_key = 'salinas_corrected'
        label_key = 'salinas_gt'
    # else:
    #     # 如果数据集名称不正确，抛出异常
    #     raise ValueError('Invalid dataset name.')

    # num_class = 15 if name == "UH" else 9 if name in ["UP", "DUP", "DUPr"] else 16
    num_class = 16

    # 加载数据和标签
    data = loadmat(data_path)[data_key]
    label = loadmat(label_path)[label_key]

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])

    # 数据归一化处理
    # 根据参数选择不同的归一化方法
    if preprocessing == "standard": data = StandardScaler().fit_transform(data)
    elif preprocessing == "minmax": data = MinMaxScaler().fit_transform(data)
    elif preprocessing == "none": pass
    else: print("[WARNING] No preprocessing method selected")

    # 数据降维处理
    # 根据参数选择是否进行降维处理
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components

    data = data.reshape(shapeor)
    # 将数据和标签打包成一个字典并返回
    return data, label, num_class


def split_data(data, label, test_percent=0.2, random_state=42):
    """
    划分数据集为训练集和测试集

    参数:
    data: numpy array, 数据
    label: numpy array, 标签
    test_percent: float, 测试集比例，默认为0.2
    random_state: int, 随机数种子，默认为42

    返回:
    训练集和测试集的数据和标签，共四个变量，分别为train_data, test_data, train_label, test_label
    """

    train_data, test_data, train_label, test_label = train_test_split(data, label,
                                                                      test_size=test_percent,
                                                                      stratify=label,          # 保证训练集和测试集中各类别的比例与原始数据集中相同
                                                                      random_state=random_state)

    return train_data, test_data, train_label, test_label


def pad_with_zeros(X, margin=2):
    """
    将高光谱图像沿边缘填充零

    Parameters
    ----------
    X : numpy.ndarray
        高光谱图像
    margin : int, optional
        填充的大小，默认为2

    Returns
    -------
    numpy.ndarray
        填充后的数组
    """
    # 新建一个零矩阵，大小为原数组加上两倍的边缘宽度
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    # 计算偏移量
    x_offset = margin
    y_offset = margin
    # 将原数组复制到新矩阵的中心
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def create_image_cube(data, labels, window_size = 19, remove_zero_labels=True):
    """
    以data中每一个像素为中心，取一个大小为(window_size, window_size, data.shape[2])的立方体，
    将这些立方体合并成一个数组后输出。

    参数：
    data: 输入数据，形状为(m, n, k)，m和n表示行和列数，k表示通道数。
    labels: 输入标签，形状为(m, n)。
    window_size: 每个立方体的大小。

    返回值：
    patches_data: 返回一个形状为(num_patches, window_size, window_size, data.shape[2])的数组，
                  其中num_patches为立方体的总数。
    patches_labels: 返回一个形状为(num_patches,)的一维数组，对应patches_data中每个立方体的标签。
    """
    # 计算边缘宽度
    margin = int((window_size - 1) / 2)
    
    # 对输入数据进行边缘填充
    zero_padded_data = pad_with_zeros(data, margin=margin)
    
    # 创建立方体数组
    patches_data = np.zeros((data.shape[0] * data.shape[1], window_size, window_size, data.shape[2]))
    patches_labels = np.zeros((data.shape[0] * data.shape[1]))
    patch_index = 0
    
    # 对每个像素点构建立方体
    for r in range(margin, zero_padded_data.shape[0] - margin):
        for c in range(margin, zero_padded_data.shape[1] - margin):
            patch = zero_padded_data[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patches_labels[patch_index] = labels[r-margin, c-margin]
            patch_index = patch_index + 1
            
    # 如果移除零标签
    if remove_zero_labels:
        patches_data = patches_data[patches_labels>0,:,:,:]
        patches_labels = patches_labels[patches_labels>0]
        patches_labels -= 1

    return patches_data, patches_labels.astype("int")



if __name__ == "__main__":
    datasets = ['indian_pines', 'salinas']
    for dataset in datasets:
        data, label = load_data(dataset)
        print(f"Dataset: {dataset}")
        print(f"Data shape: {data['data'].shape}")
        print(f"Label shape: {label['label'].shape}")