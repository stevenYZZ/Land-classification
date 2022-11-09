import torch
from torch import Tensor
from sklearn.metrics import accuracy_score


def IoU(img1: Tensor, img2: Tensor):
    """
    conpute the Intersection over Union score of the input images(Binary image)
    size example: (256, 256); (1, 256, 256); (100, 256, 256)
    the true value of pixel must be greater than 0
    the false value of pixel must be equal to 0

    Parameters:
        img1, img2 - the input imgs (Tensor)

    Returns:
	    Return the average IoU score (Tensor)
    """
    assert img1.size() == img2.size(), "two imgs have different size"
    assert img1.ndim == 2 or img1.ndim == 3, "wrong number of dimension"
    if(img1.ndim == 2):
        intersection = torch.sum(img1 * img2)
        union = img1.sum() + img2.sum() - intersection
    elif(img1.ndim == 3):
        intersection = torch.sum(img1 * img2, dim = (1, 2))
        union = img1.sum(dim = (1, 2)) + img2.sum(dim = (1, 2)) - intersection

    return (intersection / union).mean()


def IoU_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """
    conpute the IoU loss of the input images(Binary image)
    fomula : IoU_loss = 1 - IoU
    size example: (256, 256); (1, 256, 256); (100, 256, 256)
    the true value of pixel must be greater than 0
    the false value of pixel must be equal to 0

    Parameters:
        input, target - the input imgs (Tensor)
        multiclass - not consider this situation for the time being

    Returns:
	    Return the IoU Loss (Tensor)
    """
    return 1 - IoU(input, target)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)



if __name__ == "__main__":
    a = torch.Tensor((1, 0, 0, 1, 0, 0, 1, 0, 0)).long().reshape(1, 3, 3).expand(1, 3, 3)
    b = torch.Tensor((1, 0, 1, 1, 0, 1, 1, 0, 1)).long().reshape(1, 3, 3).expand(1, -1, -1)
    a = torch.Tensor((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)).long().reshape(-1, 2, 2)
    b = torch.Tensor((1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1)).long().reshape(-1, 2, 2)
    # a = torch.Tensor((1, 0, 0, 1, 0, 0, 1, 0, 0)).long().reshape(3, 3)
    # b = torch.Tensor((1, 0, 1, 1, 0, 1, 1, 0, 1)).long().reshape(3, 3)
    # print(IoU(a, b))
    print(a.dtype, b.dtype)
    # print(accuracy(b, a))

