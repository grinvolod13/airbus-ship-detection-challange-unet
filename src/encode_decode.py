from .const import *
import numpy as np


# https://www.kaggle.com/code/stainsby/fast-tested-rle/notebook
def decode(mask_rle: np.ndarray):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    img=np.zeros(SIZE_FULL*SIZE_FULL, dtype=np.float32)
    if not(type(mask_rle) is float):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1.0
    return img.reshape((SIZE_FULL, SIZE_FULL)).T


def encode(img: np.ndarray)->list[str]:
    """Encodes image to run-length string format

    Args:
        img (np.ndarray): image to encode

    Returns:
        list[str]: list of str items
    """
    flag = False
    out = []
    it = img.T.flat
    for i in range(img.size):
        if not flag and it[i]:
            out.append(str(i+1))
            flag = True
        elif flag and not it[i]:
            out.append(str(i-int(out[-1])+1))
            flag = False
    if flag:
        out.append(str(img.size-int(out[-1])))
    return out 

