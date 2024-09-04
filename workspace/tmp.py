import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter
from skimage import morphology

def load_tensor222(file):

    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."

    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)
    # print(ndims, dims, dtype)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    elif dtype == 3:
          np_dtype = np.uint8
          return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"

    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


def load_tensor(file):

    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."

    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)
    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"

    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)

def gaussian_smooth(x, sigma=4):
    im_blur = np.zeros(x.shape, dtype=np.float32)
    bs = x.shape[0]
    for idx in range(bs):
        im_blur[idx] = gaussian_filter(x[idx], sigma=sigma)

    return im_blur


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())



if __name__ == "__main__":
    output_file = r"/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/inf_tmp/output"
    output_array_device = r"/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/inf_tmp/output_array_device_used"
    
    # output_array_device_opening_f = r"/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/inf/mask_array_opening_out_device"
    outpath = r'/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/inf_tmp/out-device2py222.png'
    
    # img = load_tensor(output_file)
    # opening_img = load_tensor222(output_array_device_opening_f)
    # print(opening_img.max())
    
    threshold = 0.5
    
    img = load_tensor(output_file)
    mask = gaussian_smooth(img[0], sigma=4)
    
    mask = load_tensor(output_array_device)
    # print(mask.max(), mask.min())
    mask = rescale(mask).squeeze()
    
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *=255
    cv2.imwrite(outpath, mask)