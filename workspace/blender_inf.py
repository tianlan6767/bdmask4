# 以下代码是python中加载Tensor
import numpy as np

import torch.nn.functional  as F
from torchvision.ops import roi_align 
from onnxsim import simplify
from torch import nn
import math
import onnx
import torch
import cv2
from glob import glob
import os.path as osp

def load_tensor(file):

    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."

    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)
    print(ndims, dims, dtype)

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

def bytes2np(file, shape=()):
    # 假设二进制数据存储在data.bin文件中
    with open(file, 'rb') as f:
        # 读取二进制数据
        binary_data = f.read()

    # 将二进制数据转换为NumPy数组
    # 假设数据类型为float32，形状为(1000, 10)
    arr = np.frombuffer(binary_data, dtype=np.uint8).reshape(shape)
    return arr



def do_paste_mask(masks, boxes, img_h: int, img_w: int):

    device = masks.device
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = int(masks.shape[0]) 

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, int(img_y.size(1)), int(img_x.size(1)))
    gy = img_y[:, :, None].expand(N, int(img_y.size(1)), int(img_x.size(1)))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)
    
    return img_masks


def merge_bases(rois, coeffs):
    # merge predictions
    N, coeffs_b = map(int, coeffs.size())
    N, B, H, W = map(int,rois.size())
    coeffs = coeffs.view(-1, B, 14, 14) 
    coeffs = F.interpolate(coeffs, scale_factor=4,
                            mode='bilinear', align_corners=False).softmax(dim=1)
    masks_preds = (rois * coeffs).sum(dim=1)
    return masks_preds.view(-1, int(H * W))  


def blender(bases, box_feat):
    _, _, b_h, b_w = map(int, bases.shape)
    
        
    box_pred = box_feat[:, :5]
    top_feat = box_feat[:, 5:]
    rois = roi_align(
        bases,   
        box_pred, 
        output_size=(56, 56), 
        spatial_scale=(0.25),
        sampling_ratio=1,
        aligned=True
    )     # N*4*56*56
    pred_mask_logits = merge_bases(rois, top_feat).sigmoid()
    pred_mask_logits = pred_mask_logits.view(
    -1, 1, 56, 56)
    pred_mask_logits = do_paste_mask(pred_mask_logits, box_pred[:,1:], int(b_h*4), int(b_w*4))
    pred_mask_logits = pred_mask_logits.view(-1, int(b_h*4), int(b_w*4))
    return pred_mask_logits



if __name__=="__main__":
    np.set_printoptions(suppress=True)
    # base_input_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/base_data_61_23"
    # box_feat_input_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/box_feat61_23"
    # box_feat_output_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/box_feat8_3"

    # imgs = glob(r"/media/ps/data/train/LQ/LQ/bdmask/workspace/inf" + "/*.jpg")
    # for im in imgs:
    #     imn = "+" + osp.basename(im)
    #     nim = load_tensor(im).squeeze()
    #     print(nim)
    #     cv2.imwrite(f"/media/ps/data/train/LQ/LQ/bdmask/workspace/inf/{imn}", nim)
    # im = r"/media/ps/data/train/LQ/LQ/bdmask5-back/workspace/inf/1"
    # nim = load_tensor(im).squeeze()
    # cv2.imwrite(f"/media/ps/data/train/LQ/LQ/bdmask5-back/workspace/inf/1.jpg", nim)
    



    # base = torch.tensor(load_tensor(base_input_f))
    # box_feat = torch.tensor(load_tensor(box_feat_input_f))
    # box_feat = torch.tensor(load_tensor(box_feat_output_f))

    
    
    
    # base_input_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/pdata/bases.data"
    # box_input_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/pdata/boxes.data"
    # top_feat_input_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/pdata/top_feat.data"
    
    # base = torch.tensor(bytes2np(base_input_f, shape=(1,4,512,512)))
    # box = torch.tensor(bytes2np(box_input_f, shape=(37,5)))
    # feat = torch.tensor(bytes2np(top_feat_input_f, shape=(37,784)))
    
    
    # print(base[:,0,])
    
    # print(box_feat[0].shape)   
    # box = box_feat[:, :5]
    # feat = box_feat[:, 5:]
    
    # masks = blender(base, box_feat[0].unsqueeze(0))
    # masks = blender(base, box_feat)
    
    
    
    
    # mask_pred_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/mask_output_1_61_23"
    # feature_f = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/feature_device_8_3"
    # masks = torch.tensor(load_tensor(mask_pred_f))
    # fmasks = torch.tensor(load_tensor(feature_f))
    
    
    # masks = masks.squeeze().numpy()
    # # masks = np.where(masks > 0.5, 255, 0)
    # chs, rows, cols = np.where(masks>0.5)
    
    # for i in list(zip(chs, rows, cols)):
    #     print(i)
    
    
    
    # masks = masks.squeeze().numpy()
    # masks = np.where(masks > 0.5, 255, 0)
    # print(masks.shape)
    # for idx, m in enumerate(masks):
    #     print(m.shape)
    #     if np.count_nonzero(m):
    #         cv2.imwrite(f"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tres/image_mask2_{idx}.jpg", m)
        
    # fmasks = fmasks.squeeze().numpy()
    # for idx, m in enumerate(fmasks):
    #     print(m.shape)
    #     if np.count_nonzero(m):
    #         cv2.imwrite(f"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tres/image_mask_feature_{idx}.jpg", m)


    # output = r"/media/ps/data/train/LQ/LQ/bdmask5-back/workspace/inf/orig/used/output"
    # output = load_tensor(output).squeeze()
    # print(output.shape)
    # np.set_printoptions(suppress=True)
    # i = 1
    # for o in output:
    #     # print(o.shape)
    #     score = o[4]
    #     if score > 0.15:
    #         print(i,math.sqrt(o[4]),o[:5], "\n")
    #         i += 1
            # break


    input_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/1-input_1"
    input_2 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-input_2"
    input_2_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-input_1"

    ax_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/1-affin_matrix_device_1"
    ax_2 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-affin_matrix_device_2"




    output1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/1-orig-output_1"
    output2_2 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-orig-output_2"
    output2_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-orig-output_1"


    # output_zero_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/output_zero_1_1"
    # output_zero_2_2 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/output_zero_2_2"

    # output_zero_2_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/output_zero_2_1"



    output_nonms_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/1-output_array_device_nonms_1"
    output_nonms_2_2 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-output_array_device_nonms_2"

    output_nonms_2_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-output_array_device_nonms_1"


    output__hasnms_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/1-output_array_hasnms_1"
    output__hasnms_2_2 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-output_array_hasnms_2"

    output__hasnms_2_1 = r"/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-output_array_hasnms_1"

    


    input_1 = load_tensor(input_1)
    input_2 = load_tensor(input_2)
    input_2_1 = load_tensor(input_2_1)

    ax_1 = load_tensor(ax_1)
    ax_2 = load_tensor(ax_2)



    output1 = load_tensor(output1)
    output2_2 = load_tensor(output2_2)
    output2_1 = load_tensor(output2_1)


    # output_zero1 = load_tensor(output_zero_1)
    # output_zero_2_2 = load_tensor(output_zero_2_2)

    # output_zero_2_1 = load_tensor(output_zero_2_1)


    output_nonms_1 = load_tensor(output_nonms_1)
    output_nonms_2_2 = load_tensor(output_nonms_2_2)

    output_nonms_2_1 = load_tensor(output_nonms_2_1)

    
    output__hasnms_1 = load_tensor(output__hasnms_1)
    output__hasnms_2_2 = load_tensor(output__hasnms_2_2)

    output__hasnms_2_1 = load_tensor(output__hasnms_2_1)


    print("input_1 == input2", (input_1==input_2).all())
    print("input_1 == input2_1", (input_1==input_2_1).all())
    print("ax_1 == ax_2", (ax_1==ax_2).all())


    print("output1 == output2_2", (output2_2==output1).all())

    # print("output_zero1 == output_zero_2_2", (output_zero1==output_zero_2_2).all())
    # print("output_zero1 == output_zero_2_1", (output_zero1==output_zero_2_1).all())

    print("output_nonms_2_2 == output_nonms_1", (output_nonms_2_2==output_nonms_1).all())

    print("output__hasnms_1 == output__hasnms_2_2", (output__hasnms_1==output__hasnms_2_2).all())
    print("output__hasnms_2_2 == output__hasnms_2_1", (output__hasnms_2_2==output__hasnms_2_1).all())

    # output_zero1_ptr = output_zero1[0][0]
    # output_zero1 = output_zero1[0][1:].reshape((1024, 791))

    # output_zero_2_2_ptr = output_zero_2_2[0][0]
    # output_zero_2_2 = output_zero_2_2[0][1:].reshape((1024, 791))

    # output_zero_2_1_ptr = output_zero_2_1[0][0]
    # output_zero_2_1 = output_zero_2_1[0][1:].reshape((1024, 791))

    # output1 = output1.reshape((87296,814))
    # output2_2 = output2_2.reshape((87296,814))

    output_nonms_1_ptr = output_nonms_1[0][0]
    output_nonms1 = output_nonms_1[0][1:].reshape((1024, 791))

    output_nonms_2_2_ptr = output_nonms_2_2[0][0]
    output_nonms_2_2 = output_nonms_2_2[0][1:].reshape((1024, 791))

    output_nonms_2_1_ptr = output_nonms_2_1[0][0]
    output_nonms_2_1 = output_nonms_2_1[0][1:].reshape((1024, 791))

   

    output__hasnms_1_ptr = output__hasnms_1[0][0]
    output__hasnms_1 = output__hasnms_1[0][1:].reshape((1024, 791))

    output__hasnms_2_2_ptr = output__hasnms_2_2[0][0]
    output__hasnms_2_2 = output__hasnms_2_2[0][1:].reshape((1024, 791))

    output__hasnms_2_1_ptr = output__hasnms_2_1[0][0]
    output__hasnms_2_1 = output__hasnms_2_1[0][1:].reshape((1024, 791))


    print("***********",output1.shape, output2_2.shape, output_nonms1.shape, output_nonms_1_ptr, output_nonms_2_2.shape, output_nonms_2_2_ptr,output_nonms_2_1.shape, output_nonms_2_1_ptr)
    print(output__hasnms_1.shape, output__hasnms_1_ptr, output__hasnms_2_2.shape, output__hasnms_2_2_ptr, output__hasnms_2_1.shape,  output__hasnms_2_1_ptr)

    # print("*********************",np.sum(output_zero_2_1), np.sum(output_zero1))

    # onum1=1
    # for o in output1[0]:
    #     if o[4] > 0.15:
    #         print(f"当前有{onum1}个数","值为", o[:5], "\n")   
    #         onum1 += 1
    #     # if onum1 > 61:
    #     #     break

    # onum2=1
    # for o in output2_2[0]:
    #     if o[4] > 0.15:
    #         print(f"当前有{onum2}个数","值为", o[:5], "\n")   
    #         onum2 += 1

    onum1=1
    for o in output_nonms_2_2:
        if int(o[6]) == 1 or (int(o[6]) == 0):
            print(f"当前有{onum1}个数","值为", o[:7], "\n")   
            onum1 += 1
        if onum1 > 62:
            break


    # onum2=1
    # for o in output_nonms_2_2:
    #     if int(o[6]) == 1 or (int(o[6]) == 0):
    #         print(f"当前有{onum2}个数","值为", o[:7], "\n")   
    #         onum2 += 1
    #     if onum2 > 65:
    #         break

    # onum3=1
    # for o in output3:
    #     if int(o[6]) == 1 or (int(o[6]) == 0):
    #         print(f"当前有{onum3}个数","值为", o[:7], "\n")   
    #         onum3 += 1
    #     if onum3 > 65:
    #         break

    # onum_nonms1=1
    # for o in output_nonms1:
    #     # print(f"当前有{onum_nonms1}个数","值为", o[:7], "\n")   
    #     # break
    #     if int(o[6]) == 1 or (int(o[6]) == 0):
    #         print(f"当前有{onum_nonms1}个数","值为", o[:7], "\n")   
    #         onum_nonms1 += 1
    #     if onum_nonms1 > 56:
    #         break
    
    # onum_nonms2=1
    # for o in output_nonms3:
    #     # print(f"当前有{onum_nonms1}个数","值为", o[:7], "\n")   
    #     # break
    #     if int(o[6]) == 1 or (int(o[6]) == 0):
    #         print(f"当前有{onum_nonms2}个数","值为", o[:7], "\n")   
    #         onum_nonms2 += 1
    #     if onum_nonms2 > 62:
    #         break
