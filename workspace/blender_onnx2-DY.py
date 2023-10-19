import torch.nn.functional  as F
from torchvision.ops import roi_align 
from onnxsim import simplify
from torch import nn
import math
import onnx
import torch

class Blender(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):    # 1*4*512*512  # N*784   # N*5
        bases, box, feat = x
        _, _, b_h, b_w = bases.shape              
        rois = roi_align(
            bases,   
            box, 
            output_size=(56, 56), 
            spatial_scale=(0.25),
            sampling_ratio=1,
            aligned=True
        )     # N*4*56*56
        pred_mask_logits = merge_bases(rois, feat).sigmoid()
        pred_mask_logits = pred_mask_logits.view(
        -1, 1, 56, 56)
        pred_mask_logits = do_paste_mask(pred_mask_logits, box[:,1:], b_h*4, b_w*4)
        pred_mask_logits = pred_mask_logits.view(-1, b_h*4, b_w*4)
        
        return pred_mask_logits

def merge_bases(rois, coeffs):
    # merge predictions
    N, coeffs_b = map(int, coeffs.size())
    N, B, H, W = map(int,rois.size())
    coeffs = coeffs.view(-1, B, 14, 14)
    coeffs = F.interpolate(coeffs, scale_factor=4,
                            mode='bilinear', align_corners=False).softmax(dim=1)
    masks_preds = (rois * coeffs).sum(dim=1)
    return masks_preds.view(-1, int(H * W))   

def do_paste_mask(masks, boxes, img_h: int, img_w: int):
    device = masks.device
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = int(masks.shape[0])
    # N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)
    
    return img_masks

if __name__ == "__main__":
    DEVICE = "cuda:0"
    image_height = 2048
    image_width = 2048
    assert (image_height % 4 == 0 and image_width % 4 == 0), "图像尺寸需整除4"
    blender_bases = torch.zeros((1, 4, int(image_height/4), int(image_width/4))).to(DEVICE)
    blender_box = torch.zeros((1, 5)).to(DEVICE)
    blender_feat = torch.zeros((1, 784)).to(DEVICE)

    blender_inputs = [blender_bases, blender_box, blender_feat]
    blender = Blender().eval()
    blender_input_name = ["bases", "box", "feat"]
    blender_output_name = ["mask_pred"]
    blender_out = blender(blender_inputs)
    
    blender_output = r"/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/blender-dy.onnx"
    
    torch.onnx.export(
        blender,
        blender_inputs,
        blender_output,
        verbose=False,
        export_params=True,
        input_names=blender_input_name,
        output_names=blender_output_name,
        keep_initializers_as_inputs=True,
        opset_version=16,
        dynamic_axes = {
            "bases":{2:"bases_h", 3:"bases_w"},
            "mask_pred":{1:"mask_pred_h", 2:"mask_pred_w"}
        },        
    )
    
    onnx_model_blender = onnx.load(blender_output)
    model_simp_blender, check = simplify(onnx_model_blender)
    assert check,  "Simplified ONNX model could not be validated"
    onnx.save(model_simp_blender, blender_output)
    print("Done. The onnx model is saved into {}.".format(blender_output))