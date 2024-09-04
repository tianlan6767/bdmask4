import torch
import onnxruntime as ort
import torch.onnx
import cv2
import numpy as np
import os
import json, time
import argparse
import onnx
import logging
import warnings
from torchvision.ops import roi_align
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
from torchvision import transforms
from typing import List
from glob import glob
from tqdm import tqdm
import math
from onnxconverter_common import float16
import torch.nn.functional  as F
# from evalmodel import ModelAnalyzer
import time
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings("ignore")

class inference:
    def __init__(self,model_path:str,is_half=False) -> None:
        self.model_path=model_path
        self.is_half=is_half

        self._load_model() #初始化模型
        self._init_model() #初始化推理会话
        
    def _load_model(self):
        self.model=onnx.load(self.model_path)
        
    def _init_model(self):
        session_option=ort.SessionOptions()
        session_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        provider=["CUDAExecutionProvider","CPUExecutionProvider"]

        if self.is_half:
            fcos_model = self.model
            fcos_model_fp16 = float16.convert_float_to_float16(fcos_model)
            fcos_fp16_path = os.path.join("/".join(self.model_path.split("/")[:-1]), "focs_fc16_model.onnx")
            onnx.save(fcos_model_fp16, fcos_fp16_path)
            
            self.onnx_session=ort.InferenceSession(
                fcos_model_fp16.SerializePartialToString(),
                providers=provider,
                sess_options=session_option
            )

        else:
            self.onnx_session=ort.InferenceSession(
                self.model.SerializePartialToString(),
                providers=provider,
                sess_options=session_option
            )
        
        self.input_name=self.onnx_session.get_inputs()[0].name
         

    def infer(self,input_img:np.ndarray,local:np.ndarray,img:np.ndarray,img_name:str):
        
        start_time = time.time()
        self.base,self.pred = self.onnx_session.run(
            None,{self.input_name:input_img}
            )
        torch.cuda.synchronize()
        self.fcos_spend_time = time.time() - start_time

        jf,imgg=self.run_mask(img,img_name,local)
        print(f"inference time: {self.fcos_spend_time}")

        return jf,imgg
        
    def run_mask(self,img,img_name,local):
        self.mask_time = 0
        nms_boxes, nms_top_feats, nms_class, nms_scores = self.get_mask_input(local)
                
        jf = dict()
        jf[img_name] = {}
        jf[img_name]["filename"] = img_name
        jf[img_name]["regions"] = []
        for i in range(len(nms_boxes)):
            topf = nms_top_feats[i][None, :]
            box = np.insert(nms_boxes[i], 0, values=np.array([0]), axis=0)
            box = box[None, :]
            if len(nms_class.shape)==1:
                class_name=np.argsort(nms_class)[-1]
            elif len(nms_class.shape)==2:
                class_name=np.argsort(nms_class[i])[-1]
            score=math.sqrt(nms_scores[i])          
            
            mask=blender(self.base,box,topf)           
            
            mask_ = np.where(mask[0] > 0.5, 255, 0).astype(np.uint8)
            mask_ = mask_.squeeze()

            contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(0,255,0),2)

            x, y, w, h = cv2.boundingRect(mask_)
            cv2.putText(img, f"{class_name+1} : {round(score,3)}", (int(x+w//2), int(y+h//2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            regions_list = self.create_json(mask, self.idxs[i], self.scores[i])
            for reg in regions_list:
                jf[img_name]["regions"].append(reg)
        jf[img_name]["type"] = "inf"

        return jf,img
         
    def get_mask_input(self,local):
            sq_pred = self.pred.squeeze()
            sq_pred[:,0]=local[:,0]-sq_pred[:,0]
            sq_pred[:,1]=local[:,1]-sq_pred[:,1]
            sq_pred[:,2]=local[:,0]+sq_pred[:,2]
            sq_pred[:,3]=local[:,1]+sq_pred[:,3]
            
            filter_pred = sq_pred[np.where(sq_pred[:, 4] >= 0.16)]
            boxes = filter_pred[:, 0:4]
            scores = filter_pred[:, 4]
            classes = filter_pred[:, 5:30]
            top_feat = filter_pred[:, 30:]
            
            boxes_tensor = torch.from_numpy(boxes)
            scores_tensor = torch.from_numpy(scores)
            
            idxs = torch.from_numpy(np.where(np.isin(classes, scores)==True)[1])
            keep = batched_nms(boxes_tensor, scores_tensor, idxs, 0.1)
            nms_boxes = np.array(boxes_tensor[keep], dtype=np.float16 if self.is_half else np.float32)
            nms_top_feats = top_feat[keep]
            nms_class=classes[keep]
            
            nms_scores=scores[keep]
            if isinstance(nms_scores,np.float32):
                nms_scores=[nms_scores]
        
            if len(nms_top_feats.shape) == 1:
                nms_top_feats = nms_top_feats[None, :]
            self.idxs = idxs[keep]
            self.scores = scores[keep]
            if len(self.scores.shape) == 0:
                self.scores = [self.scores]
                
            return nms_boxes, nms_top_feats,nms_class,nms_scores   
     
    def create_json(self, mask, id, score):
        regions_list = list()
        mask = np.where(mask[0] > 0.5, 255, 0).astype(np.uint8)
        mask = mask.squeeze()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            region = {}
            region["shape_attributes"] = {}
            region["region_attributes"] = {}
            region["region_attributes"]["regions"] = str(int(id)+1)
            region["region_attributes"]["score"] = np.sqrt(score).astype(np.float64)
            region["region_attributes"]["fpn_levels"] = "0"
            sq_cont = np.squeeze(cont)
            if len(sq_cont.shape) == 1:
                continue
            xs, ys = sq_cont[:, 0], sq_cont[:, 1]
            region["shape_attributes"]["all_points_x"] = xs.astype(np.int64).tolist()
            region["shape_attributes"]["all_points_y"] = ys.astype(np.int64).tolist()
            
            
            regions_list.append(region)
            
        return regions_list
    
def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    assert boxes.shape[-1] == 4
    
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes.float(), scores.float(), idxs, iou_threshold)
    
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().reshape(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().reshape(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep        

def merge_bases(rois, coeffs):
    N, B, H, W = map(int,rois.size())
    coeffs = coeffs.reshape(-1, B, 14, 14)

    coeffs = F.interpolate(coeffs, scale_factor=4,
                            mode='bilinear', align_corners=False).softmax(dim=1)
    masks_preds = (rois * coeffs).sum(dim=1)[None,]
    return masks_preds
  
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

def blender(bases,box,feat):
    _, _, b_h, b_w = bases.shape
    bases=torch.tensor(bases)
    box=torch.tensor(box)
    feat=torch.tensor(feat)
    rois=roi_align(
        bases,
        box,
        output_size=(56,56),
        spatial_scale=(0.25),
        sampling_ratio=1,
        aligned=True
    )
    
    pred_mask_logits = merge_bases(rois, feat).sigmoid()
    # pred_mask_logits = pred_mask_logits.reshape(-1, 1, 56, 56)
    pred_mask_logits = do_paste_mask(pred_mask_logits, box[:,1:], int(b_h*4), int(b_w*4))
    pred_mask_logits = pred_mask_logits.reshape(-1, int(b_h*4), int(b_w*4))
    
    return pred_mask_logits

def compute_location(h, w, stride):
    # print(stride, h, w, h*stride, w*stride)
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device="cuda"
    )       # [0,8,16,24,...,2040]
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device="cuda"
    )       # [0,8,16,24,...,2040]
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    # print("shift_x:",shift_x)
    shift_x = shift_x.reshape(-1)
    # print("shift_x1:", shift_x)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def compute_locations(img_h,img_w):
    stride = [8, 16, 32, 64, 128]
    locations = None
    if img_h%32!=0:
        img_h=32*round(img_h/32)
    if img_w%32!=0:
        img_w=32*round(img_w/32)
    for level in stride:
        h, w = round(img_h/level), round(img_w/level)
        locations_per_level = compute_location(
            h, w, level,      # [8, 16, 32, 64, 128]
        )
        if locations is None:
            locations=locations_per_level.cpu().numpy()
        else:
            locations=np.concatenate((locations,locations_per_level.cpu().numpy()),axis=0)
    return locations

def predict(args):
    data_json = dict()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info(args)
    img_files = glob(os.path.join(args.img_path, "*.jpg"))
    
    onnx_C = inference(args.fcos,args.is_half)
    
    for img_file in tqdm(img_files):
        basename = os.path.basename(img_file)
        if len(args.mean) == 1:
            img = cv2.imread(img_file, 0)
            h,w = img.shape
            if h%32!=0:
                img_h=32*round(h/32)
            else:
                img_h=h
            if w%32!=0:
                img_w=32%round(w/32)
            else:
                img_w=w
            img_mask=np.zeros((img_h,img_w))
            img_mask[:h,:w]=img
        else:
            img = cv2.imread(img_file, 1)
            h, w ,c = img.shape
            if h%32!=0:
                img_h=32*round(h/32)
            else:
                img_h=h
            if w%32!=0:
                img_w=32%round(w/32)
            else:
                img_w=w
            img_mask=np.zeros((img_h,img_w,c))
            img_mask[:h,:w,:]=img

        if args.is_half:
            input_img = img_mask.astype(np.float16)
        else:
            input_img = img_mask.astype(np.float32)
        input_img=torch.as_tensor(input_img.astype('float32').transpose(2,0,1)) 
        
        mean=torch.Tensor(args.mean).reshape(3,1,1)
        std=torch.Tensor(args.std).reshape(3,1,1)
        
        input_img=(input_img - mean) / std
        input_img=input_img.numpy()[None,...]
        
        center_locations=compute_locations(h,w)   
        
        jf,imggg=onnx_C.infer(input_img,center_locations,img,basename)
        data_json.update(jf)
            
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path) 
        cv2.imwrite(f"{os.path.join(args.output_path,basename)}",imggg)
        
    json_name = f"onnx_model_{int(time.time())}.json"
        
    with open(f"{args.output_path}/{json_name}", "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False)
        
def args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcos", default=r"/media/ps/data1/train/HJL/test/testmodel/model_1421999.onnx", help="this is the fcos-onnx`s path")
    parser.add_argument("--mean", default=  [43,43,43], help="if image is 3 channel, the length of mean is 3") #[85.12,85.12,85.12]
    parser.add_argument("--std", default= [39,39,39], help="if image is 3 channel, the length of std is 3")#[59.53,59.53,59.53]
    parser.add_argument("--img-path", default=r"/media/ps/data1/train/HJL/test/0814夜", help="the inference image path")
    parser.add_argument("--output-path", default=r"/media/ps/data1/train/HJL/test/0814夜_json", help="result save output path")
    parser.add_argument("--is_half", default=False, help="True is fp16 else fp32")
    
    args = parser.parse_args()
    
    return args 

if __name__=="__main__":
    print(ort.get_device())
    args = args()
    predict(args)
  