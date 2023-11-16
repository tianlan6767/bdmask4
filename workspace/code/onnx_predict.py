import torch
import onnxruntime as rt
import torch.onnx
import cv2
import numpy as np
import os
import json, time
import argparse
import onnx
import logging
import warnings

from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
from torchvision import transforms
from typing import List
from glob import glob
from tqdm import tqdm
from onnxconverter_common import float16

from evalmodel import ModelAnalyzer
import time
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


class ONNX_Predictor:
    def __init__(self, fcos_onnx_path, mask_onnx_path, is_half) -> None:
        """初始化 onnx """
        sess_option = rt.SessionOptions()
        sess_option.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if is_half:
            fcos_model = onnx.load(fcos_onnx_path)
            mask_model = onnx.load(mask_onnx_path)
            fcos_model_fp16 = float16.convert_float_to_float16(fcos_model)
            mask_model_fp16 = float16.convert_float_to_float16(mask_model)
            fcos_fp16_path = os.path.join("/".join(fcos_onnx_path.split("/")[:-1]), "focs_fc16_model.onnx")
            mask_fp16_path = os.path.join("/".join(mask_onnx_path.split("/")[:-1]), "mask_fc16_model.onnx")
            onnx.save(fcos_model_fp16, fcos_fp16_path)
            onnx.save(mask_model_fp16, mask_fp16_path)
            self.fcos_session = rt.InferenceSession(fcos_fp16_path, sess_options=sess_option, providers=["CUDAExecutionProvider"])
            self.mask_session = rt.InferenceSession(mask_fp16_path, sess_options=sess_option, providers=["CUDAExecutionProvider"])
        else:
            self.fcos_session = rt.InferenceSession(fcos_onnx_path, sess_options=sess_option, providers=["CUDAExecutionProvider"])
            self.mask_session = rt.InferenceSession(mask_onnx_path, sess_options=sess_option, providers=["CUDAExecutionProvider"])
        self.input_fcos_name, self.input_mask_name = self.get_input_name()
        self.output_fcos_name, self.output_mask_name = self.get_output_name()
        self.is_half = is_half
        
    def get_input_name(self):
        '''获取输入节点名称'''
        input_fcos_name = []
        input_mask_name = []
        for node in self.fcos_session.get_inputs():
            input_fcos_name.append(node.name)
            
        for node in self.mask_session.get_inputs():
            input_mask_name.append(node.name)
        return input_fcos_name, input_mask_name
    
    def get_output_name(self):
        '''获取输出节点名称'''
        output_fcos_name = []
        output_mask_name = []
        for node in self.fcos_session.get_outputs():
            output_fcos_name.append(node.name)
            
        for node in self.mask_session.get_inputs():
            output_mask_name.append(node.name)
        return output_fcos_name, output_mask_name
    
    def get_input_feed(self, image_tensor):
        ''' 获取输入tensor '''
        input_feed = {}
        for name in self.input_fcos_name:
            input_feed[name] = image_tensor
            
        return input_feed
    
    def get_mask_input(self):
        sq_pred = self.pred.squeeze()
        filter_pred = sq_pred[np.where(sq_pred[:, 4] >= 0.09)]
        boxes = filter_pred[:, 0:4]
        scores = filter_pred[:, 4]
        classes = filter_pred[:, 5:30]
        top_feat = filter_pred[:, 30::]
        
        boxes_tensor = torch.from_numpy(boxes)
        scores_tensor = torch.from_numpy(scores)
        # labels = np.isin(classes, scores)
        # index = np.where(np.isin(classes, scores)==True)
        
        idxs = torch.from_numpy(np.where(np.isin(classes, scores)==True)[1])
        keep = batched_nms(boxes_tensor, scores_tensor, idxs, 0.1)
        nms_boxes = np.array(boxes_tensor[keep], dtype=np.float16 if self.is_half else np.float32)
        nms_top_feats = top_feat[keep]
        if len(nms_top_feats.shape) == 1:
            nms_top_feats = nms_top_feats[np.newaxis, :]
        self.idxs = idxs[keep]
        self.scores = scores[keep]
        if len(self.scores.shape) == 0:
            self.scores = [self.scores]
            
        return nms_boxes, nms_top_feats
    
    def run_fcos(self, input_img, output_names = None):
        '''run FCOS onnx'''
        input_feed = self.get_input_feed(input_img)
        start_time = time.time()
        results = self.fcos_session.run(output_names, input_feed)
        torch.cuda.synchronize()
        self.fcos_spend_time = time.time() - start_time
        self.bases = results[0]
        self.pred = results[1]
    
    def run_mask(self, img_name):
        self.mask_time = 0
        nms_boxes, nms_top_feats = self.get_mask_input()
        jf = dict()
        jf[img_name] = {}
        jf[img_name]["filename"] = img_name
        jf[img_name]["regions"] = []
        for i in range(len(nms_boxes)):
            topf = nms_top_feats[i][np.newaxis, :]
            box = np.insert(nms_boxes[i], 0, values=np.array([0]), axis=0)
            box = box[np.newaxis, :]
            mask_input_feed = {
                "bases": self.bases,
                "box": box,
                "top_feat": topf
            }
            start_time = time.time()
            mask = self.mask_session.run(None, input_feed=mask_input_feed)
            torch.cuda.synchronize()
            spend_time = time.time() - start_time
            self.mask_time += spend_time
            
            regions_list = self.create_json(mask, self.idxs[i], self.scores[i])
            for reg in regions_list:
                jf[img_name]["regions"].append(reg)
        jf[img_name]["type"] = "inf"
        
        return jf      
    
    def create_json(self, mask, id, score):
        regions_list = list()
        mask = np.where(mask[0] > 0.5, 255, 0).astype(np.uint8)
        mask = mask.squeeze()
        mask = np.where(mask > 0.5, mask*0+255, mask*0+0)
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
    
    def run(self, input_img, img_name):
        self.run_fcos(input_img)
        jf = self.run_mask(img_name)
        print(f"inference time: {self.fcos_spend_time + self.mask_time}")
        
        return jf
    
def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    assert boxes.shape[-1] == 4
    
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes.float(), scores.float(), idxs, iou_threshold)
    
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep        

def predict(args):
    data_json = dict()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info(args)
    img_files = glob(os.path.join(args.img_path, "*.jpg"))
    
    onnx_C = ONNX_Predictor(args.fcos, args.mask, args.is_half)
    
    for img_file in tqdm(img_files):
        basename = os.path.basename(img_file)
        if len(args.mean) == 1:
            img = cv2.imread(img_file, 0)
        else:
            img = cv2.imread(img_file, 1)
        
        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        w, h = img.shape
        transform = transforms.Normalize(mean=args.mean, std=args.std)
        if args.is_half:
            input_img = img.reshape(1, len(args.mean), w, h).astype(np.float16)
        else:
            input_img = img.reshape(1, len(args.mean), w, h).astype(np.float32)
        input_img = transform(torch.from_numpy(input_img))
        input_img = np.array(input_img)
        jf = onnx_C.run(input_img, basename)
        data_json.update(jf)
        
    json_name = f"onnx_model_{int(time.time())}.json"
    with open(f"{args.output_path}/{json_name}", "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False)
        
def args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcos", default=r"/media/ps/train/LZC/code/ort_inf/weights/Q4/model-FCOS_2.onnx", help="this is the fcos-onnx`s path")
    parser.add_argument("--mask", default=r"/media/ps/train/LZC/code/ort_inf/weights/Q4/mask_onnx.onnx", help="this is blender-onnx`s path")
    parser.add_argument("--mean", default=[86], help="if image is 3 channel, the length of mean is 3")
    parser.add_argument("--std", default=[76], help="if image is 3 channel, the length of std is 3")
    parser.add_argument("--img-path", default=r"/media/ps/train/LZC/code/ort_inf/img", help="the inference image path")
    parser.add_argument("--output-path", default=r"/media/ps/train/LZC/code/ort_inf/output", help="result save output path")
    parser.add_argument("--is-half", default=False, help="True is fp16 else fp32")
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    print(rt.get_device())
    args = args()
    predict(args)
    
    # # 测试数据图像文件夹
    images_folder = args.img_path

    # 测试数据的标注信息
    imps = glob(images_folder + "/*.jpg")
    # print(len(imps))
    mark_json = glob(images_folder + "/*.json")[0]

    # 模型推理结果文件夹
    inf_folder = args.output_path
    # # 构造模型分析器
    MA = ModelAnalyzer.load(images_folder,mark_json, inf_folder)
    # # 模型评估并推荐
    MA.filter_inf_regions(min_score=0.4, min_area=10)
    good_models = MA.recommend_model(top_k=15, min_iou=0.1, min_chk_rate=0.0, over_factor=0)
    # # 保存模型的性能指标
    MA.to_excel()
    # # 输出模型的评估报告
    # good_models = ["model_0239999"]
    # print(good_models)
    # MA.gen_report(good_models, reporter='Tom')
    # MA.get_defect(good_models, 'cmp')
    # MA.get_custom_defect(good_models, min_iou = 0.0001, max_score = 0.5)  
    MA.get_image(good_models, 'miss', True)
    # MA.get_defect(good_models, "check")
    MA.get_image(good_models, "check", True)
    MA.get_image(good_models, "over", True)
    # MA.get_defect(good_models, "miss")

    # MA.get_image(good_models, "abs-over", save_format=".jpg", compared=True)
    # MA.get_update_mark(good_models)

    # MA.get_image(good_models, "abs-over", True, ".jpg", crop_size=(2048, 2048), crop_save=False)
    # MA.get_image(good_models, "abs-over", True, ".jpg")