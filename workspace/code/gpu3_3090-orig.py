from adet.config import get_cfg
from detectron2.engine import DefaultPredictor
import time
import torch
import cv2
import glob, tqdm
import os.path as osp
import numpy as np
from time import sleep
import sys
import numpy as np

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import time
import json


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(1)

jf = r'/media/ps/data/train/LQ/project/OQC/train/0000/go/annotations/train.json'
imgs = r''
register_coco_instances("phone", {}, jf, imgs)
fruits_nuts_metadata = MetadataCatalog.get("phone")
dataset_dicts = DatasetCatalog.get("phone")

n = 0


predictor = None

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

class DefaultPredictor1(DefaultPredictor):
    @torch.no_grad()
    def __call__(self, original_images):
        
        inputs_lst = []
        original_images = original_images
        # print("original_images", original_images)
        
        for original_image_tmp in original_images:
            original_image = original_image_tmp
            imn = original_image_tmp[1]
            height, width = original_image.shape[:2]
            print("图像尺寸",original_image.shape)
            image = original_image
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            if self.input_format == "L":
                image = torch.as_tensor(image.astype("float32"))
                image = np.squeeze(image)
            else:
                if len(original_image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    image = image
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = {"image": image, "height": height, "width": width, "image_name":imn}
            inputs_lst.append(inputs)
        # torch.cuda.synchronize()
        # print("推理*********",image.shape)
        start1 = time.time()
        predictions = self.model(inputs_lst)
        # torch.cuda.synchronize()
        end1 = time.time()
        # print("gpu0 pure inf time:               {}".format(round(end1 - start1, 3)))
        if predictions[0]["instances"].pred_boxes.tensor.shape[0] > 0:
            print("当前图片名{}---{}个缺陷, 当前推理耗时:{}ms".format(imn_orig,predictions[0]["instances"].pred_boxes.tensor.shape[0], round(end1 - start1, 3)*1000))
        
        return predictions, round(end1 - start1, 3)

def init(key, iv):
    cfg = get_cfg()
    config_file = r'/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml'
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TRAIN = ("phone",)
    cfg.DATASETS.TEST = ("phone",)   # no metrics implemented for this dataset

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 25
    cfg.MODEL.FCOS.NUM_CLASSES = 25


    img_size = 2048
    cfg.INPUT.MAX_SIZE_TRAIN = img_size
    cfg.INPUT.MIN_SIZE_TRAIN = img_size
    cfg.INPUT.MAX_SIZE_TEST = img_size
    cfg.INPUT.MIN_SIZE_TEST = img_size
    cfg.MODEL.WEIGHTS = r"/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/JR_1124.pth"
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.09
    cfg.MODEL.DEVICE = "cuda:3"
    cfg.MODEL.KEY = key
    cfg.MODEL.IV =  iv
    
    # cfg.MODEL.FPN.OUT_CHANNELS = 64
    # cfg.MODEL.FCOS.NUM_CLS_CONVS = 0
    # cfg.MODEL.FCOS.NUM_BOX_CONVS = 0

    # cfg.MODEL.FCOS.NUM_CLS_CONVS_p2 = 0
    # cfg.MODEL.FCOS.NUM_BOX_CONVS_p2 = 0
    # cfg.MODEL.FCOS.NUM_SHARE_CONVS =2

    # cfg.MODEL.FCOS.TOP_LEVELS = 2            # 0:不使用p6p7, 1:使用p6, 2:使用p7
    # cfg.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]      
    # cfg.MODEL.BASIS_MODULE.COMMON_STRIDE = 8


    # cfg.MODEL.FCOS.IN_FEATURES = ["p2", "p3","p4", "p5", "p6", "p7"]  # p2------p7
    # cfg.MODEL.FCOS.FPN_STRIDES = [4, 8, 16, 32, 64, 128]              # p2: 4, p3:8, p4：16, p5:32, p6:64, p7:128
    # cfg.MODEL.FCOS.SIZES_OF_INTEREST = [32, 64, 128, 256, 512]        # p2:(0, 32), p3:(32, 64), p4:(64, 128), p5:(128, 256), p6:(256, 512), p7:(512, 正无穷)
    # cfg.MODEL.RESNETS.OUT_FEATURES = ["res2","res3","res4", "res5"]   # 未使用p2, 可以删除"res2"
    # cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4", "res5"]        # 未使用p2, 可以删除"res2"
    
    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    # cfg.MODEL.TEST_FLOAT16 = True
    
    cfg.MODEL.FCOS.CENTER_SAMPLE = "center"
    cfg.MODEL.BASIS_MODULE.LOSS_ON=False

    # cfg.INPUT.FORMAT = 'L'
    # cfg.MODEL.PIXEL_MEAN = [90]
    # cfg.MODEL.PIXEL_STD = [77]

    cfg.INPUT.FORMAT = 'BGR'
    cfg.MODEL.PIXEL_MEAN = [57.14, 55.92, 56.19]
    cfg.MODEL.PIXEL_STD = [61.46, 61.27, 61.23]


    # cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [True, True, True, True]
    # cfg.MODEL.FCOS.USE_DEFORMABLE = True
    
    global predictor
    predictor = DefaultPredictor1(cfg)


def predict(*image):
    # print("=========================", len(image))
    global imn_orig
    scores = []
    classes = []
    masks = []
    boxs = []
    original_images = []

    for i, img in enumerate(image):
        original_images.append(img)
        # cv2.imwrite(r'C:\Users\ps\Desktop\tmp\{}.jpg'.format(i), img)
    st1 = time.time()
    outputs, inf_time = predictor(original_images)
    st2 = time.time()

    et = round(st2 - st1, 3)
    # print("dsfd", et)
    # print(original_images)
    sub_save_dir = r"/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/pth-inf1124-nonpz"
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir, exist_ok=True)
    if len(original_images[0].shape) == 2:
            original_images[0] = cv2.cvtColor(original_images[0], cv2.COLOR_GRAY2BGR)
    
    # print(original_images[0][:, :, ::-1].shape)
    v = Visualizer(original_images[0][:, :, ::-1],
                               metadata=fruits_nuts_metadata, 
                               scale=1, 
                               instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                )
    # print(outputs)
    
    v = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
    # plt.imshow(v.get_image()[:, :, ::-1])

    cv2.imwrite(osp.join(sub_save_dir, imn_orig.replace(".bmp", ".jpg")), v.get_image()[:, :, ::-1])

    
    for output in outputs:
        predictions = output["instances"].to("cpu")
        scores.append(predictions.scores.tolist())
        classes.append(predictions.pred_classes.tolist())
        # masks.append(np.asarray(predictions.pred_masks))
        # print(predictions.pred_boxes.tensor.tolist())
        boxs.append(np.asarray(predictions.pred_boxes.tensor.tolist()))
        pred_masks_gpu = output["instances"].pred_masks
        masks_lst = []
        for pred_mask in pred_masks_gpu:
            mask = [m.cpu().numpy() for m in torch.where(pred_mask)]
            masks_lst.append(np.array(mask))
        masks.append(masks_lst)
    # if len(scores):
    #     for score, classid, box in zip(scores, classes, boxs):
    #         for s, cid, b in zip(score, classid, box):
    #             print(s,"**", cid)
    #             # pass
    if len(scores):
        print("保存的图片名称",imn_orig)
        for score, cid, box in zip(scores[0], classes[0], boxs[0]):
            print(round(score,3), cid, box)
    
    return [scores, classes, masks], et


#读json
def get_jf(jf):
    with open(jf, "r") as f:
        json_data = json.load(f)
    return json_data

def get_large_box(reg, h, w, offset= 100):
    x_min, y_min, x_max, y_max = xsys2box(reg)

    dis = (x_max - x_min) - (y_max - y_min)
    if dis > 0:
        y_min = y_min - int(dis / 2)
        y_max = y_max + int(dis / 2)
    else:
        x_min = x_min + int(dis / 2)
        x_max = x_max - int(dis / 2)

    np_box = np.array([x_min-offset, y_min-offset, x_max+offset, y_max+offset])
    np_box[[0,2]] = np.clip(np_box[[0,2]], 0, w)
    np_box[[1,3]] = np.clip(np_box[[1,3]], 0, h)
    return np_box

def xsys(reg):
    xs = reg["shape_attributes"]["all_points_x"]
    ys = reg["shape_attributes"]["all_points_y"]
    return xs, ys

def get_region(xs, ys):
    reg = np.dstack((xs, ys)).astype(int)
    return reg

def xsys2mask(reg):
    xs, ys = xsys(reg)
    return get_region(xs, ys)

def xsys2box(reg):
    xs, ys = xsys(reg)
    xmax = max(xs)
    ymax = max(ys)
    xmin = min(xs)
    ymin = min(ys)
    return [xmin, ymin, xmax, ymax]

if __name__ == "__main__":
    global imn_orig
    np.set_printoptions(suppress=True)
    init("8Xe0efbbhSPHmaw0", "OwXaWuIhMzErsKl5")
    src = r"/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/imgs"

    # dst = r"/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JT/pinf5"
    # if not os.path.exists(dst):
    #     os.makedirs(dst, exist_ok=True)
    
    imps = glob.glob(src + "/*.[bj][mp][pg]")
    # jf = r"/media/ps/data/train/LQ/project/OQC/train/0000/go/train/data_merge.json"
    # jd = get_jf(jf)

    # im = []
    times = []
    # origs = []
    imgs = []
    # group = 1
    # fns = []
    # for imp in tqdm.tqdm(imps):
    #     fn = osp.basename(imp)
    #     img = cv2.imread(imp, -1)
    #     h, w = img.shape[:2]
    #     reg_id = 0
    #     # print(jd[fn]["regions"])
    #     imgs.append(crop_img)
    #     fns.append(fn[:-4] + "_" + str(reg_id) + ".jpg")
            
    #     for reg in jd[fn]["regions"]:
    #         box = get_large_box(reg, h, w, offset= 5)
    #         # m = xsys2mask(reg)
    #         # m[:, :, 0] = m[:, :, 0] - box[0]
    #         # m[:, :, 1] = m[:, :, 1] - box[1]
            
    #         crop_img = img[box[1]:box[3], box[0]:box[2]]
    #         # crop_img = cv2.polylines(crop_img, [m], True, 255, 1)
    #         imgs.append(crop_img)
    #         fns.append(fn[:-4] + "_" + str(reg_id) + ".jpg")
    #         reg_id += 1
    #     # print(img.shape)
        
    # for fn, img in tqdm.tqdm(zip(fns, imgs)):
    #     cv2.imwrite(osp.join(dst, fn), img)
    # for idx in range(0, len(imgs), group):
    #     origs.append(imgs[idx: idx + group])
    # imgs_origs = []   
    # for idx in range(0, len(fns), group):
    #     imgs_origs.append(fns[idx: idx + group])
    # print(origs)
    # for orig, imns_orig in tqdm.tqdm(zip(origs, imgs_origs)):
    #     imn_orig = imns_orig[0]
    #     print("推理的图片",imns_orig)
    #     result = predict(orig + imns_orig)
    #     times.append(result[1])
    # print("共{}张图片，平均{:.3f}".format(len(times[1:]), sum(times[1:])/ len(times[1:])))
    
    # img1 = cv2.imread(imps[0], 0)
    # img2 = cv2.imread(imps[1], 0)
    # result = predict(*[img1, img2])
    
    for imp in tqdm.tqdm(imps):

        imn = osp.basename(imp)
        imn_orig = imn
        img = cv2.imread(imp, 1)
        result = predict(*[img])
        times.append(result[1])
        # im.append(img)
    # print(imgs_part_times)
    print("共{}张图片，平均{:.3f}".format(len(times[1:]), sum(times[1:])/ len(times[1:])))
    
    # all_times = []
    # for values in imgs_part_times.values():
    #     all_times += values
    # all_times = np.asarray(all_times).reshape(-1, len(list(imgs_part_times.values())[0]))
    # restime = np.sum(all_times[1:], axis=0)/(all_times.shape[0] - 1)
    # [round(fpn_time, 2), round(basis_module_time, 2), round(proposal_generator_time, 2), round(blender_time, 2)]
    # print(restime)
    



    