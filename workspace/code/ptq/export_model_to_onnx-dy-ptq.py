"""
A working example to export the R-50 based FCOS model:
python onnx/export_model_to_onnx.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --output /data/pretrained/onnx/fcos/FCOS_R_50_1x_bn_head.onnx
    --opts MODEL.WEIGHTS /data/pretrained/pytorch/fcos/FCOS_R_50_1x_bn_head.pth MODEL.FCOS.NORM "BN"

python export_model_to_onnx.py \
    --config-file /home/ps/inflibs171/AdelaiDet/configs/BlendMask/R_50_3x.yaml \
    --output /media/ps/train/train_root/K11_new/train/1028/weights/1/blendmask11_level2_ep2.onnx \
    --opts MODEL.WEIGHTS /media/ps/train/train_root/K11_new/train/1028/weights/1/model_0651999.pth MODEL.FCOS.NORM "GN"
# about the upsample/interpolate
https://github.com/pytorch/pytorch/issues/10446
https://github.com/pytorch/pytorch/issues/18113
"""

import argparse
import types
import torch
from torch.nn import functional as F
from copy import deepcopy
from torchvision.ops import roi_align 
from onnxsim import simplify
import onnx
import math
import os.path as osp
import os
import cv2
from glob import glob
import imagesize
import numpy as np
import json

# multiple versions of Adet/FCOS are installed, remove the conflict ones from the path

import sys
    
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from adet.config import get_cfg
from adet.modeling import FCOS, BlendMask

import quantization.quantize as quantize


class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def patch_blendmask(model):
    def forward(self, tensor):
        images = None
        gt_instances = None
        basis_sem = None

        features = self.backbone(tensor)
        basis_out, _ = self.basis_module(features, basis_sem)
        proposals  = self.proposal_generator(images, features, gt_instances, self.top_layer)
        return basis_out["bases"][0], proposals

    model.forward = types.MethodType(forward, model)





def patch_fcos(cfg, proposal_generator):
    def proposal_generator_forward(self, images, features, gt_instances=None, top_module=None):
        features = [features[f] for f in self.in_features]
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(features, top_module, self.yield_proposal)
        results = predict_proposals(cfg, logits_pred, reg_pred, ctrness_pred, top_feats)
        return results

    proposal_generator.forward = types.MethodType(proposal_generator_forward, proposal_generator)


def predict_proposals(cfg, logits_pred, reg_pred, ctrness_pred, top_feats=None):      
    strides = cfg.MODEL.FCOS.FPN_STRIDES
    bundle = {
        "o": logits_pred,
        "r": reg_pred, "c": ctrness_pred,
        "s":strides
    }
    bundle['t'] = top_feats
    merge_logits_pred = []
    merge_box_regression = []
    merge_top_feat = []
    merge_ctrness_pred = []
    for i, per_bundle in enumerate(zip(*bundle.values())):
        per_bundle = dict(zip(bundle.keys(), per_bundle))
        o = per_bundle["o"]
        r = per_bundle["r"] * per_bundle["s"]
        c = per_bundle["c"]
        t = per_bundle["t"] if "t" in bundle else None
        out_logits_pred, out_box_regression, out_top_feat, out_ctrness_pred = forward_for_single_feature_map(cfg, o, r, c, t)
        merge_logits_pred.append(out_logits_pred)
        merge_box_regression.append(out_box_regression)
        merge_top_feat.append(out_top_feat)
        merge_ctrness_pred.append(out_ctrness_pred)
    merge_logits_pred = torch.cat(merge_logits_pred, 1)
    merge_logits_pred_max = torch.max(merge_logits_pred, 2)[0][:,:,None]
    merge_box_regression = torch.cat(merge_box_regression, 1)[0]
    merge_top_feat = torch.cat(merge_top_feat, 1) 
    pred = torch.cat([merge_box_regression[None], merge_logits_pred_max, merge_logits_pred, merge_top_feat], 2)
    return pred

    

def forward_for_single_feature_map(cfg, logits_pred, reg_pred,ctrness_pred, top_feat=None):
    # N, C, H, W = list(map(int, logits_pred.shape))
    N, C, H, W = logits_pred.shape

    # put in the same format as locations
    logits_pred = logits_pred.view(-1, C, H, W).permute(0, 2, 3, 1)      # (1,25,256,256) => (1,256,256,25)
    logits_pred = logits_pred.reshape(-1, H*W, C).sigmoid()               # (1,256,256,25) => (1,65536,25)
    box_regression = reg_pred.view(-1, 4, H, W).permute(0, 2, 3, 1)      # (1,4,256,256) => (1,256,256,4)
    box_regression = box_regression.reshape(-1, H*W, 4)                   # (1,256,256,4) => (1,65536,4)
    ctrness_pred = ctrness_pred.view(-1, 1, H, W).permute(0, 2, 3, 1)    # (1,1,256,256) => (1,256,256,1)
    ctrness_pred = ctrness_pred.reshape(-1, H*W).sigmoid()                # (1,256,256,1) => (1,65536)
    # if top_feat is not None:
    top_feat = top_feat.view(-1, 784, H, W).permute(0, 2, 3, 1)       # (1,784,256,256) => (1,256,256,784)
    top_feat = top_feat.reshape(-1, H * W, 784)                       # (1,256,256,784) => (1,65536)
    logits_pred = logits_pred * ctrness_pred[:, :, None]
    return logits_pred, box_regression, top_feat, ctrness_pred

def patch_fcos_head(cfg, fcos_head):
    # step 1. config
    norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
    head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                            cfg.MODEL.FCOS.USE_DEFORMABLE),
                    "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                             cfg.MODEL.FCOS.USE_DEFORMABLE),
                    "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                              False)}

    # step 2. seperate module
    for l in range(fcos_head.num_levels):
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 0]))
                if norm in ["GN", "NaiveGN"]:
                    tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 1]))
                elif norm in ["BN", "SyncBN"]:
                    tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 1][l]))
                tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 2]))
            fcos_head.add_module('{}_tower{}'.format(head, l), torch.nn.Sequential(*tower))

    # step 3. override fcos_head forward
    def fcos_head_forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = getattr(self, 'cls_tower{}'.format(l))(feature)
            bbox_tower = getattr(self, 'bbox_tower{}'.format(l))(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))

        return logits, bbox_reg, ctrness, top_feats, bbox_towers

    fcos_head.forward = types.MethodType(fcos_head_forward, fcos_head)


def create_train_dataloader(cfg, dataset, batch_size=1, rank=-1, world_size=1, workers=8):
    imps = glob(dataset + "/*.jpg")
    jf = glob(dataset + "/*.json")[0]
    if jf:
        pass
        
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    train_mapper = []
    for imp in imps:
        image = cv2.imread(imp, -1)
        if cfg.INPUT.FORMAT == "L":
            image = torch.as_tensor(image.astype("float32"))
            image = np.squeeze(image)
        else:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
        
        w, h = imagesize.get(imp)
        train_mapper.append(
            {   
                "image":image,
                "filename":imp,
                "height":int(h),
                "width":int(w),
            }
        )
    return torch.utils.data.DataLoader(train_mapper,
                                       batch_size=batch_size,
                                       num_workers=nw,
                                       sampler=sampler,
                                       pin_memory=True)
        
        
        
def cmd_quantize(cfg,args, model, dataset, save_dir, eval_origin=False, eval_ptq=False, ignore_policy=None, supervision_stride=1, iters=200):
    
    model_name = osp.basename(cfg.MODEL.WEIGHTS).rsplit(".")[0]
    save_ptq = osp.join(save_dir, model_name + "_ptq.pth")

    save_qat = osp.join(save_dir, model_name + "_qat.pth")
    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)
    
    quantize.initialize()
    device  = torch.device(cfg.MODEL.DEVICE)
    train_dataloader = create_train_dataloader(cfg, dataset)
    # val_dataloader   = create_coco_val_dataloader(dataset)
    quantize.replace_to_quantization_module(model)
    # quantize.apply_custom_rules_to_quantizer(cfg, args, model, export_onnx)
    quantize.calibrate_model(cfg, model, train_dataloader, device)

    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    # summary = SummaryTool(summary_file)

    # if eval_origin:
    #     print("Evaluate Origin...")
    #     with quantize.disable_quantization(model):
    #         ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
    #         summary.append(["Origin", ap])

    # if eval_ptq:
    #     print("Evaluate PTQ...")
    #     ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
    #     summary.append(["PTQ", ap])

    if save_ptq:
        print(f"Save ptq model to {save_ptq}")
        # torch.save(model.state_dict(), save_ptq)
        torch.save(model, save_ptq)

    if save_qat is None:
        print("Done as save_qat is None.")
        return

    # best_ap = 0
    # def per_epoch(model, epoch, lr):

    #     nonlocal best_ap
    #     ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
    #     summary.append([f"QAT{epoch}", ap])

    #     if ap > best_ap:
    #         print(f"Save qat model to {save_qat} @ {ap:.5f}")
    #         best_ap = ap
    #         torch.save({"model": model}, save_qat)

    # def preprocess(datas):
    #     return datas[0].to(device).float() / 255.0

    # def supervision_policy():
    #     supervision_list = []
    #     for item in model.model:
    #         supervision_list.append(id(item))

    #     keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
    #     keep_idx.append(len(model.model) - 2)
    #     def impl(name, module):
    #         if id(module) not in supervision_list: return False
    #         idx = supervision_list.index(id(module))
    #         if idx in keep_idx:
    #             print(f"Supervision: {name} will compute loss with origin model during QAT training")
    #         else:
    #             print(f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
    #         return idx in keep_idx
    #     return impl

    # quantize.finetune(
    #     model, train_dataloader, per_epoch, early_exit_batchs_per_epoch=iters, 
    #     preprocess=preprocess, supervision_policy=supervision_policy())



def export_onnx(cfg, args, model : BlendMask, onnx_path, dynamic=False):
    
    input_names = ["input_image"]
    input = torch.zeros((1, args.channel, args.height, args.width)).to(cfg.MODEL.DEVICE)
    output_names = ["bases", "pred"]

    quantize.export_onnx(
        model,
        input,
        onnx_path,
        verbose=False,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=False,
        opset_version=13,
        dynamic_axes = {
            "input_image":{2:"h", 3:"w"},
            "bases":{2: "bases_h", 3:"bases_w"},
            "pred":{1:"pred_nums"}
        } if dynamic else None
    )
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check,  "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("Done. The onnx model is saved into {}.".format(onnx_path))


def cmd_export(cfg, args, model):
    model_name = osp.basename(cfg.MODEL.WEIGHTS)
    onnx_path = osp.join(args.output, model_name.replace(".pth", ".onnx"))
    
    quantize.initialize()
    # model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS, map_location="cpu"))
    model = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    model.to(cfg.MODEL.DEVICE)
    export_onnx(cfg, args, model, onnx_path, args.dynamic)
    # quantize.export_onnx(model, input, onnx_path, opset_version=11, 
    #     input_names=input_names, output_names=output_names, export_params=True,
    #     dynamic_axes = {
    #         "input_image":{2:"h", 3:"w"},
    #         "bases":{2: "bases_h", 3:"bases_w"},
    #         "pred":{1:"pred_nums"}
    #     } if dynamic else None
    # )
    # print(f"Save onnx to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model to the onnx format")
    parser.add_argument(
        "--config-file",
        default="/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--width', default=2048, type=int)
    parser.add_argument('--height', default=2048, type=int)
    parser.add_argument('--channel', default=3, type=int)
    parser.add_argument('--dynamic', default=False, type=bool)
    parser.add_argument('--ptq', default=True, type=bool)
    parser.add_argument(
        "--weights",
        default="/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826.pth",
        # default="/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826_ptq.pth",
        metavar="FILE",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--output",
        default="/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT",
        metavar="path",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    # train.py -----------------------------------------------------------------------------------------
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    config_file = '/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml'
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda:1"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 25  # 3 classes (data, fig, hazelnut)
    cfg.MODEL.FCOS.NUM_CLASSES = 25
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    cfg.INPUT.FORMAT = 'BGR'
    cfg.MODEL.PIXEL_MEAN = [41,41,41]
    cfg.MODEL.PIXEL_STD = [34,34,34]


    cfg.MODEL.BASIS_MODULE.NORM = 'BN'
    cfg.freeze()
    # -------------------------------------------------------------------------------------------------------------------------
    model = build_model(cfg)


    # model.eval()
    model.to(cfg.MODEL.DEVICE)
    if args.ptq:
        checkpointer = DetectionCheckpointer(model)
        _ = checkpointer.load(cfg.MODEL.WEIGHTS)    

    if isinstance(model, BlendMask):
        patch_blendmask(model)

    if hasattr(model, 'proposal_generator'):
        if isinstance(model.proposal_generator, FCOS):
            patch_fcos(cfg, model.proposal_generator)
            patch_fcos_head(cfg, model.proposal_generator.fcos_head)
            
    if not osp.exists(osp.dirname(args.output)):
        os.makedirs(osp.dirname(args.output), exist_ok=True)
    

    train_dataset = r"/media/ps/data/train/LQ/project/OQC/train/0923/add-ngs"
    # val_dataset = r"/media/ps/data/train/LQ/project/OQC/train/0922/add/ngs2"
    

    # cmd_quantize(cfg, args, model, train_dataset, args.output)
    
    export_onnx(cfg, args, model, osp.join(args.output, "static.onnx"))
    
    # cmd_export(cfg, args, model)
    
if __name__ == "__main__":
    main()
