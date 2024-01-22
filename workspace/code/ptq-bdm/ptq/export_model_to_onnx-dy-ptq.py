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

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import argparse
import types
import torch
from torch.nn import functional as F
from copy import deepcopy
from onnxsim import simplify
import onnx
import os.path as osp
import os

import quantization.quantize as quantize

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from adet.config import get_cfg
from adet.modeling import FCOS, BlendMask

from detectron2.data import build_detection_test_loader

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.utils.visualizer import GenericMask

from detectron2.data.datasets import register_coco_instances

train_dataset_name = "phone_train"

jf_train = '/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/imgs_jpg-1207/train/train_jpg/train.json'
imgs_train = '/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/imgs_jpg-1207/train/train_jpg'
register_coco_instances(train_dataset_name, {}, jf_train, imgs_train)

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
    merge_box_regression = torch.cat(merge_box_regression, 1)
    merge_top_feat = torch.cat(merge_top_feat, 1) 
    pred = torch.cat([merge_box_regression, merge_logits_pred_max, merge_logits_pred, merge_top_feat], 2)
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


def export_onnx(cfg, args, model : BlendMask, onnx_path):
    
    input_names = ["input_image"]
    input = torch.zeros((1, args.channel, args.height, args.width)).to(cfg.MODEL.DEVICE)
    output_names = ["bases", "pred"]
    
    if isinstance(model, BlendMask):
        patch_blendmask(model)

    if hasattr(model, 'proposal_generator'):
        if isinstance(model.proposal_generator, FCOS):
            patch_fcos(cfg, model.proposal_generator)
            patch_fcos_head(cfg, model.proposal_generator.fcos_head)

    quantize.export_onnx(
        model,
        input,
        onnx_path,
        verbose=False,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=False,
        opset_version=16,
        dynamic_axes = {
            "input_image":{2:"h", 3:"w"},
            "bases":{2: "bases_h", 3:"bases_w"},
            "pred":{1:"pred_nums"}
        } if args.dynamic else None
    )
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check,  "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("Done. The onnx model is saved into {}.".format(onnx_path))

def cmd_quantize(cfg, args, model, save_dir, eval_origin=False, eval_ptq=False, ignore_policy=None, supervision_stride=1, iters=100):
    
    model_name = osp.basename(cfg.MODEL.WEIGHTS).rsplit(".")[0]
    save_ptq = osp.join(save_dir, model_name + "_ptq.pth")

    save_qat = osp.join(save_dir, model_name + "_qat.pth")
    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)
    

    # 量化初始化
    quantize.initialize()
    device  = torch.device(cfg.MODEL.DEVICE)

    #数据集准备
    train_dataloader = build_detection_test_loader(cfg, train_dataset_name)
    # val_dataloader   = build_detection_test_loader(cfg, val_dataset_name)
    
    quantize.replace_bottleneck_forward(model)

    # 自定义量化层，忽略指定量化层
    quantize.replace_to_quantization_module(model, ignore_policy)

    # 标定模型
    quantize.calibrate_model(cfg, model, train_dataloader, device, num_batch=iters)
    
    #导出模型
    export_onnx(cfg, args, model, osp.join(save_dir, f"ptq-{iters}.onnx"))

def setup(args):
    # train.py -----------------------------------------------------------------------------------------
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    config_file = '/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml'
    cfg.merge_from_file(config_file)

    cfg.DATASETS.TRAIN = ("phone_train",)
    cfg.DATASETS.TEST = ("phone_test",)   # no metrics implemented for this dataset
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.09

    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda:2"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 25   # 3 classes (data, fig, hazelnut)
    cfg.MODEL.FCOS.NUM_CLASSES = 25     

    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.FCOS.CENTER_SAMPLE = "center"
        
    if args.channel == 1:
        cfg.INPUT.FORMAT = 'L'
        cfg.MODEL.PIXEL_MEAN = [90]
        cfg.MODEL.PIXEL_STD = [77]
    else:
        cfg.INPUT.FORMAT = 'BGR'
        cfg.MODEL.PIXEL_MEAN = [57.14, 55.92, 56.19]
        cfg.MODEL.PIXEL_STD = [61.46, 61.27, 61.23]
    cfg.MODEL.BASIS_MODULE.LOSS_ON=False 
    cfg.MODEL.BASIS_MODULE.NORM = 'BN'
    cfg.freeze()
    return cfg

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
    parser.add_argument('--dynamic', default=True, action="store_true")
    parser.add_argument('--ptq', default=True, action="store_true")
    
    parser.add_argument(
        "--weights",
        default="/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/JR_1124.pth",
        metavar="FILE",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--output",
        default="/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/model_no",
        metavar="FILE",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = setup(args)

    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)    
    
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    ignore_policy = ['top_layer']
    # 量化模型
    if args.ptq:
        cmd_quantize(cfg, args, model, args.output, ignore_policy=ignore_policy, iters=1)

    #单独导出模型
    else:
        export_onnx(cfg, args, model, osp.join(args.output,"fcos.onnx"))
    
if __name__ == "__main__":
    main()
