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


h,w
./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy.onnx \
          --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy \
          --minShapes=input_image:1x1x1024x1024,bases:1x4x256x256 \
          --optShapes=input_image:1x1x2048x2048,bases:1x4x512x512 \
          --maxShapes=input_image:1x1x4096x5472,bases:1x4x1024x1368 \
          --fp16 \
          --workspace=20480 \
          --preview=+fasterDynamicShapes0805

./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy666.onnx \
          --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy-4096 \
          --minShapes=input_image:1x1x2048x2048 \
          --optShapes=input_image:1x1x2048x2048  \
          --maxShapes=input_image:1x1x4096x5472  \
          --fp16 \
          --device=2 \
          --workspace=10240

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
# multiple versions of Adet/FCOS are installed, remove the conflict ones from the path
try:
    from remove_python_path import remove_path
    remove_path()
except:
    import sys
    print(sys.path)
    
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from adet.config import get_cfg
from adet.modeling import FCOS, BlendMask


def patch_blendmask(cfg, model, output_names):
    def forward(self, tensor):
        images = None
        gt_instances = None
        basis_sem = None

        features = self.backbone(tensor)
        basis_out, basis_losses = self.basis_module(features, basis_sem)
        proposals  = self.proposal_generator(images, features, gt_instances, self.top_layer)
        return basis_out["bases"][0], proposals
        # return proposals

    model.forward = types.MethodType(forward, model)
    # output
    output_names.extend(["bases"])
    output_names.extend(["pred"])
    

    # for item in ["pred", "mask_logits"]:
    #     output_names.extend([item])
    
def patch_fcos(cfg, proposal_generator):
    def proposal_generator_forward(self, images, features, gt_instances=None, top_module=None):
        features = [features[f] for f in self.in_features]
        logits_pred, reg_pred, ctrness_pred = self.fcos_head(features)
        preds = predict_proposals(cfg, logits_pred, reg_pred, ctrness_pred)
        merge_features = []
        for f in features:
            N, C, H, W = f.shape
            f = f.view(-1, 256, H, W).permute(0, 2, 3, 1)      # (1,4,256,256) => (1,256,256,4)
            f = f.reshape(-1, H*W, 256)                   # (1,256,256,4) => (1,65536,4)
            merge_features.append(f)
        merge_features = torch.cat(merge_features, 1)
        results = torch.cat([preds, merge_features], 2)
        return results

    proposal_generator.forward = types.MethodType(proposal_generator_forward, proposal_generator)


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full_like(
        box_tensor[:, :1], batch_index, dtype=box_tensor.dtype, device=box_tensor.device
    )
    return torch.cat((repeated_index, box_tensor), dim=1)


def predict_proposals(cfg, logits_pred, reg_pred, ctrness_pred):      
    strides = cfg.MODEL.FCOS.FPN_STRIDES
    bundle = {
        "o": logits_pred,
        "r": reg_pred, "c": ctrness_pred,
        "s":strides
    }
    merge_logits_pred = []
    merge_box_regression = []
    merge_ctrness_pred = []
    for i, per_bundle in enumerate(zip(*bundle.values())):
        per_bundle = dict(zip(bundle.keys(), per_bundle))
        o = per_bundle["o"]
        r = per_bundle["r"] * per_bundle["s"]
        c = per_bundle["c"]
        out_logits_pred, out_box_regression, out_ctrness_pred = forward_for_single_feature_map(o, r, c)
        merge_logits_pred.append(out_logits_pred)
        merge_box_regression.append(out_box_regression)
        merge_ctrness_pred.append(out_ctrness_pred)
    merge_logits_pred = torch.cat(merge_logits_pred, 1)
    merge_logits_pred_max = torch.max(merge_logits_pred, 2)[0][:,:,None]
    merge_box_regression = torch.cat(merge_box_regression, 1)[0]
    pred = torch.cat([merge_box_regression[None], merge_logits_pred_max, merge_logits_pred], 2)
    return pred

    

def forward_for_single_feature_map(logits_pred, reg_pred, ctrness_pred, top_feat=None):
    # N, C, H, W = list(map(int, logits_pred.shape))
    N, C, H, W = logits_pred.shape

    # put in the same format as locations
    logits_pred = logits_pred.view(-1, C, H, W).permute(0, 2, 3, 1)      # (1,25,256,256) => (1,256,256,25)
    logits_pred = logits_pred.reshape(-1, H*W, C).sigmoid()               # (1,256,256,25) => (1,65536,25)
    box_regression = reg_pred.view(-1, 4, H, W).permute(0, 2, 3, 1)      # (1,4,256,256) => (1,256,256,4)
    box_regression = box_regression.reshape(-1, H*W, 4)                   # (1,256,256,4) => (1,65536,4)
    ctrness_pred = ctrness_pred.view(-1, 1, H, W).permute(0, 2, 3, 1)    # (1,1,256,256) => (1,256,256,1)
    ctrness_pred = ctrness_pred.reshape(-1, H*W).sigmoid()                # (1,256,256,1) => (1,65536)
    if top_feat is not None:
        top_feat = top_feat.view(-1, 784, H, W).permute(0, 2, 3, 1)       # (1,784,256,256) => (1,256,256,784)
        top_feat = top_feat.reshape(-1, H * W, 784)                       # (1,256,256,784) => (1,65536)
    logits_pred = logits_pred * ctrness_pred[:, :, None]
    return logits_pred, box_regression,ctrness_pred

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
    # for l in range(fcos_head.num_levels):
    for head in head_configs:
        tower = []
        num_convs, use_deformable = head_configs[head]
        for i in range(num_convs):
            tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 0]))
            tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 1]))
            tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 2]))
        fcos_head.add_module('{}_tower'.format(head), torch.nn.Sequential(*tower))

    # step 3. override fcos_head forward
    def fcos_head_forward(self, x):
        logits = []
        bbox_reg = []
        ctrness = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower =self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            # if top_module is not None:
            #     top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, ctrness

    fcos_head.forward = types.MethodType(fcos_head_forward, fcos_head)

def main():
    parser = argparse.ArgumentParser(description="Export model to the onnx format")
    parser.add_argument(
        "--config-file",
        default="/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--width', default=5472, type=int)
    parser.add_argument('--height', default=4096, type=int)
    parser.add_argument('--channel', default=3, type=int)
    
    parser.add_argument(
        "--weights",
        default="/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826.pth",
        metavar="FILE",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--output",
        default="/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826-dd.onnx",
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
    # train.py -----------------------------------------------------------------------------------------
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    config_file = '/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml'
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda:0"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 25  # 3 classes (data, fig, hazelnut)
    cfg.MODEL.FCOS.NUM_CLASSES = 25

    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
        
    # cfg.INPUT.FORMAT = 'L'
    # cfg.MODEL.PIXEL_MEAN = [59.406]
    # cfg.MODEL.PIXEL_STD = [59.32]
    
    cfg.INPUT.FORMAT = 'BGR'
    cfg.MODEL.PIXEL_MEAN = [41,41,41]
    cfg.MODEL.PIXEL_STD = [34,34,34]


    cfg.MODEL.BASIS_MODULE.NORM = 'BN'
    cfg.freeze()
    # -------------------------------------------------------------------------------------------------------------------------
    model = build_model(cfg)


    model.eval()
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)

    height, width = 2048, 2048
    if args.width > 0:
        width = args.width
    if args.height > 0:
        height = args.height
    input_names = ["input_image"]
    
    # 加载图像
    # im = cv2.imread(r"/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/imgs/2048.jpg", 0)
    
    dummy_input = torch.zeros((1, args.channel, height, width)).to(cfg.MODEL.DEVICE)

    output_names = []
    if isinstance(model, BlendMask):
        patch_blendmask(cfg, model, output_names)

    if hasattr(model, 'proposal_generator'):
        if isinstance(model.proposal_generator, FCOS):
            patch_fcos(cfg, model.proposal_generator)
            patch_fcos_head(cfg, model.proposal_generator.fcos_head)
            
    if not osp.exists(osp.dirname(args.output)):
        os.makedirs(osp.dirname(args.output), exist_ok=True)
        

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        verbose=False,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=False,
        opset_version=11,
        # dynamic_axes = {
        #     "input_image":{2:"h", 3:"w"},
        #     "bases":{2: "bases_h", 3:"bases_w"},
        #     "pred":{1:"pred_nums"}
        # }
    )
    

    onnx_model = onnx.load(args.output)
    model_simp, check = simplify(onnx_model)
    assert check,  "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output)
    print("Done. The onnx model is saved into {}.".format(args.output))
    
    submodel_input = torch.zeros((1, 256, 256, 256)).to(cfg.MODEL.DEVICE)
    submodel_output = "/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826-dd_submodel.onnx"
    submodel = model.proposal_generator.fcos_head.bbox_tower
    submodel.add_module("top_layer", model.top_layer)
    torch.onnx.export(
        submodel,
        submodel_input,
        submodel_output,
        verbose = False,
        export_params=True,
        input_names=["feature"],
        output_names=["top_feat"],
        keep_initializers_as_inputs=False,
        opset_version=11,
    )
    

if __name__ == "__main__":
    main()
