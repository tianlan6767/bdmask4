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
        
        # pred_mask_logits = patch_blender(basis_out["bases"][0], proposals, gt_instances)
        return basis_out["bases"][0], proposals

    model.forward = types.MethodType(forward, model)
    # output
    output_names.extend(["bases"])
    output_names.extend(["pred"])
    # for item in ["pred", "mask_logits"]:
    #     output_names.extend([item])


def patch_blender(bases, proposals, gt_instances):
    box_pred = proposals[0][:,:,:4]
    index = torch.zeros((1,box_pred.shape[1],1), device=box_pred.device, dtype=box_pred.dtype)
    pooler_fmt_boxes = torch.cat([index, box_pred], 2).squeeze()
    top_feat =proposals[1].squeeze()
    rois = roi_align(
        bases, 
        pooler_fmt_boxes, 
        output_size=(56, 56), 
        spatial_scale=(0.25),
        sampling_ratio=1,
        aligned=True
    )
    pred_mask_logits = merge_bases(rois, top_feat).sigmoid()
    pred_mask_logits = pred_mask_logits.view(
    -1, 1, 56, 56)
    
    # do_paste_mask(pred_mask_logits, box_pred.squeeze(), 2048, 2048)
    return pred_mask_logits

def merge_bases(rois, coeffs):
    # merge predictions
    N, coeffs_b = coeffs.size()
    N, B, H, W = rois.size()
    coeffs = coeffs.view(-1, B, int(math.sqrt(coeffs_b // B)), int(math.sqrt(coeffs_b // B)))
    coeffs = F.interpolate(coeffs, (H, W),
                            mode='bilinear', align_corners=False).softmax(dim=1)
    masks_preds = (rois * coeffs).sum(dim=1)
    return masks_preds.view(N, (H * W))  


def patch_fcos(cfg, proposal_generator):
    def proposal_generator_forward(self, images, features, gt_instances=None, top_module=None):
        # print("当前使用",features.keys(), "\n")
        features = [features[f] for f in self.in_features]
        # print("当前使用********* :",self.in_features, len(features), "\n")
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(features, top_module, self.yield_proposal)
        results = predict_proposals(cfg, logits_pred, reg_pred, ctrness_pred, locations, top_feats)
        return results

    proposal_generator.forward = types.MethodType(proposal_generator_forward, proposal_generator)


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full_like(
        box_tensor[:, :1], batch_index, dtype=box_tensor.dtype, device=box_tensor.device
    )
    return torch.cat((repeated_index, box_tensor), dim=1)

def predict_proposals(cfg, logits_pred, reg_pred, ctrness_pred, locations, top_feats=None):
      
    strides = cfg.MODEL.FCOS.FPN_STRIDES
    bundle = {
        "l": locations, "o": logits_pred,
        "r": reg_pred, "c": ctrness_pred,
        "s":strides
    }
    
    if len(top_feats) > 0:
        bundle['t'] = top_feats
    
    merge_location = []
    merge_logits_pred = []
    merge_box_regression = []
    merge_top_feat = []
    merge_ctrness_pred = []
    for i, per_bundle in enumerate(zip(*bundle.values())):
        per_bundle = dict(zip(bundle.keys(), per_bundle))
        l = per_bundle["l"]
        o = per_bundle["o"]
        r = per_bundle["r"] * per_bundle["s"]
        c = per_bundle["c"]
        t = per_bundle["t"] if "t" in bundle else None
        out_logits_pred, out_box_regression, out_top_feat, out_ctrness_pred = forward_for_single_feature_map(cfg, o, r, c, t)
        # print(l.shape, out_logits_pred.shape, out_box_regression.shape, out_top_feat.shape)
        merge_location.append(l)
        merge_logits_pred.append(out_logits_pred)
        merge_box_regression.append(out_box_regression)
        merge_top_feat.append(out_top_feat)
        merge_ctrness_pred.append(out_ctrness_pred)
    
    merge_location = torch.cat(merge_location, 0) # torch.Size([65536, 2])  torch.Size([16384, 2]) torch.Size([4096, 2]) 
    merge_logits_pred = torch.cat(merge_logits_pred, 1)
    merge_logits_pred_max = torch.max(merge_logits_pred, 2)[0][:,:,None]
    merge_box_regression = torch.cat(merge_box_regression, 1).squeeze()
    merge_top_feat = torch.cat(merge_top_feat, 1) 
    detections = torch.stack([
        merge_location[:, 0] - merge_box_regression[:, 0],
        merge_location[:, 1] - merge_box_regression[:, 1],
        merge_location[:, 0] + merge_box_regression[:, 2],
        merge_location[:, 1] + merge_box_regression[:, 3],
    ], dim=1)
    # batch_index = torch.zeros((1, detections.shape[0], 1), dtype=detections.dtype, device=detections.device)
    # pred = torch.cat([batch_index, detections[None], merge_logits_pred_max, merge_logits_pred], 2)
    pred = torch.cat([detections[None], merge_logits_pred_max, merge_logits_pred, merge_top_feat], 2)
    
    return pred
    

def forward_for_single_feature_map(cfg, logits_pred, reg_pred,ctrness_pred, top_feat=None):
    thresh_with_ctr = cfg.MODEL.FCOS.THRESH_WITH_CTR
    N, C, H, W = logits_pred.shape

    # put in the same format as locations
    logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)      # (1,25,256,256) => (1,256,256,25)
    logits_pred = logits_pred.reshape(N, -1, C).sigmoid()               # (1,256,256,25) => (1,65536,25)
    box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)      # (1,4,256,256) => (1,256,256,4)
    box_regression = box_regression.reshape(N, -1, 4)                   # (1,256,256,4) => (1,65536,4)
    ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)    # (1,1,256,256) => (1,256,256,1)
    ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()                # (1,256,256,1) => (1,65536)
    if top_feat is not None:
        top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)       # (1,784,256,256) => (1,256,256,784)
        top_feat = top_feat.reshape(N, H * W, -1)                       # (1,256,256,784) => (1,65536)

    # if self.thresh_with_ctr is True, we multiply the classification
    # scores with centerness scores before applying the threshold.
    if thresh_with_ctr:
        logits_pred = logits_pred * ctrness_pred[:, :, None]

    if not thresh_with_ctr:
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
    # cfg.MODEL.BACKBONE.FREEZE_AT = 0
        
    cfg.INPUT.FORMAT = 'BGR'
    cfg.MODEL.PIXEL_MEAN = [41,41,41]
    cfg.MODEL.PIXEL_STD = [34,34,34]


    cfg.MODEL.BASIS_MODULE.NORM = 'BN'
    cfg.freeze()
    # print("*********************",cfg.MODEL.FCOS.IN_FEATURES)
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
    dummy_input = torch.zeros((1, args.channel, height, width)).to(cfg.MODEL.DEVICE)
    # print(dummy_input.shape, dummy_input.type())
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
        opset_version=11
    )

    onnx_model = onnx.load(args.output)
    model_simp, check = simplify(onnx_model)
    assert check,  "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output)
    print("Done. The onnx model is saved into {}.".format(args.output))
    

if __name__ == "__main__":
    main()
