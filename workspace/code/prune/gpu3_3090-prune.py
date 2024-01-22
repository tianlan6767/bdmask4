from adet.config import get_cfg
import torch
from torch.nn import BatchNorm2d
import numpy as np
from detectron2.engine import DefaultPredictor
import torch_pruning as tp

def init(key, iv):
    cfg = get_cfg()
    config_file = r'/home/ps/adet/AdelaiDet/configs/BlendMask/R_50_3x.yaml'
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TRAIN = ("phone",)
    cfg.DATASETS.TEST = ("phone",)   # no metrics implemented for this dataset

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 25
    cfg.MODEL.FCOS.NUM_CLASSES = 25
    
    cfg.MODEL.WEIGHTS = r"/media/ps/data/train/LQ/task/prune/weights-allbn-orig/model_0022999.pth"
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.16
    cfg.MODEL.DEVICE = "cuda:0"
    cfg.MODEL.KEY = key
    cfg.MODEL.IV =  iv
    
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.PRUNE = True
    cfg.MODEL.FCOS.CENTER_SAMPLE = "center"
    cfg.MODEL.BASIS_MODULE.LOSS_ON=False

    # cfg.INPUT.FORMAT = 'L'
    # cfg.MODEL.PIXEL_MEAN = [86]
    # cfg.MODEL.PIXEL_STD = [76]

    cfg.INPUT.FORMAT = 'BGR'
    cfg.MODEL.PIXEL_MEAN = [40,40,40]
    cfg.MODEL.PIXEL_STD = [40,40,40]
    
    global predictor
    predictor = DefaultPredictor(cfg)
    return cfg 

def tp_prune(model, input, prune_model_file, save_model=False):
    dummy_input = [{"image": image, "height": 2048, "width": 2048}]
     # TP
    # 1. 使用我们上述定义的重要性评估
    imp = tp.importance.MagnitudeImportance(p=1)

    ignored_layers = [model.backbone.fpn_output5,
                      model.proposal_generator.fcos_head.cls_logits,
                      model.proposal_generator.fcos_head.bbox_pred,
                      model.proposal_generator.fcos_head.ctrness,
                      model.top_layer,
                      model.basis_module.tower[5]
                      ]
    # 3. 初始化剪枝器
    for p in model.parameters():
        p.requires_grad_(True)
    iterative_steps = 1
    unwrapped_parameters = []
    # 构建剪枝网络
    pruner = tp.pruner.MetaPruner(
        model,
        dummy_input,  # 用于分析依赖的伪输入
        importance=imp,  # 重要性评估指标
        iterative_steps=iterative_steps,  # 迭代剪枝，设为1则一次性完成剪枝
        global_pruning=True,
        pruning_ratio=0.5,
        ignored_layers=ignored_layers,  
        unwrapped_parameters=unwrapped_parameters
    )

    #########################################
    # Pruning 
    #########################################
    # print("==============Before pruning=================")
    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_input)
    pruner.step()
    # print("==============After pruning=================")
    # pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, dummy_input)
    # print(model)
    # print("Before Pruning: MACs=%f G, #Params=%f M" % (base_macs / 1e9, base_nparams / 1e6))
    # # print("After Pruning: MACs=%f G, #Params=%f M" % (pruned_macs / 1e9, pruned_nparams / 1e6)
    print(model)
    if save_model:
        torch.save({"struct":model,
                    "model":model.state_dict()}, prune_model_file)

if __name__ == "__main__":
    cfg = init("8Xe0efbbhSPHmaw0", "OwXaWuIhMzErsKl5")
    prune_model_file = r"/media/ps/data/train/LQ/task/prune/prune_tptmp.pth"
    model = predictor.model.eval()
    image = torch.rand(3, 2048, 2048)
    # tp 剪枝
    tp_prune(model, image, prune_model_file, save_model=True)


    