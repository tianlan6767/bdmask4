# 剪枝

### 1.安装

```python
pip install torch-pruning 
```

### 2.代码修改

- 文件路径:/home/ps/adet_v2/AdelaiDet/adet/modeling/blendmask/blendmask.py

  ```python
  class BlendMask(nn.Module):
      """
      Main class for BlendMask architectures (see https://arxiv.org/abd/1901.02446).
      """
  
      def __init__(self, cfg):
          super().__init__()
          self.device = torch.device(cfg.MODEL.DEVICE)
          self.instance_loss_weight = cfg.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT
          if cfg.MODEL.FINETUNE:
              assert cfg.MODEL.WEIGHTS, "finetuning model need a model file!"
              model = torch.load(cfg.MODEL.WEIGHTS,map_location=self.device)["struct"]
              
              self.backbone = model.backbone
              self.proposal_generator = model.proposal_generator
              self.blender = model.blender
              self.basis_module = model.basis_module
              self.top_layer = model.top_layer
  
              self.model_prune = False
              self.proposal_generator.model_prune = False
              self.basis_module.loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
          else:
              self.backbone = build_backbone(cfg)
              self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
              self.blender = build_blender(cfg)
              self.basis_module = build_basis_module(cfg, self.backbone.output_shape())
              # build top module
              self.proposal_generator.model_prune = cfg.MODEL.PRUNE
              self.model_prune = cfg.MODEL.PRUNE
              in_channels = cfg.MODEL.FPN.OUT_CHANNELS
              num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
              attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE
              attn_len = num_bases * attn_size * attn_size
              self.top_layer = nn.Conv2d(
                  in_channels, attn_len,
                  kernel_size=3, stride=1, padding=1)
              torch.nn.init.normal_(self.top_layer.weight, std=0.01)
              torch.nn.init.constant_(self.top_layer.bias, 0)
          # options when combining instance & semantic outputs
          self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
          if self.combine_on:
              self.panoptic_module = build_sem_seg_head(cfg, self.backbone.output_shape())
              self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
              self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
              self.combine_instances_confidence_threshold = (
                  cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH)
  
          pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(len(cfg.MODEL.PIXEL_MEAN), 1, 1)
          pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(len(cfg.MODEL.PIXEL_MEAN), 1, 1)
          self.normalizer = lambda x: (x - pixel_mean) / pixel_std
          self.to(self.device)
  ```

- 文件路径:/home/ps/adet_v2/detectron2/detectron2/modeling/backbone/resnet.py

  ```python
  class BasicBlock(CNNBlockBase):
      """
      The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
      with two 3x3 conv layers and a projection shortcut if needed.
      """
  
      def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
          """
          Args:
              in_channels (int): Number of input channels.
              out_channels (int): Number of output channels.
              stride (int): Stride for the first conv.
              norm (str or callable): normalization for all conv layers.
                  See :func:`layers.get_norm` for supported format.
          """
          super().__init__(in_channels, out_channels, stride)
  
          if in_channels != out_channels:
              self.shortcut = Conv2d(
                  in_channels,
                  out_channels,
                  kernel_size=1,
                  stride=stride,
                  bias=False,
                  norm=None,
              )
              self.norm3 = get_norm(norm, out_channels)
          else:
              self.shortcut = None
  
          self.conv1 = Conv2d(
              in_channels,
              out_channels,
              kernel_size=3,
              stride=stride,
              padding=1,
              bias=False,
              norm=None,
          )
  
          self.norm1 = get_norm(norm, out_channels)
  
          self.conv2 = Conv2d(
              out_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              padding=1,
              bias=False,
              norm=None,
          )
          self.norm2 = get_norm(norm, out_channels)
  
          for layer in [self.conv1, self.conv2, self.shortcut]:
              if layer is not None:  # shortcut can be None
                  weight_init.c2_msra_fill(layer)
  
      def forward(self, x):
          out = self.conv1(x)
          out = self.norm1(out)
          out = F.relu_(out)
          out = self.conv2(out)
          out = self.norm2(out)
  
          if self.shortcut is not None:
              shortcut = self.norm3(self.shortcut(x))
          else:
              shortcut = x
  
          out += shortcut
          out = F.relu_(out)
          return out
  ```

- 文件路径:/home/ps/adet_v2/detectron2/detectron2/checkpoint/detection_checkpoint.py  91行

  ```python
      def save(self, name: str, **kwargs: Any) -> None:
          """
          Dump model and checkpointables to a file.
  
          Args:
              name (str): name of the file.
              kwargs (dict): extra arbitrary data to save.
          """
          if not self.save_dir or not self.save_to_disk:
              return
  
          data = {}
          data["struct"] = self.model  # 新增
          data["model"] = self.model.state_dict()
          for key, obj in self.checkpointables.items():
              data[key] = obj.state_dict()
          data.update(kwargs)
  
          basename = "{}.pth".format(name)
          save_file = os.path.join(self.save_dir, basename)
          assert os.path.basename(save_file) == basename, basename
          self.logger.info("Saving checkpoint to {}".format(save_file))
          with self.path_manager.open(save_file, "wb") as f:
              torch.save(data, cast(IO[bytes], f))
          self.tag_last_checkpoint(basename)
  ```

- 文件路径:/home/ps/adet_v2/AdelaiDet/adet/config/defaults.py

  ```python
  # 新增
  _C.MODEL.PRUNE = False   # forward 及时跳出
  _C.MODEL.FINETUNE = False      # 加载剪枝模型
  ```

### 3.剪枝使用

 ``` python
 # 模型剪枝
   cfg.MODEL.PRUNE = True
 # 剪枝模型微调训练
   cfg.MODEL.FINETUNE = True
   cfg.MODEL.RESNETS.NORM = "BN"
 ```



​	