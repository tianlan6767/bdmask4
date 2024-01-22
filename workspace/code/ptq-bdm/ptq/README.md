# 代码修改
- 文件路径:/home/ps/adet_v2/detectron2/detectron2/data/build.py
```python
## 411行
def _test_loader_from_config(cfg, dataset_name, mapper=None):
    ## 当前位置修改
    return {"dataset": dataset, "mapper": mapper, "num_workers": cfg.DATALOADER.NUM_WORKERS, "cfg":cfg}


## 445行
@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *,  mapper, cfg, num_workers=0):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    # 当前位置修改
    if mapper is not None:
        dataset = MapDataset(dataset, mapper, cfg)

```
# 运行
```python
    # 量化模型
    cmd_quantize(cfg, args, model, args.output, ignore_policy=ignore_policy)
    
    # 敏感层分析
    # cmd_sensitive_analysis(cfg, model, args.output)    

    # 导出模型
    # export_onnx(cfg, args, model, osp.join(args.output, "static.onnx"))
```
```bash
python /media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/ptq-bdm/ptq/export_model_to_onnx-dy-ptq.py
```