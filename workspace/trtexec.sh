#! /usr/bin/bash 
export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/lib64:$LD_LIBRARY_PATH

# 转换fcos分支

cd /home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/bin
# ./trtexec --onnx=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JT/model-1016-batch10.onnx \
#           --saveEngine=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JT/model-1016-batch10 \
#           --minShapes=input_image:1x3x2048x2048 \
#           --optShapes=input_image:10x3x4096x4096  \
#           --maxShapes=input_image:10x3x4096x5472 \
#           --fp16 \
#           --device=3 \
#           --workspace=10240 \
#           --preview=+fasterDynamicShapes0805



# ./trtexec --onnx=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/OQC/model_0413999-orig.onnx \
#           --saveEngine=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/OQC/model3/ptq-all-trainall_hasrule-all-basicblock100 \
#           --fp16 --int8 \
#           --device=0 \
#           --workspace=10240 \
#           --exportLayerInfo=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/OQC/model3/ptq-all-trainall_hasrule-all-basicblock100-layer.json \
#           --profilingVerbosity=detailed  \
#           --exportProfile=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/OQC/model3/ptq-all-trainall_hasrule-all-basicblock100_profile.json


# 转换mask分支
# ./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/blender-new-boxMask-nogrid-samples.onnx \
#           --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/blender-new-boxMask-nogrid-samples \
#           --minShapes=bases:1x4x512x512 \
#           --optShapes=bases:1x4x1024x2048  \
#           --maxShapes=bases:1x4x1024x2048 \
#           --fp16 \
#           --device=3 \
#           --workspace=4096

# ./trtexec --onnx=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/model_0413999-orig.onnx \
#           --saveEngine=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/model_0413999-orig-fp32 \
#           --device=3 \
#           --workspace=10240

# 转换ptq模型
# ./trtexec --onnx=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/ptq.onnx \
#           --saveEngine=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/OQC-ptq \
#           --dumpLayerInfo \
#           --exportLayerInfo=/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/OQC-ptq.graph \
#           --fp16 --int8 \
#           --workspace=10240 \
#           --device=3 \
#           --profilingVerbosity=detailed > ./output.txt


