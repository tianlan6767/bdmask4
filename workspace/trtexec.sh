#! /usr/bin/bash 
export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/lib64:$LD_LIBRARY_PATH

# 转换fcos分支

cd /home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/bin
# ./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826.onnx \
#           --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826-8192 \
#           --minShapes=input_image:1x3x2048x2048 \
#           --optShapes=input_image:1x3x4096x8192  \
#           --maxShapes=input_image:1x3x4096x8192  \
#           --fp16 \
#           --device=3 \
#           --workspace=10240 \
#           --preview=+fasterDynamicShapes0805



# ./trtexec --onnx=static.onnx \
#           --saveEngine=static-imp-int8 \
#           --fp16 --int8 \
#           --device=1 \
#           --workspace=10240

# 转换mask分支
# ./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/blender-new-boxMask-nogrid-samples.onnx \
#           --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/blender-new-boxMask-nogrid-samples \
#           --minShapes=bases:1x4x512x512 \
#           --optShapes=bases:1x4x1024x2048  \
#           --maxShapes=bases:1x4x1024x2048 \
#           --fp16 \
#           --device=3 \
#           --workspace=4096

./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/model_1016-2.onnx \
          --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/model_1016-2\
          --fp16 \
          --device=0 \
          --workspace=4096

# 转换ptq模型
# ./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826_ptq.onnx \
#           --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826_ptq-2 \
#           --dumpLayerInfo \
#           --exportLayerInfo=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826_ptq.graph \
#           --fp16 --int8 \
#           --workspace=10240 \
#           --device=1 \
#           --profilingVerbosity=detailed > ./output.txt


