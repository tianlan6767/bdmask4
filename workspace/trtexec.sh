#! /usr/bin/bash 
export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/lib64:$LD_LIBRARY_PATH


# 转换fcos分支

cd /home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/bin
./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826.onnx \
          --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826 \
          --minShapes=input_image:1x3x2048x2048 \
          --optShapes=input_image:1x3x4096x5472  \
          --maxShapes=input_image:1x3x4096x5472  \
          --fp16 \
          --device=3 \
          --workspace=20480



# ./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dd.onnx \
#           --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dd \
#           --fp16 \
#           --device=3 \
#           --workspace=10240

# 转换mask分支
# ./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/blender.onnx \
#           --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/blender \
#           --minShapes=bases:1x4x512x512 \
#           --optShapes=bases:1x4x1024x1368  \
#           --maxShapes=bases:1x4x1024x1368  \
#           --fp16 \
#           --device=3 \
#           --workspace=20480

