#! /usr/bin/bash 

export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/lib64:$LD_LIBRARY_PATH
cd /home/ps/anaconda3/envs/py-38/lib/python3.8/site-packages/trtpy/trt852cuda115cudnn8/bin
./trtexec --onnx=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy.onnx \
          --saveEngine=/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dy-4096-0908 \
          --minShapes=input_image:1x1x2048x2048 \
          --optShapes=input_image:1x1x4096x5472  \
          --maxShapes=input_image:1x1x4096x5472  \
          --fp16 \
          --device=3 \
          --workspace=10240
