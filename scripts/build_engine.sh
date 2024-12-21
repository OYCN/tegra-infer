#!/bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --builderOptimizationLevel=3 --saveEngine=fp32.opt3.engine 2>&1 | tee fp32.opt3.log
/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --builderOptimizationLevel=3 --fp16 --saveEngine=fp16.opt3.engine 2>&1 | tee fp16.opt3.log
/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --builderOptimizationLevel=5 --fp16 --saveEngine=fp16.opt5.engine 2>&1 | tee fp16.opt5.log
