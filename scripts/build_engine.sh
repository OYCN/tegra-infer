#!/bin/bash

# median = 8.1814 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait        --inputIOFormats=fp32:chw  --separateProfileRun --saveEngine=fp32.fp32chw.engine
# median = 7.62463 ms
/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait        --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=fp32.fp32hwc.engine
#  median = 4.08179 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp16:chw  --separateProfileRun --saveEngine=fp16.fp32hwc.engine
# median = 4.05347 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp16:chw  --separateProfileRun --saveEngine=fp16.fp16chw.engine
# median = 3.95715 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp16:chw4 --separateProfileRun --saveEngine=fp16.fp16chw4.engine
