#!/bin/bash

# yolon

# median = 7.62463 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait        --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolon.fp32.fp32hwc.engine
# median = 3.65143 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolon.fp16.fp32hwc.engine

# yolos

# median = 12.7041 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11s.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait        --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolos.fp32.fp32hwc.engine
# median = 5.97595 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11s.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolos.fp16.fp32hwc.engine

# yolom

# median = 28.2385 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11m.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait        --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolom.fp32.fp32hwc.engine
# median = 12.2662 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11m.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolom.fp16.fp32hwc.engine

# yolol

#
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11l.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait        --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolol.fp32.fp32hwc.engine
# median = 16.2296 ms
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11l.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolol.fp16.fp32hwc.engine

# yolox

#
#/usr/src/tensorrt/bin/trtexec --onnx=./yolo11x.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait        --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolox.fp32.fp32hwc.engine
# median = 32.2266 ms
/usr/src/tensorrt/bin/trtexec --onnx=./yolo11x.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --fp16 --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolox.fp16.fp32hwc.engine
