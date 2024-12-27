# Tegra-infer

## Environment configuration

Not include python env
```bash
./scripts/init_env.sh
```

## Build Project

```bash
./script/init_repo.sh
cmake -B build -S . -GNinja
cmake --build build -t main
# output is ./build/src/main
```

## Process

Fetch or download yolo11 onnx model

```bash
python export_model.py
```

Build tensorrt engine file, ref `scripts/build_engine.sh`

```bash
/usr/src/tensorrt/bin/trtexec --onnx=./yolo11n.onnx --timingCacheFile=tc --profilingVerbosity=detailed --builderOptimizationLevel=3 --useCudaGraph --useSpinWait --inputIOFormats=fp32:hwc  --separateProfileRun --saveEngine=yolon.fp32.fp32hwc.engine
```

Edit `config.json`

```bash
...
"trt_engine": "yolon.fp32.fp32hwc.engine",
...
```

Run
```bash
# will load config.json in current dir
./build/src/main
```

## Helper

### Query all camera

```bash
v4l2-ctl --list-devices
```

### Query specific device

```bash
# example for <device>: `/dev/video0`
v4l2-ctl --device=<device> --list-formats-ext
```
