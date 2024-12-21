from ultralytics import YOLO

model = YOLO("yolo11n.pt")
# model.export(format="onnx")
onnx_model = YOLO("yolo11n.onnx")
results = onnx_model("./bus.jpg")

