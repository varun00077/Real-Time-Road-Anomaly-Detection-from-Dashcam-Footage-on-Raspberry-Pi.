import ultralytics
from ultralytics import YOLO

# Load YOLOv8n model from best.pt
model = YOLO('best.pt')

# Export the model to ONNX format with opset 21
model.export(format='onnx', dynamic=True, imgsz=640, simplify=True, opset=21)

print('Conversion to ONNX completed: best.onnx')
