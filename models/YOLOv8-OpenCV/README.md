# Export YOLOv8 models to ONNX format for OpenCV Dnn module:

https://github.com/ultralytics/ultralytics
https://docs.ultralytics.com/tasks/

https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python
*Make sure to include "opset=12"

Install Pip install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.
```
!pip install ultralytics
```

Export to ONNX models
```
# Export detect models
!yolo export model=yolov8n.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8s.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8m.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8l.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8x.pt format=onnx opset=12  # export official model

# Export segment models
!yolo export model=yolov8n-seg.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8s-seg.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8m-seg.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8l-seg.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8x-seg.pt format=onnx opset=12  # export official model

# Export classify models
!yolo export model=yolov8n-cls.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8s-cls.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8m-cls.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8l-cls.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8x-cls.pt format=onnx opset=12  # export official model

# Export pose models
!yolo export model=yolov8n-pose.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8s-pose.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8m-pose.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8l-pose.pt format=onnx opset=12  # export official model
#!yolo export model=yolov8x-pose.pt format=onnx opset=12  # export official model
```

