
# Just run:
# yolo export model=best.pt format=onnx imgsz=800

# import torch
# from pathlib import Path
# from utilities import load_detection_model


# def main():
#     modelPath = Path("/Users/uzbit/Documents/projects/tableizer/tableizer/exp7/weights/best.pt")
#     model = load_detection_model(modelPath)
#     dummy_input = torch.randn(1, 3, 800, 800)
#     torch.onnx.export(model, dummy_input, modelPath.with_suffix(".onnx"), opset_version=12)


# if __name__ == "__main__":
#     main()