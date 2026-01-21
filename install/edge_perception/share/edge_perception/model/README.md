# Semantic Segmentation Models

Place your ONNX segmentation model here. The default model name is `segmentation.onnx`.

## Example Models

You can use lightweight models for Jetson deployment:

- **BiSeNetV2**: Fast real-time segmentation
- **DDRNet-23-Slim**: Efficient dual-resolution network
- **SegFormer-B0**: Transformer-based, good accuracy/speed tradeoff
- **Fast-SCNN**: Ultra-fast for embedded devices

## Converting to ONNX

Example with PyTorch:
```python
import torch
model = ...  # Load your model
dummy_input = torch.randn(1, 3, 360, 640)
torch.onnx.export(model, dummy_input, "segmentation.onnx", 
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
```

## Model Requirements

- Input: NCHW format, RGB, float32, normalized to [0, 1]
- Output: (1, num_classes, H, W) logits
- Default input size: 640x360 (configurable via launch params)
