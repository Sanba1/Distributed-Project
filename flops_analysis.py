import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis
from torchvision.models import resnet18, ResNet18_Weights

# Load ResNet model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Generate a dummy image input (batch size = 1, 3 channels, 224x224 resolution)
dummy_input = torch.randn(1, 3, 224, 224)

# Measure FLOPs
flop_counter = FlopCountAnalysis(model, dummy_input)
print(f"Total FLOPs: {flop_counter.total()}")
