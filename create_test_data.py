import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50

# Create model and load pretrained weights
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Create random input
x = torch.rand(1, 3, 224, 224)

# Get output
with torch.no_grad():
    output = model(x)['out']
    
print('Output shape:', output.shape)

# Save output for testing
torch.save(output, './tests/static/deeplabv3_resnet50.pth')
print('Test data saved successfully')
