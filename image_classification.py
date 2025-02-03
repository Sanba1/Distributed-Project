import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import requests

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Download an example image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/640px-Example.jpg"
image_path = "sample.jpg"

response = requests.get(image_url, stream=True)
with open(image_path, 'wb') as f:
    f.write(response.content)

# Load image
image = Image.open(image_path)
plt.imshow(image)
plt.axis("off")
plt.show()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(image_tensor)

# Load ImageNet class labels
imagenet_classes = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.split("\n")

# Get top 5 predictions
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Print results
for i in range(5):
    print(f"{imagenet_classes[top5_catid[i]]}: {top5_prob[i].item():.4f}")
