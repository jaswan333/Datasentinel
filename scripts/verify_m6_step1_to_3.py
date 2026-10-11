import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

print("=" * 60)
print("DataSentinel - Milestone 6: Deep Learning Pipeline Verification")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load CIFAR-10
cifar_dataset = datasets.CIFAR10(
    root="data/image/raw",
    train=True,
    download=False
)
print(f"\n1. CIFAR-10 Loaded:")
print(f"   Total images: {len(cifar_dataset)}")
sample_img, sample_lbl = cifar_dataset[0]
print(f"   Sample image type: {type(sample_img)}")
print(f"   Sample image size: {sample_img.size}")
print(f"   Sample image label: {sample_lbl}")

# 2. Convert to Tensor
transform = transforms.ToTensor()
tensor_img = transform(sample_img)
print(f"\n2. Image to Tensor:")
print(f"   Tensor shape: {tensor_img.shape}  (Channels, Height, Width)")
print(f"   Tensor dtype: {tensor_img.dtype}")

# 3. Create DataLoader
transformed_dataset = datasets.CIFAR10(
    root="data/image/raw",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)
loader = DataLoader(transformed_dataset, batch_size=64, shuffle=True, num_workers=0)
batch_images, batch_labels = next(iter(loader))
print(f"\n3. Batch DataLoader:")
print(f"   Batch images shape: {batch_images.shape}  (Batch, Channels, H, W)")
print(f"   Batch labels shape: {batch_labels.shape}")

# 4. Configure ResNet18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
print(f"\n4. ResNet18 Configured:")
print(f"   Original fc layer: {model.fc}")
model.fc = nn.Linear(model.fc.in_features, 10)
print(f"   Updated fc layer:  {model.fc}")
model = model.to(device)

# 5. One Forward Pass
model.eval()
with torch.no_grad():
    batch_images = batch_images.to(device)
    outputs = model(batch_images)
    predictions = outputs.argmax(dim=1)

print(f"\n5. One Forward Pass:")
print(f"   Output logits shape: {outputs.shape}")
print(f"   First 5 predicted class indices: {predictions[:5].tolist()}")
print(f"   First 5 actual labels:           {batch_labels[:5].tolist()}")
print("\n[SUCCESS] M6.1 - M6.3 verified end-to-end!")
