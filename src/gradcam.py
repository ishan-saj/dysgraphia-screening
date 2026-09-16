import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_ID = "p06-052"

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "raw" / "IAM" / "images"
OUTPUT_DIR = BASE_DIR / "data" / "metadata"

OUTPUT_FILE = OUTPUT_DIR / f"{SAMPLE_ID}_gradcam.png"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(sample_id):

    matches = list(IMAGE_DIR.rglob(f"{sample_id}.png"))

    if not matches:
        raise FileNotFoundError(
            f"Image not found for {sample_id}"
        )

    return matches[0]


image_path = find_image(SAMPLE_ID)

print("=" * 70)
print(f"GRAD-CAM: {SAMPLE_ID}")
print("=" * 70)

print()
print("Image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

original = cv2.imread(str(image_path))

if original is None:
    raise ValueError("Could not read image.")

original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

# Resize to ResNet input size
image_resized = cv2.resize(
    original_rgb,
    (224, 224)
)


# ============================================================
# PREPARE TENSOR
# ============================================================

image_tensor = torch.tensor(
    image_resized,
    dtype=torch.float32
)

image_tensor = image_tensor.permute(
    2, 0, 1
)

image_tensor = image_tensor / 255.0

image_tensor = image_tensor.unsqueeze(0)

# ImageNet normalization
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)

image_tensor = (
    image_tensor - mean
) / std

image_tensor = image_tensor.to(device)


# ============================================================
# LOAD RESNET50
# ============================================================

print()
print("Loading pretrained ResNet50...")

model = resnet50(
    weights=ResNet50_Weights.DEFAULT
)

model = model.to(device)
model.eval()


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = []
gradients = []


# ============================================================
# HOOK FUNCTIONS
# ============================================================

def forward_hook(module, input, output):

    activations.append(
        output.detach()
    )


def backward_hook(module, grad_input, grad_output):

    gradients.append(
        grad_output[0].detach()
    )


# ResNet50 final convolutional layer
target_layer = model.layer4[-1].conv3

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# FORWARD PASS
# ============================================================

print("Running forward pass...")

output = model(image_tensor)

predicted_class = output.argmax(
    dim=1
).item()

print(
    f"Predicted ImageNet class: {predicted_class}"
)


# ============================================================
# BACKWARD PASS
# ============================================================

model.zero_grad()

target_score = output[
    0,
    predicted_class
]

target_score.backward()


# ============================================================
# GET ACTIVATIONS + GRADIENTS
# ============================================================

activation = activations[0][0]

gradient = gradients[0][0]


# ============================================================
# COMPUTE CHANNEL WEIGHTS
# ============================================================

weights = gradient.mean(
    dim=(1, 2)
)


# ============================================================
# CREATE CAM
# ============================================================

cam = torch.zeros(
    activation.shape[1:],
    dtype=torch.float32
)

for i, weight in enumerate(weights):

    cam += weight * activation[i]


# ReLU
cam = F.relu(cam)


# Normalize
cam -= cam.min()

if cam.max() > 0:

    cam /= cam.max()


# ============================================================
# RESIZE CAM
# ============================================================

cam = cam.cpu().numpy()

cam = cv2.resize(
    cam,
    (224, 224)
)


# ============================================================
# CREATE HEATMAP
# ============================================================

heatmap = np.uint8(
    255 * cam
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# OVERLAY
# ============================================================

overlay = (
    0.55 * image_resized +
    0.45 * heatmap
)

overlay = np.clip(
    overlay,
    0,
    255
).astype(np.uint8)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(image_resized)

plt.title(
    f"{SAMPLE_ID} - Original"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(cam, cmap="jet")

plt.title(
    "Grad-CAM"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(overlay)

plt.title(
    "Grad-CAM Overlay"
)

plt.axis("off")


plt.tight_layout()


plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("GRAD-CAM COMPLETED")
print("=" * 70)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("DONE.")