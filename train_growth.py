import os
import numpy as np
import torch
from PIL import Image
from nca_model import NeuralCA

# Configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
TRAIN_STEPS = 500
LEARNING_RATE = 0.002
GRID_SIZE = 72

torch.manual_seed(42)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load target image
image = Image.open("data/target_72.png").convert("RGBA")
target = torch.from_numpy(
    np.asarray(image, dtype=np.float32).copy() / 255.0
).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Initialize model
model = NeuralCA().to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

def make_seed():
    x = torch.zeros(1, 16, GRID_SIZE, GRID_SIZE, device=DEVICE)
    x[:, 3:, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0
    return x

loss_history = []

# Training
for step in range(TRAIN_STEPS):
    ca_steps = int(np.random.randint(64, 97))
    x = make_seed()

    for _ in range(ca_steps):
        x = model(x)

    loss = ((x[:, :4] - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (step + 1) % 50 == 0:
        torch.save(
            model.state_dict(),
            f"models/nca_checkpoint_{step + 1}.pth"
        )
        np.save(
            "results/loss_full.npy",
            np.array(loss_history)
        )

    print(
        f"Training {step + 1}/{TRAIN_STEPS} "
        f"| Loss: {loss.item():.6f}",
        flush=True
    )

# Save model
torch.save(model.state_dict(), "models/nca_full.pth")

# Save loss values
np.save("results/loss_full.npy", np.array(loss_history))

# Generate sample image
model.eval()
with torch.no_grad():
    x = make_seed()
    for _ in range(96):
        x = model(x)

    rgba = x[0, :4].permute(1, 2, 0)
    rgba = (rgba.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(rgba, "RGBA").save(
        "results/output_full.png"
    )

print("\nTraining completed.")
print("Model saved: models/nca_full.pth")
print("Loss saved: results/loss_full.npy")
print("Image saved: results/output_full.png")
