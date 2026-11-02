import os
import random
import numpy as np
import torch
from PIL import Image
from nca_model import NeuralCA

# Training configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
GRID_SIZE = 72
POOL_SIZE = 1024
BATCH_SIZE = 8
TRAIN_STEPS = 500
MIN_CA_STEPS = 64
MAX_CA_STEPS = 96
LEARNING_RATE = 0.001

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load target
image = Image.open("data/target_72.png").convert("RGBA")
target = torch.from_numpy(
    np.asarray(image, dtype=np.float32).copy() / 255.0
).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Load our previously trained model
model = NeuralCA().to(DEVICE)
model.load_state_dict(
    torch.load(
        "models/nca_pool_full.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

def make_seed(batch_size=1):
    x = torch.zeros(
        batch_size, 16, GRID_SIZE, GRID_SIZE,
        device=DEVICE
    )
    x[:, 3:, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0
    return x

# Store pool on CPU to reduce GPU memory usage
seed = make_seed().cpu()
pool = seed.repeat(POOL_SIZE, 1, 1, 1).clone()


def damage_batch(batch, radius_min=5, radius_max=14):
    """
    Remove random circular areas from cell states.
    All 16 channels are removed inside each circle.
    """
    batch = batch.clone()
    _, _, height, width = batch.shape

    yy, xx = torch.meshgrid(
        torch.arange(height, device=batch.device),
        torch.arange(width, device=batch.device),
        indexing="ij"
    )

    for i in range(batch.shape[0]):
        center_y = random.randint(height // 4, 3 * height // 4)
        center_x = random.randint(width // 4, 3 * width // 4)
        radius = random.randint(radius_min, radius_max)

        circle = (
            (xx - center_x) ** 2 +
            (yy - center_y) ** 2
        ) <= radius ** 2

        batch[i, :, circle] = 0.0

    return batch


loss_history = []

for step in range(TRAIN_STEPS):
    model.train()

    # Select different states from the pool
    indices = np.random.choice(
        POOL_SIZE, BATCH_SIZE, replace=False
    )
    batch = pool[indices].to(DEVICE).clone()

    # Replace the highest-loss sample with a new seed
    with torch.no_grad():
        errors = (
            (batch[:, :4] - target).square()
            .mean(dim=(1, 2, 3))
        )
        worst = torch.argmax(errors).item()
        batch[worst] = make_seed()[0]

        # Damage three of the best samples
        errors = (
            (batch[:, :4] - target).square()
            .mean(dim=(1, 2, 3))
        )

        best = torch.argsort(errors)[:3]
        batch[best] = damage_batch(batch[best])

    # Random number of growth steps
    ca_steps = random.randint(
        MIN_CA_STEPS, MAX_CA_STEPS
    )

    for _ in range(ca_steps):
        batch = model(batch)

    # Compute loss
    loss = (
        (batch[:, :4] - target).square()
        .mean(dim=(1, 2, 3))
        .mean()
    )

    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss detected")

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )

    optimizer.step()

    # Return new states to the pool
    pool[indices] = batch.detach().cpu()

    loss_history.append(loss.item())

    if (step + 1) % 50 == 0:
        checkpoint = {
            "step": step + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "pool": pool,
            "loss_history": loss_history.copy(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }

        torch.save(
            checkpoint,
            f"models/nca_regeneration_checkpoint_{step + 1}.pth"
        )

        np.save(
            "results/loss_regeneration.npy",
            np.array(loss_history)
        )

    print(
        f"Step {step + 1}/{TRAIN_STEPS} | "
        f"CA steps: {ca_steps} | "
        f"Loss: {loss.item():.6f}",
        flush=True
    )

# Save separate outputs
torch.save(
    model.state_dict(),
    "models/nca_regeneration.pth"
)

np.save(
    "results/loss_regeneration.npy",
    np.array(loss_history)
)

print("Regeneration training completed.")
print("Model saved: models/nca_regeneration.pth")

