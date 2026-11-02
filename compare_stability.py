import torch
import matplotlib.pyplot as plt

from nca_model import NeuralCA

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
STEPS = [64, 96, 128, 192]
GRID_SIZE = 72

torch.manual_seed(42)

def evaluate_model(model_path):
    model = NeuralCA().to(DEVICE)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )
    model.eval()

    x = torch.zeros(1, 16, GRID_SIZE, GRID_SIZE, device=DEVICE)
    x[:, 3:, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0

    images = {}

    with torch.no_grad():
        for step in range(1, max(STEPS) + 1):
            x = model(x)

            if step in STEPS:
                rgba = x[0, :4].permute(1, 2, 0)
                images[step] = rgba.clamp(0, 1).cpu().numpy()

    return images

baseline = evaluate_model("models/nca_full.pth")

# Reset random seed for comparable random updates
torch.manual_seed(42)

pool_model = evaluate_model("models/nca_pool_full.pth")

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for col, step in enumerate(STEPS):
    axes[0, col].imshow(baseline[step])
    axes[0, col].set_title(f"{step} Steps")
    axes[0, col].axis("off")

    axes[1, col].imshow(pool_model[step])
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Baseline", fontsize=12)
axes[1, 0].set_ylabel("Sample Pool", fontsize=12)

plt.suptitle("Stability Comparison: Before vs After Sample Pool")
plt.tight_layout()
plt.savefig("results/stability_comparison.png", dpi=200)
plt.close()

print("Saved: results/stability_comparison.png")
