import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from nca_model import NeuralCA

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
GRID_SIZE = 72
GROW_STEPS = 96
REPAIR_STEPS = 64

target_img = Image.open("data/target_72.png").convert("RGBA")
target = torch.from_numpy(
    np.asarray(target_img, dtype=np.float32).copy() / 255.0
).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

def to_image(state):
    return state[0, :4].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

def evaluate_model(model_path, seed=42):
    torch.manual_seed(seed)

    model = NeuralCA().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    x = torch.zeros(1, 16, GRID_SIZE, GRID_SIZE, device=DEVICE)
    x[:, 3:, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0

    with torch.no_grad():
        for _ in range(GROW_STEPS):
            x = model(x)

        before = x.clone()

        # Remove right half of all 16 channels
        damaged = x.clone()
        damaged[:, :, :, GRID_SIZE // 2:] = 0.0

        after_damage = damaged.clone()

        for _ in range(REPAIR_STEPS):
            damaged = model(damaged)

        repaired = damaged.clone()

        loss_before = ((before[:, :4] - target) ** 2).mean().item()
        loss_damaged = ((after_damage[:, :4] - target) ** 2).mean().item()
        loss_repaired = ((repaired[:, :4] - target) ** 2).mean().item()

    return (
        [to_image(before), to_image(after_damage), to_image(repaired)],
        [loss_before, loss_damaged, loss_repaired]
    )

baseline_images, baseline_losses = evaluate_model(
    "models/nca_pool_full.pth"
)

regen_images, regen_losses = evaluate_model(
    "models/nca_regeneration.pth"
)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

titles = ["Before Damage", "After Damage", "After 64 Repair Steps"]

for row, images in enumerate([baseline_images, regen_images]):
    for col, image in enumerate(images):
        axes[row, col].imshow(image)
        axes[row, col].set_title(titles[col])
        axes[row, col].axis("off")

axes[0, 0].set_ylabel("Sample Pool", fontsize=12)
axes[1, 0].set_ylabel("Regeneration", fontsize=12)

plt.suptitle("Regeneration: Before vs After Damage Training")
plt.tight_layout()
plt.savefig("results/regeneration_comparison.png", dpi=200)
plt.close()

print("\nL2 Loss Comparison")
print("                     Before     Damaged    Repaired")
print(
    "Sample Pool:       ",
    *[f"{v:.6f}" for v in baseline_losses]
)
print(
    "Regeneration:      ",
    *[f"{v:.6f}" for v in regen_losses]
)
print("\nSaved: results/regeneration_comparison.png")
