import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCA(nn.Module):
    def __init__(self, channels=16, hidden=128, fire_rate=0.5):
        super().__init__()

        self.channels = channels
        self.fire_rate = fire_rate

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32) / 8.0

        sobel_y = sobel_x.T.contiguous()

        kernels = torch.stack([sobel_x, sobel_y])
        self.register_buffer("sobel", kernels[:, None])

        self.network = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1)
        )

        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def perceive(self, x):
        gradients = []

        for kernel in self.sobel:
            weight = kernel.unsqueeze(0).repeat(
                self.channels, 1, 1, 1
            )
            grad = F.conv2d(
                x, weight, padding=1, groups=self.channels
            )
            gradients.append(grad)

        return torch.cat([x] + gradients, dim=1)

    def alive_mask(self, x):
        alpha = x[:, 3:4]
        return F.max_pool2d(
            alpha, kernel_size=3, stride=1, padding=1
        ) > 0.1

    def forward(self, x):
        pre_alive = self.alive_mask(x)

        perception = self.perceive(x)
        dx = self.network(perception)

        random_mask = (
            torch.rand_like(x[:, :1]) <= self.fire_rate
        ).to(x.dtype)

        x = x + dx * random_mask

        post_alive = self.alive_mask(x)
        x = x * (pre_alive & post_alive).to(x.dtype)

        return x
