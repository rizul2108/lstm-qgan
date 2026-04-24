import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_size=784, use_sigmoid=False):
        super().__init__()
        self.image_size  = image_size
        self.use_sigmoid = use_sigmoid

        self.features = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.head    = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.sigmoid(self.head(self.features(x)))
