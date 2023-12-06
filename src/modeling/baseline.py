from torch import nn


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        # intput size (..., 1, 48, 48)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid', dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid'),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            # in_features=c*w*h=16*36*36
            nn.Linear(in_features=20736, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=7),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        y = self.head(x)
        return y
