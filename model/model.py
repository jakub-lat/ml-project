import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(5, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 88 * 88, 120),
            nn.Linear(120, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 88 * 88)
        x = self.fc(x)
        return x
