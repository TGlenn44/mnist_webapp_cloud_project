
from torch import nn

# Fully Connected Neural Network for Image Classification
class FullyConnectedClassifier(nn.Module):
    def __init__(self):
        super(FullyConnectedClassifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 把 28x28 展平为 784
            nn.Linear(28 * 28, 256),  # 第一层，全连接
            nn.ReLU(),
            nn.Linear(256, 128),  # 第二层
            nn.ReLU(),
            nn.Linear(128, 64),  # 第三层
            nn.ReLU(),
            nn.Linear(64, 10),  # 输出层，10 类
        )

    def forward(self, x):
        return self.fc_layers(x)
