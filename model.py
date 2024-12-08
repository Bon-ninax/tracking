import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingModel, self).__init__()
        # 保存されたモデルに合わせたアーキテクチャ
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
