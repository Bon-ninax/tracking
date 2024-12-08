import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
from PIL import Image

# デバイスの選択（MPS対応）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# データ前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# カスタムデータセット（TripletImageFolder）
class TripletImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.imgs = self.dataset.imgs

    def __getitem__(self, index):
        # アンカー画像
        anchor_img, label = self.dataset[index]
        
        # 同じクラスのポジティブ画像
        positive_idx = self._get_positive_index(label)
        positive_img, _ = self.dataset[positive_idx]
        
        # 異なるクラスのネガティブ画像
        negative_idx = self._get_negative_index(label)
        negative_img, _ = self.dataset[negative_idx]

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def _get_positive_index(self, label):
        # 同じクラスのランダムなインデックスを選択
        same_class_indices = [i for i, (_, lbl) in enumerate(self.dataset.imgs) if lbl == label]
        return random.choice(same_class_indices)

    def _get_negative_index(self, label):
        # 異なるクラスのランダムなインデックスを選択
        different_class_indices = [i for i, (_, lbl) in enumerate(self.dataset.imgs) if lbl != label]
        return random.choice(different_class_indices)

    def __len__(self):
        return len(self.dataset)

# モデルの準備
class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingModel, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')  # weightsでモデルをロード
        self.backbone.classifier = nn.Identity()  # 分類器を削除
        self.embedding = nn.Linear(960, embedding_dim)  # MobileNetV3の出力をembedding_dimに接続

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings

# モデルの準備
embedding_dim = 128
model = EmbeddingModel(embedding_dim).to(device)

# 損失関数とオプティマイザ
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# データセットとデータローダーの作成
data_dir = "data"  # データセットのパス
train_dataset = TripletImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset = TripletImageFolder(os.path.join(data_dir, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 学習ループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for anchor, positive, negative in tqdm(train_loader):
        # デバイスに転送
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # 順伝播
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        # 損失計算
        loss = criterion(anchor_out, positive_out, negative_out)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

# モデルの保存
torch.save(model.state_dict(), "embedding_model.pth")
print("Model saved to embedding_model.pth")
