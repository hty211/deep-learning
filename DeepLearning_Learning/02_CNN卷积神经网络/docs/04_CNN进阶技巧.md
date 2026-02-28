# CNN进阶技巧

## 1. Batch Normalization

批归一化加速训练并提高稳定性。

```python
import torch
import torch.nn as nn

class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CNNWithBN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNBlock(3, 32),
            nn.MaxPool2d(2),
            ConvBNBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBNBlock(64, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = CNNWithBN()
x = torch.randn(32, 3, 32, 32)
output = model(x)
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
```

## 2. Dropout

Dropout防止过拟合。

```python
class CNNWithDropout(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = CNNWithDropout(dropout_rate=0.5)

model.train()
x = torch.randn(1, 3, 32, 32)
output1 = model(x)
output2 = model(x)
print(f"训练模式两次输出差异: {torch.abs(output1 - output2).max():.6f}")

model.eval()
output1 = model(x)
output2 = model(x)
print(f"评估模式两次输出差异: {torch.abs(output1 - output2).max():.6f}")
```

## 3. 数据增强

### 3.1 常用数据增强方法

```python
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                        std=[0.2470, 0.2435, 0.2616])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                        std=[0.2470, 0.2435, 0.2616])
])

print("训练数据增强:")
for t in transform_train.transforms:
    print(f"  - {t.__class__.__name__}")

print("\n测试数据增强:")
for t in transform_test.transforms:
    print(f"  - {t.__class__.__name__}")
```

### 3.2 自定义数据增强

```python
import torch
import numpy as np

class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam

cutout = Cutout(n_holes=1, length=8)
img = torch.randn(3, 32, 32)
img_cutout = cutout(img)
print(f"Cutout输入: {img.shape}")
print(f"Cutout输出: {img_cutout.shape}")

mixup = Mixup(alpha=0.2)
images = torch.randn(32, 3, 32, 32)
labels = torch.randint(0, 10, (32,))
mixed, la, lb, lam = mixup(images, labels)
print(f"\nMixup混合图像: {mixed.shape}")
print(f"混合系数: {lam:.4f}")
```

## 4. 学习率调度

### 4.1 StepLR

```python
import matplotlib.pyplot as plt

model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('StepLR Learning Rate Schedule')
plt.grid(True)
plt.show()
```

### 4.2 CosineAnnealingLR

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('CosineAnnealingLR Learning Rate Schedule')
plt.grid(True)
plt.show()
```

### 4.3 OneCycleLR

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, total_steps=50
)

lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('OneCycleLR Learning Rate Schedule')
plt.grid(True)
plt.show()
```

## 5. 早停（Early Stopping）

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

early_stopping = EarlyStopping(patience=5, min_delta=0.001)

val_losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48]

for epoch, loss in enumerate(val_losses):
    if early_stopping(loss):
        print(f"早停于 Epoch {epoch}")
        break
    print(f"Epoch {epoch}: val_loss = {loss:.4f}")
```

## 6. 模型保存与加载

```python
import os

model = CNNWithBN(num_classes=10)
optimizer = torch.optim.Adam(model.parameters())

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

save_checkpoint(model, optimizer, 0, 1.0, f"{save_dir}/model.pth")
print("模型已保存")

epoch, loss = load_checkpoint(model, optimizer, f"{save_dir}/model.pth")
print(f"模型已加载: epoch={epoch}, loss={loss}")

torch.save(model.state_dict(), f"{save_dir}/model_weights.pth")
print("\n仅保存权重")

model.load_state_dict(torch.load(f"{save_dir}/model_weights.pth"))
print("权重已加载")
```

## 7. 梯度裁剪

```python
model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

output = model(x)
loss = nn.CrossEntropyLoss()(output, y)

optimizer.zero_grad()
loss.backward()

grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"裁剪前梯度范数: {grad_norm_before:.4f}")

grad_norm_after = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"裁剪后梯度范数: {grad_norm_after:.4f}")
```

## 8. 总结

| 技巧 | 作用 | 使用场景 |
|------|------|---------|
| Batch Normalization | 加速训练、稳定收敛 | 几乎所有CNN |
| Dropout | 防止过拟合 | 全连接层后 |
| 数据增强 | 增加数据多样性 | 训练阶段 |
| 学习率调度 | 动态调整学习率 | 训练全过程 |
| 早停 | 防止过拟合 | 训练监控 |
| 梯度裁剪 | 防止梯度爆炸 | RNN、深层网络 |
