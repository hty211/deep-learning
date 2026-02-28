# 经典CNN架构

## 1. LeNet-5 (1998)

LeNet是最早的卷积神经网络之一，用于手写数字识别。

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.pool1(self.tanh(self.conv1(x)))
        x = self.pool2(self.tanh(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

model = LeNet5()
x = torch.randn(1, 1, 28, 28)
print(f"LeNet-5 输入: {x.shape}")
print(f"LeNet-5 输出: {model(x).shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

## 2. AlexNet (2012)

AlexNet在ImageNet竞赛中取得突破性成果。

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = AlexNet(num_classes=10)
x = torch.randn(1, 3, 224, 224)
print(f"AlexNet 输入: {x.shape}")
print(f"AlexNet 输出: {model(x).shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

## 3. VGG (2014)

VGG使用小卷积核(3x3)构建深层网络。

```python
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3, padding=1
            ))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = VGG16(num_classes=10)
x = torch.randn(1, 3, 224, 224)
print(f"VGG16 输入: {x.shape}")
print(f"VGG16 输出: {model(x).shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

## 4. ResNet (2015)

ResNet引入残差连接，解决深层网络训练问题。

### 4.1 残差块

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = ResNet18(num_classes=10)
x = torch.randn(1, 3, 224, 224)
print(f"ResNet18 输入: {x.shape}")
print(f"ResNet18 输出: {model(x).shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

## 5. 使用预训练模型

```python
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

print(f"ResNet18 参数量: {sum(p.numel() for p in resnet18.parameters()):,}")
print(f"ResNet50 参数量: {sum(p.numel() for p in resnet50.parameters()):,}")
print(f"VGG16 参数量: {sum(p.numel() for p in vgg16.parameters()):,}")

num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 10)

print(f"\n修改后ResNet18最后一层: {resnet18.fc}")
```

## 6. 模型对比

| 模型 | 年份 | 深度 | 参数量 | 特点 |
|------|------|------|--------|------|
| LeNet-5 | 1998 | 5 | 60K | 鼻祖，手写识别 |
| AlexNet | 2012 | 8 | 60M | ReLU, Dropout |
| VGG16 | 2014 | 16 | 138M | 小卷积核 |
| ResNet18 | 2015 | 18 | 11M | 残差连接 |
| ResNet50 | 2015 | 50 | 25M | 深层残差 |

## 7. 模型选择建议

```python
def get_model(model_name, num_classes=10, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

for name in ['resnet18', 'resnet50', 'vgg16', 'mobilenet_v2']:
    model = get_model(name, num_classes=10, pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} 参数")
```

## 8. 总结

- **LeNet**：开创性工作，简单有效
- **AlexNet**：引入ReLU、Dropout
- **VGG**：小卷积核堆叠
- **ResNet**：残差连接解决退化问题
- **迁移学习**：使用预训练模型加速训练
