import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNWithBN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_models():
    print("=" * 50)
    print("CNN模型测试")
    print("=" * 50)
    
    x = torch.randn(32, 1, 28, 28)
    
    print("\n1. SimpleCNN")
    model = SimpleCNN(num_classes=10)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n2. CNNWithBN")
    model = CNNWithBN(num_classes=10)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")
    
    print("\n3. SimpleResNet")
    model = SimpleResNet(num_classes=10)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")

def demonstrate_conv_operations():
    print("\n" + "=" * 50)
    print("卷积操作演示")
    print("=" * 50)
    
    x = torch.randn(1, 3, 32, 32)
    
    print("\n1. 不同卷积核大小")
    for k in [1, 3, 5, 7]:
        conv = nn.Conv2d(3, 16, kernel_size=k, padding=k//2)
        out = conv(x)
        print(f"kernel_size={k}: {x.shape} -> {out.shape}")
    
    print("\n2. 不同步长")
    for s in [1, 2, 4]:
        conv = nn.Conv2d(3, 16, kernel_size=3, stride=s, padding=1)
        out = conv(x)
        print(f"stride={s}: {x.shape} -> {out.shape}")
    
    print("\n3. 不同填充")
    for p in [0, 1, 2]:
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=p)
        out = conv(x)
        print(f"padding={p}: {x.shape} -> {out.shape}")
    
    print("\n4. 池化操作")
    pool = nn.MaxPool2d(2, 2)
    out = pool(x)
    print(f"MaxPool2d(2,2): {x.shape} -> {out.shape}")
    
    avgpool = nn.AdaptiveAvgPool2d(1)
    out = avgpool(x)
    print(f"AdaptiveAvgPool2d(1): {x.shape} -> {out.shape}")

def demonstrate_feature_extraction():
    print("\n" + "=" * 50)
    print("特征提取演示")
    print("=" * 50)
    
    model = SimpleCNN(num_classes=10)
    
    x = torch.randn(1, 1, 28, 28)
    
    print("\n各层输出形状:")
    
    x1 = model.pool(F.relu(model.conv1(x)))
    print(f"Conv1 + Pool: {x1.shape}")
    
    x2 = model.pool(F.relu(model.conv2(x1)))
    print(f"Conv2 + Pool: {x2.shape}")
    
    x3 = model.pool(F.relu(model.conv3(x2)))
    print(f"Conv3 + Pool: {x3.shape}")
    
    x_flat = x3.view(x3.size(0), -1)
    print(f"Flatten: {x_flat.shape}")
    
    x4 = F.relu(model.fc1(x_flat))
    print(f"FC1: {x4.shape}")
    
    output = model.fc2(x4)
    print(f"FC2 (Output): {output.shape}")

if __name__ == "__main__":
    test_models()
    demonstrate_conv_operations()
    demonstrate_feature_extraction()
    print("\n" + "=" * 50)
    print("CNN基础示例完成!")
    print("=" * 50)
