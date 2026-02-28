import torch
import torch.nn as nn
import torch.nn.functional as F

def test_conv_calculations():
    print("=" * 50)
    print("卷积计算练习")
    print("=" * 50)
    
    print("\n练习1: 计算输出尺寸")
    def calc_output_size(input_size, kernel_size, stride=1, padding=0):
        return (input_size - kernel_size + 2 * padding) // stride + 1
    
    test_cases = [
        (32, 3, 1, 0),
        (32, 3, 1, 1),
        (32, 3, 2, 1),
        (224, 7, 2, 3),
        (28, 5, 1, 2),
    ]
    
    for input_size, kernel_size, stride, padding in test_cases:
        output = calc_output_size(input_size, kernel_size, stride, padding)
        print(f"输入={input_size}, 卷积核={kernel_size}, 步长={stride}, 填充={padding} -> 输出={output}")
    
    print("\n练习2: 验证PyTorch卷积输出")
    for input_size, kernel_size, stride, padding in test_cases:
        conv = nn.Conv2d(3, 16, kernel_size, stride=stride, padding=padding)
        x = torch.randn(1, 3, input_size, input_size)
        out = conv(x)
        print(f"PyTorch验证: 输入={input_size} -> 输出={out.shape[2]}")

def test_pooling():
    print("\n" + "=" * 50)
    print("池化层练习")
    print("=" * 50)
    
    print("\n练习: 不同池化方式")
    x = torch.randn(1, 3, 32, 32)
    
    max_pool = nn.MaxPool2d(2, 2)
    avg_pool = nn.AvgPool2d(2, 2)
    
    print(f"输入形状: {x.shape}")
    print(f"最大池化后: {max_pool(x).shape}")
    print(f"平均池化后: {avg_pool(x).shape}")
    
    adaptive_avg = nn.AdaptiveAvgPool2d(1)
    adaptive_max = nn.AdaptiveMaxPool2d(1)
    
    print(f"全局平均池化: {adaptive_avg(x).shape}")
    print(f"全局最大池化: {adaptive_max(x).shape}")

def test_activation_functions():
    print("\n" + "=" * 50)
    print("激活函数练习")
    print("=" * 50)
    
    x = torch.linspace(-3, 3, 10)
    
    print(f"输入: {x}")
    print(f"\nReLU: {F.relu(x)}")
    print(f"\nLeakyReLU: {F.leaky_relu(x, 0.01)}")
    print(f"\nSigmoid: {torch.sigmoid(x)}")
    print(f"\nTanh: {torch.tanh(x)}")
    print(f"\nGELU: {F.gelu(x)}")
    
    print("\n练习: Softmax")
    logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
    probs = F.softmax(logits, dim=1)
    print(f"Logits:\n{logits}")
    print(f"Softmax概率:\n{probs}")
    print(f"概率和: {probs.sum(dim=1)}")

def test_cnn_architecture():
    print("\n" + "=" * 50)
    print("CNN架构练习")
    print("=" * 50)
    
    print("\n练习1: 计算各层输出尺寸")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleCNN()
    x = torch.randn(1, 3, 32, 32)
    
    print(f"输入: {x.shape}")
    
    x1 = model.pool(F.relu(model.conv1(x)))
    print(f"Conv1 + Pool: {x1.shape}")
    
    x2 = model.pool(F.relu(model.conv2(x1)))
    print(f"Conv2 + Pool: {x2.shape}")
    
    x_flat = x2.view(x2.size(0), -1)
    print(f"Flatten: {x_flat.shape}")
    
    print("\n练习2: 计算参数量")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"模型总参数量: {count_parameters(model):,}")
    
    print("\n各层参数量:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")

def test_residual_connection():
    print("\n" + "=" * 50)
    print("残差连接练习")
    print("=" * 50)
    
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
        
        def forward(self, x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity
            out = F.relu(out)
            return out
    
    block = ResidualBlock(64)
    x = torch.randn(1, 64, 32, 32)
    out = block(x)
    
    print(f"残差块输入: {x.shape}")
    print(f"残差块输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")

def test_receptive_field():
    print("\n" + "=" * 50)
    print("感受野计算练习")
    print("=" * 50)
    
    def calc_receptive_field(layers):
        rf = 1
        for kernel_size, stride in layers:
            rf = rf + (kernel_size - 1) * stride
        return rf
    
    print("\n练习: 计算感受野")
    
    vgg_style = [(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2)]
    rf = calc_receptive_field(vgg_style)
    print(f"VGG风格层配置感受野: {rf}")
    
    resnet_style = [(7, 2), (3, 2), (3, 2), (3, 1), (3, 1)]
    rf = calc_receptive_field(resnet_style)
    print(f"ResNet风格层配置感受野: {rf}")

def cnn_design_challenges():
    print("\n" + "=" * 50)
    print("CNN设计挑战")
    print("=" * 50)
    
    print("\n挑战1: 设计一个参数量小于100K的MNIST分类器")
    
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Linear(32 * 7 * 7, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    model = SmallCNN()
    params = sum(p.numel() for p in model.parameters())
    print(f"SmallCNN参数量: {params:,}")
    
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")
    
    print("\n挑战2: 实现一个简单的Inception模块")
    
    class InceptionModule(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            
            self.branch1 = nn.Conv2d(in_channels, out_channels, 1)
            
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            
            self.branch5 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.Conv2d(out_channels, out_channels, 5, padding=2)
            )
            
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        
        def forward(self, x):
            b1 = self.branch1(x)
            b3 = self.branch3(x)
            b5 = self.branch5(x)
            bp = self.branch_pool(x)
            return torch.cat([b1, b3, b5, bp], dim=1)
    
    inception = InceptionModule(64, 32)
    x = torch.randn(1, 64, 32, 32)
    out = inception(x)
    print(f"Inception输入: {x.shape}")
    print(f"Inception输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in inception.parameters()):,}")

if __name__ == "__main__":
    test_conv_calculations()
    test_pooling()
    test_activation_functions()
    test_cnn_architecture()
    test_residual_connection()
    test_receptive_field()
    cnn_design_challenges()
    print("\n" + "=" * 50)
    print("CNN练习完成!")
    print("=" * 50)
