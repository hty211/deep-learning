import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def test_basic_transforms():
    print("=" * 50)
    print("基础数据增强练习")
    print("=" * 50)
    
    np.random.seed(42)
    img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    print(f"原始图像尺寸: {img.size}")
    
    print("\n1. 随机裁剪")
    transform = transforms.RandomCrop(80)
    cropped = transform(img)
    print(f"裁剪后尺寸: {cropped.size}")
    
    print("\n2. 随机水平翻转")
    transform = transforms.RandomHorizontalFlip(p=1.0)
    flipped = transform(img)
    print(f"翻转后尺寸: {flipped.size}")
    
    print("\n3. 随机旋转")
    transform = transforms.RandomRotation(30)
    rotated = transform(img)
    print(f"旋转后尺寸: {rotated.size}")
    
    print("\n4. 颜色抖动")
    transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
    jittered = transform(img)
    print(f"颜色抖动后尺寸: {jittered.size}")
    
    print("\n5. 随机调整大小裁剪")
    transform = transforms.RandomResizedCrop(80, scale=(0.8, 1.0))
    resized = transform(img)
    print(f"调整大小裁剪后尺寸: {resized.size}")

def test_composed_transforms():
    print("\n" + "=" * 50)
    print("组合数据增强练习")
    print("=" * 50)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("训练数据增强:")
    for t in train_transform.transforms:
        print(f"  - {t.__class__.__name__}")
    
    print("\n测试数据增强:")
    for t in test_transform.transforms:
        print(f"  - {t.__class__.__name__}")

class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
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
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        batch_size, c, h, w = images.size()
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(h, w, lam)
        
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        
        return images, labels, labels[index], lam
    
    def _rand_bbox(self, h, w, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        return bbx1, bby1, bbx2, bby2

def test_advanced_augmentation():
    print("\n" + "=" * 50)
    print("高级数据增强练习")
    print("=" * 50)
    
    print("\n1. Cutout测试")
    cutout = Cutout(n_holes=1, length=16)
    img = torch.randn(3, 32, 32)
    img_cutout = cutout(img)
    print(f"Cutout输入: {img.shape}")
    print(f"Cutout输出: {img_cutout.shape}")
    
    print("\n2. Mixup测试")
    mixup = Mixup(alpha=0.2)
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    mixed, la, lb, lam = mixup(images, labels)
    print(f"Mixup混合图像: {mixed.shape}")
    print(f"混合系数: {lam:.4f}")
    
    print("\n3. CutMix测试")
    cutmix = CutMix(alpha=1.0)
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    mixed, la, lb, lam = cutmix(images, labels)
    print(f"CutMix混合图像: {mixed.shape}")
    print(f"混合系数: {lam:.4f}")

def visualize_augmentations():
    print("\n" + "=" * 50)
    print("数据增强可视化")
    print("=" * 50)
    
    np.random.seed(42)
    img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    transforms_list = [
        ('原始图像', transforms.Compose([])),
        ('随机裁剪', transforms.RandomCrop(80)),
        ('水平翻转', transforms.RandomHorizontalFlip(p=1.0)),
        ('随机旋转', transforms.RandomRotation(30)),
        ('颜色抖动', transforms.ColorJitter(brightness=0.3, contrast=0.3)),
    ]
    
    fig, axes = plt.subplots(1, len(transforms_list), figsize=(15, 3))
    
    for ax, (name, transform) in zip(axes, transforms_list):
        if name == '原始图像':
            augmented = img
        else:
            augmented = transform(img)
        
        ax.imshow(augmented)
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150)
    print("增强示例已保存为 augmentation_examples.png")
    plt.show()

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def test_mixup_training():
    print("\n" + "=" * 50)
    print("Mixup训练示例")
    print("=" * 50)
    
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    mixup = Mixup(alpha=0.2)
    
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    
    model.train()
    optimizer.zero_grad()
    
    mixed_images, labels_a, labels_b, lam = mixup(images, labels)
    
    outputs = model(mixed_images)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    
    loss.backward()
    optimizer.step()
    
    print(f"混合系数: {lam:.4f}")
    print(f"损失值: {loss.item():.4f}")

def augmentation_comparison():
    print("\n" + "=" * 50)
    print("数据增强方法对比")
    print("=" * 50)
    
    methods = {
        'RandomCrop': '随机裁剪，增加位置不变性',
        'RandomHorizontalFlip': '随机水平翻转，增加方向不变性',
        'RandomRotation': '随机旋转，增加旋转不变性',
        'ColorJitter': '颜色抖动，增加颜色鲁棒性',
        'Cutout': '随机遮挡，增加局部特征学习',
        'Mixup': '样本混合，增加决策边界平滑',
        'CutMix': '区域混合，结合Cutout和Mixup优点',
    }
    
    print("\n数据增强方法及作用:")
    for method, description in methods.items():
        print(f"  {method}: {description}")
    
    print("\n推荐组合:")
    print("  轻量级: RandomCrop + RandomHorizontalFlip")
    print("  标准级: RandomCrop + RandomHorizontalFlip + ColorJitter")
    print("  重量级: RandomCrop + RandomHorizontalFlip + ColorJitter + Cutout/Mixup")

if __name__ == "__main__":
    test_basic_transforms()
    test_composed_transforms()
    test_advanced_augmentation()
    visualize_augmentations()
    test_mixup_training()
    augmentation_comparison()
    print("\n" + "=" * 50)
    print("数据增强练习完成!")
    print("=" * 50)
