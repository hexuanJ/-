import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import math
from pathlib import Path
import argparse
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViTForJaguarReID, MegaDescriptorWrapper, ImprovedReIDModel, ResNet50ReID
from losses import CircleLoss, SubCenterArcFace, AdaFace


class JaguarDataset(Dataset):
    """
    美洲豹数据集 - 使用 train.csv 格式
    CSV 格式: filename, ground_truth
    """

    def __init__(self, csv_file, img_dir, transform=None, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mode = mode

        # 创建标签映射 (ground_truth -> index)
        self.labels = sorted(self.data['ground_truth'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.labels)

        # 计算每个类别的样本数（用于加权采样）
        self.class_counts = self.data['ground_truth'].value_counts().to_dict()

        print(f"加载数据集: {len(self.data)} 张图片, {self.num_classes} 个类别")
        print(f"类别分布: 最多 {max(self.class_counts.values())} 张, "
              f"最少 {min(self.class_counts.values())} 张")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['filename']
        img_path = self.img_dir / img_name

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label = row['ground_truth']
            label_idx = self.label_to_idx[label]
            return image, label_idx
        else:
            return image, img_name

    def get_sample_weights(self):
        """获取每个样本的权重，用于 WeightedRandomSampler"""
        weights = []
        for idx in range(len(self.data)):
            label = self.data.iloc[idx]['ground_truth']
            # 权重与类别频率成反比
            weight = 1.0 / self.class_counts[label]
            weights.append(weight)
        return weights


class FocalLoss(nn.Module):
    """Focal Loss 处理类别不平衡"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """组合损失: Focal Loss + Center Loss"""

    def __init__(self, num_classes, feat_dim, device,
                 alpha=1.0, gamma=0.5, use_focal=True):
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.gamma = gamma  # Center Loss 权重
        self.device = device

        if use_focal:
            self.cls_loss = FocalLoss(alpha=1, gamma=2)
        else:
            self.cls_loss = nn.CrossEntropyLoss()

        self.center_loss = CenterLoss(num_classes, feat_dim, device)

    def forward(self, outputs, targets, features):
        cls_loss = self.cls_loss(outputs, targets)
        center_loss = self.center_loss(features, targets)

        total_loss = self.alpha * cls_loss + self.gamma * center_loss
        return total_loss, cls_loss, center_loss


class CenterLoss(nn.Module):
    """Center Loss for metric learning"""

    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, x, labels):
        batch_size = x.size(0)

        # 计算特征与中心的距离
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = distmat.addmm(x, self.centers.t(), beta=1, alpha=-2)

        # 只计算对应类别的距离
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def create_data_loaders(data_dir, batch_size=16, val_split=0.2, use_weighted_sampling=True):
    """创建数据加载器"""

    # 数据增强 - 训练集
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # 随机擦除增强鲁棒性
    ])

    # 验证集 - 无增强
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 读取完整数据
    csv_path = os.path.join(data_dir, 'train.csv')
    img_dir = os.path.join(data_dir, 'train')

    full_df = pd.read_csv(csv_path)

    # 分层划分训练/验证集（保持类别比例）
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(sss.split(full_df, full_df['ground_truth']))

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    # 保存临时 CSV 文件
    train_csv_temp = os.path.join(data_dir, 'train_split.csv')
    val_csv_temp = os.path.join(data_dir, 'val_split.csv')
    train_df.to_csv(train_csv_temp, index=False)
    val_df.to_csv(val_csv_temp, index=False)

    # 创建数据集
    train_dataset = JaguarDataset(train_csv_temp, img_dir, transform=train_transform, mode='train')
    val_dataset = JaguarDataset(val_csv_temp, img_dir, transform=val_transform, mode='train')

    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # 加权采样处理类别不平衡
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.num_classes, train_dataset.label_to_idx


_ZERO = torch.tensor(0.0)


def compute_loss(criterion, logits, labels, features):
    """Unified loss computation supporting both CombinedLoss and embedding losses."""
    if isinstance(criterion, (CircleLoss, SubCenterArcFace, AdaFace)):
        # Joint loss: metric loss (embedding-based) + cross-entropy (classifier logits)
        # Equal 1:1 weighting follows standard practice in Re-ID (e.g., Bag of Tricks paper)
        metric_loss = criterion(features, labels)
        cls_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        total = cls_loss + metric_loss
        return total, cls_loss, metric_loss
    result = criterion(logits, labels, features)
    if isinstance(result, tuple):
        # CombinedLoss returns (total, cls, center)
        return result
    return result, result, _ZERO


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass – all models accept optional 'labels' kwarg
        import inspect
        sig = inspect.signature(model.forward)
        if 'labels' in sig.parameters:
            logits, features = model(images, labels=labels)
        else:
            logits, features = model(images)

        # 计算损失
        total_loss, cls_loss, _ = compute_loss(criterion, logits, labels, features)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        running_loss += total_loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {total_loss.item():.4f} '
                  f'(cls: {cls_loss.item():.4f}) '
                  f'Acc: {100. * correct / total:.2f}%')

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, features = model(images)
            total_loss, _, _ = compute_loss(criterion, logits, labels, features)

            running_loss += total_loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def check_gpu():
    """检查 GPU 状态"""
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        return True
    else:
        print("❌ CUDA 不可用")
        return False


def main():
    parser = argparse.ArgumentParser(description='Jaguar Re-ID Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录 (包含 train.csv 和 train/ 文件夹)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Head 学习率（backbone 自动为 1/10）')
    parser.add_argument('--model', type=str, default='vit',
                        choices=['vit', 'mega', 'improved', 'resnet50'],
                        help='模型选择: vit | mega | improved | resnet50')
    parser.add_argument('--backbone', type=str,
                        default='swin_base_patch4_window12_384',
                        help='improved 模型使用的 timm backbone 名称')
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['combined', 'circle', 'subcenter_arcface', 'adaface'],
                        help='损失函数: combined(Focal+Center) | circle | subcenter_arcface | adaface')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_focal_loss', action='store_true', default=True)
    parser.add_argument('--use_weighted_sampling', action='store_true', default=True)

    args = parser.parse_args()

    # GPU 检查
    has_gpu = check_gpu()
    device = torch.device('cuda' if has_gpu else 'cpu')

    if has_gpu:
        torch.backends.cudnn.benchmark = True

    print(f'使用设备: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    # 创建数据加载器
    train_loader, val_loader, num_classes, label_to_idx = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        use_weighted_sampling=args.use_weighted_sampling
    )
    print(f'类别数量: {num_classes}')

    # 创建模型
    if args.model == 'vit':
        model = ViTForJaguarReID(
            num_classes=num_classes,
            image_size=384,
            patch_size=32,
            dim=1024,
            depth=12,
            heads=16,
            mlp_dim=2048
        )
    elif args.model == 'resnet50':
        print('使用 ResNet50ReID (torchvision, Last Stride=1, GeM, BNNeck)')
        model = ResNet50ReID(num_classes=num_classes, feat_dim=512)
    elif args.model == 'improved':
        print(f'使用预训练 backbone: {args.backbone}')
        model = ImprovedReIDModel(
            num_classes=num_classes,
            backbone_name=args.backbone,
            feat_dim=512,
            pretrained=True
        )
    else:
        model = MegaDescriptorWrapper(num_classes=num_classes, model_size='base')

    model = model.to(device)

    # 损失函数
    feat_dim = 512
    if args.loss == 'circle':
        criterion = CircleLoss(in_features=feat_dim, num_classes=num_classes)
        print('使用 Circle Loss (CVPR 2020)')
    elif args.loss == 'subcenter_arcface':
        criterion = SubCenterArcFace(in_features=feat_dim, num_classes=num_classes)
        print('使用 SubCenter ArcFace (ECCV 2020)')
    elif args.loss == 'adaface':
        criterion = AdaFace(in_features=feat_dim, num_classes=num_classes)
        print('使用 AdaFace (CVPR 2022)')
    else:
        criterion = CombinedLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            device=device,
            use_focal=args.use_focal_loss
        )
        print('使用 Focal + Center Loss (combined)')
    criterion = criterion.to(device)

    # 差分学习率：backbone 用小学习率，head 用大学习率
    if args.model == 'resnet50':
        backbone_params = model.get_backbone_parameters()
        head_params = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr},
            {'params': criterion.parameters(), 'lr': args.lr},
        ], weight_decay=0.01)
        print(f'差分学习率: backbone={args.lr * 0.1:.2e}, head={args.lr:.2e}')
    elif args.model == 'improved' and hasattr(model, 'backbone'):
        backbone_params = list(model.backbone.parameters())
        head_params = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr},
            {'params': criterion.parameters(), 'lr': args.lr},
        ], weight_decay=0.01)
        print(f'差分学习率: backbone={args.lr * 0.1:.2e}, head={args.lr:.2e}')
    else:
        optimizer = optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=args.lr, weight_decay=0.01
        )

    # Cosine Annealing with Linear Warmup
    warmup_epochs = max(5, args.epochs // 20)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 恢复训练
    start_epoch = 0
    best_acc = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"加载检查点: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f'{"=" * 60}')

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scheduler
        )

        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f'\n📊 Epoch {epoch + 1} 结果:')
        print(f'   训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'   验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'label_to_idx': label_to_idx
            }, save_path)
            print(f'✅ 保存最佳模型，准确率: {best_acc:.2f}%')

        if has_gpu:
            print(f'   GPU 显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB')

    print(f'\n🎉 训练完成! 最佳验证准确率: {best_acc:.2f}%')


if __name__ == '__main__':
    main()