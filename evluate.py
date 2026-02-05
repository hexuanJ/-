import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import argparse
import os


class SimpleJaguarModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.feature_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        x = self.backbone(x)
        features = self.feature_layer(x)
        if return_features:
            return F.normalize(features, p=2, dim=1)
        logits = self.classifier(features)
        return logits, features


class EvalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        self.labels = sorted(self.data['ground_truth'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / row['filename']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_idx = self.label_to_idx[row['ground_truth']]
        return image, label_idx, row['filename'], row['ground_truth']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录（含train.csv和图片）')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    num_classes = checkpoint['num_classes']
    best_acc = checkpoint.get('best_acc', 0)

    print(f"  类别数: {num_classes}")
    print(f"  训练时最佳验证准确率: {best_acc:.2f}%")

    model = SimpleJaguarModel(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    csv_path = os.path.join(args.data_dir, 'train.csv')
    dataset = EvalDataset(csv_path, args.data_dir, transform=transform)
    print(f"\n数据集: {len(dataset)} 张图片, {len(dataset.labels)} 个类别")

    # 划分验证集
    full_df = pd.read_csv(csv_path)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(sss.split(full_df, full_df['ground_truth']))

    val_dataset = Subset(dataset, val_idx.tolist())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    print(f"验证集: {len(val_dataset)} 张图片")

    # 提取特征
    print("\n提取特征 (CPU 模式)...")
    all_features = []
    all_labels = []
    all_preds = []
    all_label_names = []

    with torch.no_grad():
        for images, labels, names, label_names in tqdm(val_loader):
            logits, _ = model(images)
            features = model(images, return_features=True)
            _, preds = logits.max(1)

            all_features.extend(features.numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_label_names.extend(label_names)

    # 计算指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = (all_labels == all_preds).mean() * 100

    # mAP
    features_np = np.array(all_features)
    sim_matrix = np.dot(features_np, features_np.T)

    aps = []
    for i in range(len(all_features)):
        query_label = all_label_names[i]
        y_true = np.array([1 if l == query_label else 0 for l in all_label_names])
        y_true[i] = 0  # 排除自己
        if y_true.sum() == 0:
            continue
        sim_scores = sim_matrix[i].copy()
        sim_scores[i] = -1
        ap = average_precision_score(y_true, sim_scores)
        aps.append(ap)
    mAP = np.mean(aps)

    # Top-1, Top-5
    np.fill_diagonal(sim_matrix, -1)
    top1, top5 = 0, 0
    for i in range(len(all_features)):
        sorted_idx = np.argsort(-sim_matrix[i])
        if all_label_names[sorted_idx[0]] == all_label_names[i]:
            top1 += 1
        if all_label_names[i] in [all_label_names[j] for j in sorted_idx[:5]]:
            top5 += 1

    top1_acc = top1 / len(all_features) * 100
    top5_acc = top5 / len(all_features) * 100

    # 输出结果
    print(f"\n{'='*60}")
    print(f"📊 评估结果")
    print(f"{'='*60}")
    print(f"  分类准确率: {accuracy:.2f}%")
    print(f"  mAP: {mAP:.4f} ({mAP*100:.2f}%)")
    print(f"  Top-1 检索准确率: {top1_acc:.2f}%")
    print(f"  Top-5 检索准确率: {top5_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()