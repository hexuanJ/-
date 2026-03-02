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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reranking import re_ranking


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


def build_tta_transforms(use_tta):
    """Return a list of transforms for TTA (or just the base transform)."""
    base = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if not use_tta:
        return [base]
    return [
        base,
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录（含train.csv和图片）')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_reranking', action='store_true', default=False,
                        help='使用 k-reciprocal re-ranking 后处理')
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='使用多尺度+翻转 Test Time Augmentation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='推理设备: cpu 或 cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
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
    model = model.to(device)
    model.eval()

    # 加载数据（基础 transform，不做 TTA）
    base_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    csv_path = os.path.join(args.data_dir, 'train.csv')
    dataset = EvalDataset(csv_path, args.data_dir, transform=base_transform)
    print(f"\n数据集: {len(dataset)} 张图片, {len(dataset.labels)} 个类别")

    # 划分验证集
    full_df = pd.read_csv(csv_path)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(sss.split(full_df, full_df['ground_truth']))
    val_idx_list = val_idx.tolist()

    print(f"验证集: {len(val_idx_list)} 张图片")

    # Build list of (image_path, label_idx, filename, label_name) for val set
    val_rows = [dataset.data.iloc[i] for i in val_idx_list]

    # TTA: extract features for each transform and accumulate
    tta_transforms = build_tta_transforms(args.use_tta)
    feat_accum = None
    all_labels = []
    all_preds = []
    all_label_names = []

    for t_idx, transform in enumerate(tta_transforms):
        dataset_t = EvalDataset(csv_path, args.data_dir, transform=transform)
        val_dataset_t = Subset(dataset_t, val_idx_list)
        val_loader = DataLoader(val_dataset_t, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        feats_t = []
        labels_t = []
        preds_t = []
        names_t = []

        with torch.no_grad():
            for images, labels, names, label_names in tqdm(
                    val_loader,
                    desc=f"提取特征 (TTA {t_idx+1}/{len(tta_transforms)})",
                    leave=False):
                images = images.to(device)
                logits, _ = model(images)
                features = model(images, return_features=True)
                _, preds = logits.max(1)
                feats_t.extend(features.cpu().numpy())
                if t_idx == 0:
                    labels_t.extend(labels.numpy())
                    preds_t.extend(preds.cpu().numpy())
                    names_t.extend(label_names)

        feats_arr = np.array(feats_t)
        if feat_accum is None:
            feat_accum = feats_arr
            all_labels = np.array(labels_t)
            all_preds = np.array(preds_t)
            all_label_names = names_t
        else:
            feat_accum = feat_accum + feats_arr

    # L2-normalise averaged features
    norms = np.linalg.norm(feat_accum, axis=1, keepdims=True) + 1e-8
    features_np = feat_accum / norms

    # 计算分类准确率
    accuracy = (all_labels == all_preds).mean() * 100

    # mAP & Top-k with optional re-ranking
    if args.use_reranking:
        print("\n执行 Re-Ranking...")
        dist_matrix = re_ranking(features_np, features_np, k1=20, k2=6, lambda_value=0.3)
        max_dist = dist_matrix.max() + 1e-8
        sim_matrix = 1.0 - dist_matrix / max_dist
    else:
        sim_matrix = np.dot(features_np, features_np.T)

    aps = []
    for i in range(len(features_np)):
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
    sim_diag = sim_matrix.copy()
    np.fill_diagonal(sim_diag, -1)
    top1, top5 = 0, 0
    for i in range(len(features_np)):
        sorted_idx = np.argsort(-sim_diag[i])
        if all_label_names[sorted_idx[0]] == all_label_names[i]:
            top1 += 1
        if all_label_names[i] in [all_label_names[j] for j in sorted_idx[:5]]:
            top5 += 1

    top1_acc = top1 / len(features_np) * 100
    top5_acc = top5 / len(features_np) * 100

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