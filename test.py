import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reranking import re_ranking


class SimpleJaguarModel(nn.Module):
    """与训练时完全相同的模型结构"""

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


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_list, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_files = image_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


def extract_features_tta(model, dataset, image_dir, image_list, device, batch_size,
                         use_tta=False):
    """Extract features with optional Test Time Augmentation (TTA).

    TTA: average over original + horizontal-flip + two additional scales.
    """
    base_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not use_tta:
        transforms_list = [base_transform]
    else:
        transforms_list = [
            base_transform,
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

    features_accum = {}  # name -> accumulated feature
    for t_idx, transform in enumerate(transforms_list):
        ds = ImageDataset(image_dir, image_list, transform=transform)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for images, names in tqdm(dl,
                                      desc=f"提取特征 (TTA {t_idx+1}/{len(transforms_list)})",
                                      leave=False):
                images = images.to(device)
                feats = model(images, return_features=True)
                for i, name in enumerate(names):
                    feat_np = feats[i].cpu().numpy()
                    if name in features_accum:
                        features_accum[name] += feat_np
                    else:
                        features_accum[name] = feat_np

    # L2-normalise the averaged features
    features_dict = {}
    for name, feat in features_accum.items():
        norm = np.linalg.norm(feat) + 1e-8
        features_dict[name] = feat / norm
    return features_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--test_csv', type=str, required=True, help='test.csv 路径')
    parser.add_argument('--test_img_dir', type=str, required=True, help='测试图片目录')
    parser.add_argument('--output', type=str, default='submission.csv')
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
    print(f"  最佳验证准确率: {best_acc:.2f}%")

    model = SimpleJaguarModel(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 读取 test.csv
    print(f"\n读取测试文件: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    print(f"测试配对数: {len(test_df)}")
    print(f"列名: {test_df.columns.tolist()}")
    print(f"前3行:\n{test_df.head(3)}")

    # 确定列名
    cols = test_df.columns.tolist()
    if 'query_image' in cols:
        col1, col2 = 'query_image', 'gallery_image'
    elif 'image1' in cols:
        col1, col2 = 'image1', 'image2'
    elif 'img1' in cols:
        col1, col2 = 'img1', 'img2'
    else:
        col1, col2 = cols[1], cols[2]

    print(f"使用列: {col1}, {col2}")

    # 获取所有需要处理的图片
    all_images = list(set(test_df[col1].tolist() + test_df[col2].tolist()))
    print(f"需要处理的图片数: {len(all_images)}")

    # 提取特征
    print(f"\n提取特征 (TTA={'开启' if args.use_tta else '关闭'})...")
    features_dict = extract_features_tta(
        model, None, args.test_img_dir, all_images, device,
        args.batch_size, use_tta=args.use_tta
    )
    print(f"提取完成: {len(features_dict)} 张")

    # 构建相似度矩阵（可选 re-ranking）
    if args.use_reranking:
        print("\n执行 Re-Ranking...")
        feat_matrix = np.array([features_dict[n] for n in all_images])
        dist_matrix = re_ranking(feat_matrix, feat_matrix, k1=20, k2=6, lambda_value=0.3)
        # Convert distance to similarity
        max_dist = dist_matrix.max() + 1e-8
        sim_matrix_full = 1.0 - dist_matrix / max_dist
        img_to_idx = {name: idx for idx, name in enumerate(all_images)}
    else:
        img_to_idx = None
        sim_matrix_full = None

    # 计算相似度
    print("\n计算相似度...")
    results = []
    missing = 0

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="计算相似度"):
        img1 = row[col1]
        img2 = row[col2]

        if img1 in features_dict and img2 in features_dict:
            if args.use_reranking:
                i1 = img_to_idx[img1]
                i2 = img_to_idx[img2]
                sim = float(sim_matrix_full[i1, i2])
            else:
                f1 = features_dict[img1]
                f2 = features_dict[img2]
                sim = float(np.dot(f1, f2))  # already L2-normalised
                sim = (sim + 1) / 2  # 映射到 [0, 1]
        else:
            sim = 0.5
            missing += 1

        if 'row_id' in row:
            row_id = row['row_id']
        elif 'id' in row:
            row_id = row['id']
        else:
            row_id = idx

        results.append({
            'row_id': row_id,
            'similarity': float(sim)
        })

    if missing > 0:
        print(f"⚠️ 缺失图片的配对: {missing}")

    # 保存结果
    submission = pd.DataFrame(results)
    submission.to_csv(args.output, index=False)

    sims = [r['similarity'] for r in results]
    print(f"\n{'=' * 50}")
    print(f"✅ 提交文件已保存: {args.output}")
    print(f"{'=' * 50}")
    print(f"  总配对数: {len(results)}")
    print(f"  相似度范围: [{min(sims):.4f}, {max(sims):.4f}]")
    print(f"  相似度均值: {np.mean(sims):.4f}")
    print(f"  相似度中位数: {np.median(sims):.4f}")
    print(f"{'=' * 50}")
    print(f"\n前5行预测结果:")
    print(submission.head())


if __name__ == '__main__':
    main()