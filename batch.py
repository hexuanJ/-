"""
batch.py - 批量特征提取和提交文件生成

作用：
    使用 DataLoader 批量处理测试图片，比 test.py 快 3-5 倍

何时使用：
    1. 训练完成后，生成最终的 submission.csv
    2. 当测试图片数量大（371张）时，批处理更高效

与 test.py 的区别：
    - test.py: 逐张处理，代码简单，适合调试
    - batch.py: 批量处理，速度快，适合最终提交

用法：
    python batch.py --model_path checkpoints/best_model.pth --data_dir /path/to/data --output submission.csv
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViTForJaguarReID, MegaDescriptorWrapper
from torchvision import transforms


class TestJaguarDataset(Dataset):
    """测试集数据集"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(
            [f for f in self.image_dir.iterdir()
             if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        self.transform = transform
        self.image_names = [f.name for f in self.image_files]

        print(f"加载测试图片: {len(self.image_files)} 张")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_names[idx]


def batch_extract_features(model, test_loader, device):
    """
    批量提取所有测试图片的特征

    Returns:
        features_dict: {图片名: 特征向量}
    """
    model.eval()
    features_dict = {}

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="批量提取特征"):
            images = images.to(device, non_blocking=True)

            # 提取特征
            features = model(images, return_features=True)

            # L2 归一化（对于余弦相似度很重要）
            features = F.normalize(features, p=2, dim=1)

            for i, name in enumerate(names):
                features_dict[name] = features[i].cpu().numpy()

    return features_dict


def compute_all_similarities(features_dict, test_df):
    """
    计算所有图片对的相似度

    Args:
        features_dict: {图片名: 特征向量}
        test_df: 包含 query_image, gallery_image 的 DataFrame

    Returns:
        similarities: 相似度列表
    """
    similarities = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="计算相似度"):
        query = row['query_image']
        gallery = row['gallery_image']

        if query in features_dict and gallery in features_dict:
            feat1 = torch.from_numpy(features_dict[query])
            feat2 = torch.from_numpy(features_dict[gallery])

            # 余弦相似度 (因为已经 L2 归一化，直接点积即可)
            similarity = torch.dot(feat1, feat2).item()

            # 映射到 [0, 1] 范围（余弦相似度原本是 [-1, 1]）
            similarity = (similarity + 1) / 2

            # 确保在有效范围内
            similarity = max(0.0, min(1.0, similarity))
        else:
            print(f"警告: 缺少特征 - {query} 或 {gallery}")
            similarity = 0.5  # 默认值

        similarities.append(similarity)

    return similarities


def validate_submission(submission):
    """验证提交文件格式"""
    errors = []

    if len(submission) != 137270:
        errors.append(f"行数错误: {len(submission)}, 应为 137270")

    if list(submission.columns) != ['row_id', 'similarity']:
        errors.append(f"列名错误: {list(submission.columns)}")

    if submission['row_id'].tolist() != list(range(137270)):
        errors.append("row_id 顺序错误")

    if (submission['similarity'] < 0).any():
        errors.append(f"存在相似度 < 0")

    if (submission['similarity'] > 1).any():
        errors.append(f"存在相似度 > 1")

    if submission['similarity'].isna().any():
        errors.append(f"存在 NaN 值")

    if errors:
        print("❌ 验证失败:")
        for e in errors:
            print(f"   - {e}")
        return False
    else:
        print("✅ 提交文件格式验证通过")
        return True


def main():
    parser = argparse.ArgumentParser(description='Jaguar Re-ID 批量推理')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='模型路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录 (包含 test.csv 和 test/ 文件夹)')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='输出文件名')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    args = parser.parse_args()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    num_classes = checkpoint.get('num_classes', 31)

    model = ViTForJaguarReID(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"模型类别数: {num_classes}")

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建测试数据集和加载器
    test_dir = os.path.join(args.data_dir, 'test')
    test_dataset = TestJaguarDataset(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 不打乱顺序
        num_workers=4,
        pin_memory=True
    )

    # 批量提取特征
    print("\n开始批量提取特征...")
    features_dict = batch_extract_features(model, test_loader, device)
    print(f"提取完成: {len(features_dict)} 张图片")

    # 加载测试对
    test_csv = os.path.join(args.data_dir, 'test.csv')
    test_df = pd.read_csv(test_csv)
    print(f"\n测试对数: {len(test_df)}")

    # 计算相似度
    print("\n计算所有图片对的相似度...")
    similarities = compute_all_similarities(features_dict, test_df)

    # 创建提交文件
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'similarity': similarities
    })

    # 验证格式
    print("\n验证提交文件格式...")
    validate_submission(submission)

    # 保存
    submission.to_csv(args.output, index=False)

    # 打印统计信息
    print(f"\n✅ 提交文件已保存: {args.output}")
    print(f"\n📊 统计信息:")
    print(f"   总行数: {len(submission)}")
    print(f"   相似度范围: [{submission['similarity'].min():.6f}, {submission['similarity'].max():.6f}]")
    print(f"   相似度均值: {submission['similarity'].mean():.6f}")
    print(f"   相似度标准差: {submission['similarity'].std():.6f}")

    # 分布信息
    print(f"\n📈 相似度分布:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins) - 1):
        count = ((submission['similarity'] >= bins[i]) &
                 (submission['similarity'] < bins[i + 1])).sum()
        pct = count / len(submission) * 100
        print(f"   [{bins[i]:.1f}, {bins[i + 1]:.1f}): {count:,} ({pct:.1f}%)")


if __name__ == '__main__':
    main()