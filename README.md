# 🐆 美洲豹个体识别系统 (Jaguar Re-ID)

基于深度学习的美洲豹个体重识别项目，支持预训练强 Backbone、现代 Loss 函数、Re-Ranking 后处理和多尺度 TTA 推理。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 目录

- [项目简介](#项目简介)
- [设备配置](#设备配置)
- [环境配置](#环境配置)
- [数据集结构](#数据集结构)
- [快速开始](#快速开始)
- [训练模型](#训练模型)
- [评估模型](#评估模型)
- [生成提交文件](#生成提交文件)
- [实验结果](#实验结果)
- [项目结构](#项目结构)

---

## 📖 项目简介

本项目旨在解决美洲豹个体识别问题。通过分析美洲豹身上独特的斑纹模式，实现对不同个体的区分和识别。

### 主要特点

- ✅ 使用 ImageNet 预训练权重进行迁移学习
- ✅ 支持多种强 Backbone：Swin Transformer、ConvNeXt、ViT（通过 timm）
- ✅ GeM 广义均值池化 + BNNeck (Bag of Tricks)
- ✅ 现代 Loss 函数：Circle Loss、SubCenter ArcFace、AdaFace
- ✅ k-Reciprocal Re-Ranking 后处理提升 mAP
- ✅ 多尺度 + 翻转 Test Time Augmentation (TTA)
- ✅ 差分学习率 + Cosine Annealing with Warmup
- ✅ 加权采样处理类别不平衡问题

### 技术架构

```
输入图片 (384×384)
    │
    ▼
┌─────────────────────────────────┐
│  预训练 Backbone (timm)          │  ← Swin / ConvNeXt / ViT / EfficientNet
│  + GeM Pooling                  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│  投影层 (512维)  │  ← FC
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  BNNeck         │  ← BatchNorm (Bag of Tricks 2019)
└─────────────────┘
    │
    ▼
┌─────────────────┐         ┌─────────────────────┐
│ L2归一化特征     │         │ 分类 Logits          │
│ (检索/Re-Rank)  │         │ (训练时计算 Loss)     │
└─────────────────┘         └─────────────────────┘
```

---
## 🏋️ 设备配置
### 云服务器
云服务器种类，很多这里我们使用OpenI云服务平台，进行模型训练。
### 📁 数据集结构与上传
```
dataset/
├── train/
│   ├── train.csv          # 训练标注文件
│   ├── train_0001.png     # 训练图片
│   ├── train_0002.png
│   └── ...
├── test/
│   ├── test_0001.png      # 测试图片
│   ├── test_0002.png
│   └── ...
└── test.csv               # 测试配对文件
```

### train.csv 格式

| filename | ground_truth |
|----------|--------------|
| train_0001.png | Abril |
| train_0002.png | Abril |
| train_0003.png | Bagua |

### test.csv 格式

| row_id | image1 | image2 |
|--------|--------|--------|
| 0 | test_0001.png | test_0002.png |
| 1 | test_0001.png | test_0003.png |

---
上述数据集train（1895张、13.5GB）、test（371张、2.5GB），数据内存占比高，传输服务器困难，给出如下特殊传输方式：
### SDK传输
```bash
import openi  #OpenI中python的SDK传输
import os

# 1. 登录到OpenI平台（需要获取访问令牌）
token = "xxx"  # 从平台中获取token（设置）
openi.login(token=token)  # 使用 token 登录平台

# 2. 验证登录状态
user_info = openi.whoami() # 获取当前登录用户信息
# 3. 设置数据集仓库信息
repo_id = "xxx/xxxx"  # 格式: 用户名/仓库名
local_dataset_path = "xxx\\xxx\\xxx"  # 本地数据集路径

# 4. 上传数据集
    result = openi.openi_upload_file(
        repo_id=repo_id,
        file_or_folder_path=local_dataset_path,
        repo_type="dataset"  # 指定为数据集类型)
```
## 🔧 环境配置

### 依赖安装

```bash
# 创建虚拟环境（可选）
conda create -n jaguar python=3.10
conda activate jaguar

# 安装依赖
pip install torch torchvision
pip install pandas numpy pillow
pip install scikit-learn tqdm
pip install timm einops  # backbone 和 ViT 组件
```
### 快捷安装
```bash
# 使用清华源安装 requirements.txt 中的所有依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 无 (可用CPU) | NVIDIA V100 32GB |
| 内存 | 8GB | 16GB+ |
| 硬盘 | 5GB | 10GB+ |

---



## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/jaguar-reid.git
cd jaguar-reid
```

### 2. 训练模型（推荐：improved + AdaFace）

```bash
python train.py \
    --data_dir /path/to/dataset/train \
    --model improved \
    --backbone swin_base_patch4_window12_384 \
    --loss adaface \
    --batch_size 16 \
    --epochs 50
```

### 3. 评估模型（含 Re-Ranking + TTA）

```bash
python evluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir /path/to/dataset/train \
    --use_reranking \
    --use_tta
```

### 4. 生成提交文件（含 Re-Ranking + TTA）

```bash
python test.py \
    --model_path checkpoints/best_model.pth \
    --test_csv test.csv \
    --test_img_dir /path/to/test \
    --output submission.csv \
    --use_reranking \
    --use_tta
```

---

## 🏋️ 训练模型

### 模型选项 (`--model`)

| 选项 | 说明 |
|------|------|
| `vit` | 自建 ViT（从零训练，小数据集效果一般） |
| `mega` | EfficientNet 预训练 backbone |
| `improved` | **推荐**：timm 预训练强 backbone + GeM + BNNeck |

### 推荐 backbone (`--backbone`，仅 `--model improved` 时有效)

| backbone 名称 | 特点 |
|--------------|------|
| `swin_base_patch4_window12_384` | 强性能，需要较大显存 |
| `convnext_base_384_in22ft1k` | 收敛快，效果强 |
| `vit_base_patch16_384` | ViT 预训练版，效果好 |
| `tf_efficientnet_b4` | 轻量，适合显存不足时 |

### 损失函数选项 (`--loss`)

| 选项 | 方法 | 推荐度 |
|------|------|--------|
| `combined` | Focal Loss + Center Loss（原始） | 普通 |
| `circle` | Circle Loss (CVPR 2020) | 好 |
| `subcenter_arcface` | SubCenter ArcFace (ECCV 2020) | 好 |
| `adaface` | AdaFace (CVPR 2022) | **最推荐**，自适应 margin |

### 完整训练示例

```bash
# 推荐配置：Swin + AdaFace + 差分学习率
python train.py \
    --data_dir /tmp/dataset/train \
    --model improved \
    --backbone swin_base_patch4_window12_384 \
    --loss adaface \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-3

# 轻量配置（显存不足时）
python train.py \
    --data_dir /tmp/dataset/train \
    --model improved \
    --backbone tf_efficientnet_b4 \
    --loss circle \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3

# 传统配置（保持向后兼容）
python train.py \
    --data_dir /tmp/dataset/train \
    --model vit \
    --loss combined \
    --batch_size 16 \
    --epochs 100
```

### 训练策略说明

- **差分学习率**：`--model improved` 时，backbone 学习率为 `lr × 0.1`，head 学习率为 `lr`
- **Warmup + Cosine Annealing**：前 5% epoch 线性 warmup，之后 Cosine 衰减

---

## 📊 评估模型

### 基本用法

```bash
python evluate.py --model_path checkpoints/best_model.pth --data_dir /tmp/dataset/train
```

### 高级选项

| 参数 | 说明 |
|------|------|
| `--use_reranking` | 启用 k-reciprocal Re-Ranking（推荐，免费提升 mAP） |
| `--use_tta` | 启用多尺度+翻转 TTA（4x 推理，提升稳定性） |
| `--device` | 推理设备：`cpu` 或 `cuda` |

```bash
# 完整评估（最高性能）
python evluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir /tmp/dataset/train \
    --use_reranking \
    --use_tta \
    --device cuda
```

### 评估指标

- **分类准确率**: 正确分类的样本比例
- **mAP**: 平均精度均值，衡量检索效果
- **Top-1 准确率**: 最相似样本是同一个体的比例
- **Top-5 准确率**: 前5个最相似样本包含同一个体的比例

---

## 📤 生成提交文件

```bash
python test.py \
    --model_path checkpoints/best_model.pth \
    --test_csv test.csv \
    --test_img_dir /tmp/dataset/test \
    --output submission.csv \
    --use_reranking \
    --use_tta
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | best_model.pth | 模型权重路径 |
| `--test_csv` | 必填 | 测试配对 CSV |
| `--test_img_dir` | 必填 | 测试图片目录 |
| `--output` | submission.csv | 输出文件 |
| `--use_reranking` | False | 启用 Re-Ranking |
| `--use_tta` | False | 启用 TTA |
| `--device` | cpu | 推理设备 |

---


## 📂 项目结构

```
jaguar-reid/
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖列表
├── model.py               # 模型定义（ViT / MegaDescriptor / ImprovedReIDModel）
├── losses.py              # 现代 Loss（CircleLoss / SubCenterArcFace / AdaFace）
├── reranking.py           # k-Reciprocal Re-Ranking (CVPR 2017)
├── train.py               # 训练脚本
├── test.py                # 测试脚本（生成提交文件）
├── evluate.py             # 评估脚本（mAP / Top-1 / Top-5）
├── shangchuan.py          # 数据上传工具
├── checkpoints/           # 模型保存目录
│   └── best_model.pth
└── submission.csv         # 提交文件
```

---

## 🔍 常见问题

### Q1: 显存不足怎么办？

使用更小的 backbone 或减小 batch size：
```bash
python train.py \
    --data_dir /tmp/dataset/train \
    --model improved \
    --backbone tf_efficientnet_b4 \
    --batch_size 8
```

### Q2: Re-Ranking 很慢怎么办？

Re-Ranking 时间复杂度为 O(N²)，建议只在测试时开启。在 CPU 上 200 张图片约需几秒。

### Q3: TTA 对结果影响大吗？

TTA 会进行 4× 推理（原图 + 翻转 + 两种缩放），通常可以提升 1-2% mAP。

### Q4: 路径错误 FileNotFoundError？

**检查数据目录结构**:
```bash
ls -la /tmp/dataset/train/
head /tmp/dataset/train/train.csv
```

---

## 📜 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [PyTorch](https://pytorch.org/)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [Circle Loss (CVPR 2020)](https://arxiv.org/abs/2002.10857)
- [SubCenter ArcFace (ECCV 2020)](https://arxiv.org/abs/2004.01159)
- [AdaFace (CVPR 2022)](https://arxiv.org/abs/2204.00964)
- [Re-Ranking (CVPR 2017)](https://arxiv.org/abs/1701.08398)
- [OpenI 平台](https://openi.pcl.ac.cn/)

---

## 📧 联系方式

如有问题，请提交 Issue 或联系：

- 邮箱: 17763221787@163.com


---

<p align="center">
  <b>⭐ 如果这个项目对你有帮助，请给个 Star！⭐</b>
</p>
