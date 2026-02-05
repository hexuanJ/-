# 🐆 美洲豹个体识别系统 (Jaguar Re-ID)

基于深度学习的美洲豹个体重识别项目，使用预训练 ResNet50/ViT 进行迁移学习。

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
- ✅ 支持 ResNet50 和 ViT 两种骨干网络
- ✅ 加权采样处理类别不平衡问题
- ✅ 支持 GPU/CPU 训练和推理

### 技术架构

```
输入图片 (224x224)
    │
    ▼
┌─────────────────┐
│  预训练 Backbone │  ← ResNet50 / ViT
│  (特征提取)      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  特征层 (512维)  │  ← FC + BN + ReLU + Dropout
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  分类器 (31类)   │  ← 美洲豹个体类别
└─────────────────┘
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
pip install timm  # 如果使用 ViT
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

### 2. 训练模型

```bash
python train.py --data_dir /path/to/dataset/train --batch_size 64 --epochs 50
```

### 3. 评估模型

```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir /path/to/dataset/train
```

### 4. 生成提交文件

```bash
python test.py --model_path checkpoints/best_model.pth --test_csv test.csv --test_img_dir /path/to/test
```

---

## 🏋️ 训练模型

### 基本用法

```bash
python train.py --data_dir /tmp/dataset/train --batch_size 64 --epochs 50 --lr 0.001
```

### 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必填 | 训练数据目录 |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--val_split` | 0.1 | 验证集比例 |
| `--save_dir` | checkpoints | 模型保存目录 |

### 训练示例

```bash
# 使用 ResNet50 (推荐)
python train.py \
    --data_dir /tmp/dataset/train \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.005

# 使用较小学习率防止过拟合
python train.py \
    --data_dir /tmp/dataset/train \
    --batch_size 32 \
    --epochs 30 \
    --lr 0.001
```

### 训练输出示例

```
✅ CUDA 可用
   GPU: Tesla V100S-PCIE-32GB
   显存: 31.73 GB
使用设备: cuda
加载数据集: 1895 张图片, 31 个类别
类别分布: 最多 183 张, 最少 13 张
训练集: 1705 张, 验证集: 190 张

============================================================
Epoch 1/50
============================================================
  Batch [0/26] Loss: 3.4841 Acc: 3.12%
  Batch [10/26] Loss: 2.1532 Acc: 45.23%
  Batch [20/26] Loss: 1.4521 Acc: 68.45%

📊 Epoch 1 结果:
   训练 - Loss: 1.8234, Acc: 52.34%
   验证 - Loss: 1.2341, Acc: 65.26%
✅ 保存最佳模型，准确率: 65.26%
```

---

## 📊 评估模型

### 基本用法

```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir /tmp/dataset/train
```

### 评估指标

- **分类准确率**: 正确分类的样本比例
- **mAP**: 平均精度均值，衡量检索效果
- **Top-1 准确率**: 最相似样本是同一个体的比例
- **Top-5 准确率**: 前5个最相似样本包含同一个体的比例

### 输出示例

```
============================================================
📊 评估结果
============================================================
  分类准确率: 81.05%
  mAP: 0.7523 (75.23%)
  Top-1 检索准确率: 78.42%
  Top-5 检索准确率: 92.11%
============================================================
```

---

## 📤 生成提交文件

### 基本用法

```bash
python test.py \
    --model_path checkpoints/best_model.pth \
    --test_csv test.csv \
    --test_img_dir /tmp/dataset/test \
    --output submission.csv
```

### 输出格式

```csv
row_id,similarity
0,0.8234
1,0.1234
2,0.9521
...
```

---

## 📈 实验结果

### 训练曲线

| Epoch | 训练 Loss | 训练 Acc | 验证 Loss | 验证 Acc |
|-------|-----------|----------|-----------|----------|
| 1 | 2.45 | 45.2% | 1.82 | 65.3% |
| 2 | 1.23 | 84.6% | 1.54 | 76.3% |
| 3 | 1.09 | 89.0% | 1.27 | **81.1%** |
| 4 | 1.02 | 90.5% | 1.34 | 80.5% |
| 5 | 0.97 | 93.0% | 1.38 | 79.0% |

> ⚠️ Epoch 3 后出现过拟合，最佳模型在 Epoch 3

### 最佳结果

| 指标 | 数值 |
|------|------|
| 验证准确率 | 81.05% |
| mAP | 75.23% |
| Top-1 Acc | 78.42% |
| Top-5 Acc | 92.11% |

---

## 📂 项目结构

```
jaguar-reid/
├── README.md              # 项目说明文档
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── evaluate.py            # 评估脚本
├── model.py               # 模型定义
├── checkpoints/           # 模型保存目录
│   └── best_model.pth     # 最佳模型权重
├── data/                  # 数据目录
│   ├── train/
│   └── test/
└── submission.csv         # 提交文件
```

---

## 🔍 常见问题

### Q1: 训练时 Acc 一直是 0%？

**原因**: 可能是损失函数配置问题（如 ArcFace 参数不当）

**解决**: 使用简单的交叉熵损失：
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Q2: GPU 利用率显示 0%？

**原因**: `nvidia-smi` 显示的是瞬时值，数据加载时 GPU 空闲

**解决**: 查看显存占用，有占用说明正在使用 GPU

### Q3: 过拟合怎么办？

**解决方案**:
- 使用早停（Early Stopping）
- 增加数据增强
- 减小学习率
- 增加 Dropout

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
- [OpenI 平台](https://openi.pcl.ac.cn/)

---

## 📧 联系方式

如有问题，请提交 Issue 或联系：

- 邮箱: 17763221787@163.com


---

<p align="center">
  <b>⭐ 如果这个项目对你有帮助，请给个 Star！⭐</b>
</p>
