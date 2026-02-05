import openi
import os
from pathlib import Path

# 1. 登录到OpenI平台（需要获取访问令牌）
# 从平台获取token：登录后进入"设置"->"访问令牌"生成
token = "6c023df1ab74a27abab260d66d8d281d5024c8e8"  # 替换为您的实际token
openi.login(token=token)

# 2. 验证登录状态
user_info = openi.whoami()
print(f"当前登录用户: {user_info}")

# 3. 设置数据集仓库信息
repo_id = "jia11111/train"  # 格式: 用户名/仓库名
local_dataset_path = r"D:\train\train"  # 本地数据集路径

# 4. 检查本地数据集是否存在
if not os.path.exists(local_dataset_path):
    print(f"错误: 数据集路径 {local_dataset_path} 不存在")
    exit(1)  # 使用exit而不是return，因为不在函数内

# 5. 上传数据集
try:
    print("开始上传数据集...")
    result = openi.openi_upload_file(
        repo_id=repo_id,
        file_or_folder_path=local_dataset_path,
        repo_type="dataset"  # 指定为数据集类型
    )
    print("数据集上传成功!")

except Exception as e:
    print(f"上传失败: {e}")


