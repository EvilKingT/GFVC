# CUDA 12.4环境下的PyTorch和依赖安装指南

## 1. 环境准备

### 1.1 检查CUDA版本
```bash
nvcc --version
```
确保输出显示CUDA版本为12.4.x

### 1.2 检查Python版本
```bash
python --version
```
推荐使用Python 3.8-3.10版本

## 2. 创建虚拟环境

### 2.1 使用Anaconda创建环境
```bash
conda create -n gfvc_cuda124 python=3.9
conda activate gfvc_cuda124
```

### 2.2 或使用venv创建环境
```bash
python -m venv gfvc_cuda124
source gfvc_cuda124/bin/activate  # Linux/Mac
.\gfvc_cuda124\Scripts\activate  # Windows
```

## 3. 安装PyTorch和CUDA 12.4

### 3.1 使用pip安装
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3.2 或使用conda安装
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 4. 安装其他依赖

### 4.1 安装requirements_cuda124.txt中的依赖
```bash
pip install -r requirements_cuda124.txt
```

## 5. 验证安装

### 5.1 验证PyTorch和CUDA
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 5.2 验证compressai
```python
import compressai
print(compressai.__version__)
```

### 5.3 验证其他关键依赖
```python
import numpy
import opencv-python
import matplotlib
import mediapipe
```

## 6. 可能的兼容性问题及解决方案

### 6.1 compressai与PyTorch 2.4.0的兼容性
- 如果遇到compressai相关错误，可以尝试安装特定版本的compressai：
  ```bash
  pip install compressai==1.2.4 --no-deps
  ```

### 6.2 CUDA版本不匹配
- 确保NVIDIA驱动程序支持CUDA 12.4
- 可以通过NVIDIA官网更新驱动程序：https://www.nvidia.com/Download/index.aspx

### 6.3 Python版本问题
- 如果遇到Python版本不兼容的问题，可以尝试切换到Python 3.8或3.9版本

## 7. 运行项目

### 7.1 运行HDAC encoder
```bash
python source/HDAC_encoder_v0.py --help
```

## 8. 注意事项

1. 确保所有依赖项都已正确安装
2. 如果遇到任何问题，可以使用`pip check`命令检查依赖项冲突
3. 建议定期更新PyTorch和CUDA版本以获得更好的性能和兼容性