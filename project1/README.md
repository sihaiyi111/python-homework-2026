# Project 1 · 用 NumPy 实现 OpenCLIP 图像编码器

使用课程提供的模型权重，补全 OpenCLIP ViT-B-32 图像编码器的六组 TODO。输入是预处理后的 RGB 图片，输出是每张图片对应的归一化 512 维特征向量。项目只做前向计算。

## 获取代码与权重

运行需要 Python 3.9 或以上。通过 GitHub 获取项目时，需要安装 Git 和 Git LFS。模型权重 `weights/image_encoder.npz` 约 311 MiB，由 Git LFS 管理。

```powershell
git lfs install
git clone https://github.com/sihaiyi111/python-homework-2026.git
cd python-homework-2026
git lfs pull
cd project1
```

如果使用老师发的原始附件，将本仓库的 `image_encoder.py` 复制到解压后的附件目录，替换同名文件。在该目录执行下面的验证命令。

## 复现实验结果

在包含 `check.py` 的目录运行：

```powershell
python -m pip install -r requirements.txt
python -u check.py --stage all
```

检查通过时，各阶段显示 `PASS`，最后输出 `Requested numerical checks passed.`。后几个阶段会分别运行完整的 12 层 Transformer，出结果前可能需要等待。

本次在 Windows、Python 3.14.0、NumPy 2.2.6 和 Pillow 12.0.0 下实测全部通过。最终输出形状为 `(10, 512)`，最大绝对误差为 `7.302e-07`，平均绝对误差为 `8.244e-08`。改变输入图片顺序并重复一张图片的批次检查也通过了。

如果资源检查提示权重缺失，或 `image_encoder.npz` 只有几百字节，请在克隆的仓库内运行 `git lfs pull`，再重新检查。

## 生成输出文件

```powershell
python run.py --output output_tokens.npz
```

默认处理课程提供的 10 张测试图片，输出文件包含图片 ID（`ids`）和对应的特征矩阵（`output_tokens`）。也可以指定一张图片：

```powershell
python run.py --images assets/images/image_01.jpg --output single_image_tokens.npz
```

## 主要文件

| 路径 | 用途 |
| --- | --- |
| `image_encoder.py` | 本项目实现的图像编码器，含六组 TODO 和中文注释 |
| `check.py` | 与课程参考结果进行分阶段比较 |
| `run.py` | 运行图像编码并保存输出 |
| `weights/image_encoder.npz` | 课程提供的模型权重 |
| `assets/images/`、`assets/reference_outputs.npz` | 课程提供的测试图片与验证用参考数据 |
