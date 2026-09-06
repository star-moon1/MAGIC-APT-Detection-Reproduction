# MAGIC 论文复现

本仓库是 USENIX Security 2024 论文 **MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning** 的复现项目，完成了批级检测场景下两个核心数据集的快速评估实验。

## 复现环境
- 系统：WSL2 Ubuntu 22.04
- Python 3.9
- DGL 1.0.0
- scikit-learn 1.2.2
- PyTorch

## 数据集说明
本次复现覆盖两个批级检测数据集：
1. **Unicorn Wget**：150 个溯源图（125 良性 + 25 攻击）
2. **StreamSpot**：600 个溯源图（400 良性 + 200 攻击）

> 数据集与预训练权重来自官方仓库 [FDUDSDE/MAGIC](https://github.com/FDUDSDE/MAGIC)，因文件较大未上传本仓库，可前往官方下载。

## 实验结果（100 次随机种子平均）
### Wget 数据集
| 指标 | 结果 |
|------|------|
| AUC | 0.9739 ± 0.0190 |
| F1 | 0.9436 |
| Precision | 0.9139 |
| Recall | 0.9778 |

### StreamSpot 数据集
| 指标 | 结果 |
|------|------|
| AUC | 0.9995 ± 0.0007 |
| F1 | 0.9954 |
| Precision | 0.9920 |
| Recall | 0.9990 |

## 运行方式
1. 下载官方数据集解压到 `data/` 对应目录
2. 安装依赖：`pip install -r requirements.txt`
3. 运行评估：
   - Wget：`python eval.py --dataset wget`
   - StreamSpot：`python eval.py --dataset streamspot`

## 目录结构
- `code/`：项目核心代码
- `report/`：完整复现报告
- `results/`：实验结果截图与原始数据
