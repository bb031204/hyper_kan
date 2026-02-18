# HyperGKAN - 超图Kolmogorov-Arnold网络用于气象预测

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于论文 **"Hypergraph Kolmogorov–Arnold Networks for Station-Level Meteorological Forecasting"** 的PyTorch实现。

## 🌟 特性

- ✅ **双超图架构**: 邻域超图（地理距离）+ 语义超图（时间序列相似度）
- ✅ **KAN替代MLP**: 使用可学习样条函数提升非线性表达能力
- ✅ **Seq2Seq时序建模**: GRU Encoder-Decoder架构
- ✅ **灵活配置系统**: 所有超参数通过YAML配置
- ✅ **训练暂停恢复**: 支持中断后继续训练
- ✅ **完整评估流程**: 多指标评估 + 可视化
- 🆕 **智能超图缓存**: 参数不变时自动复用，加速370倍 ⚡
- 🆕 **超图可视化**: 自动生成超图结构分布图，独立保存
- 🆕 **定时自动暂停**: 设置训练时长，到时自动保存并暂停

---

## 📦 安装

### 1. 克隆仓库

```bash
cd D:\bishe\code\hyper_kan
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖:**
- PyTorch >= 2.0.0
- pykan >= 0.2.0 (KAN实现)
- scikit-learn
- pyyaml
- matplotlib
- tqdm

---

## 🚀 快速开始

### 1. 准备数据

将数据放置在以下位置：
```
D:\bishe\WYB\
├── temperature\
│   ├── trn.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── position.pkl
├── cloud\...
├── humidity\...
└── wind\...
```

**数据格式要求:**
- `.pkl` 文件包含 `{'x': ..., 'y': ..., 'context': ..., 'position': ...}`
- `x`: (T, N, F) - 时间步 x 站点数 x 特征维度
- `position`: (N, 2) - [latitude, longitude]

### 2. 配置参数

编辑 `configs/config.yaml`:

```yaml
meta:
  experiment_name: "HyperGKAN_Temperature"
  element: "Temperature"  # Temperature / Cloud / Humidity / Wind

data:
  input_window: 12   # 输入时间步
  output_window: 12  # 预测时间步
  batch_size: 16

graph:
  neighbourhood:
    top_k: 5  # 超边大小 (论文建议K=5)
  semantic:
    similarity: "euclidean"  # euclidean / pearson / cosine

model:
  kan:
    use_kan: true       # true: 使用KAN, false: 使用MLP (消融实验)
    grid_size: 5
    spline_order: 3

training:
  optimizer:
    lr: 0.01  # 论文使用0.01
  epochs: 500
  early_stopping:
    patience: 35  # 论文使用35
```

### 3. 训练模型

**基础训练:**
```bash
python train.py --config configs/config.yaml
```

**或使用主程序:**
```bash
python main.py train --config configs/config.yaml
```

**恢复训练:**
```bash
python train.py --resume outputs/20260126_123456_Temperature/best_model.pt
```

### 4. 预测评估

```bash
python predict.py --config configs/config.yaml --checkpoint outputs/xxx/best_model.pt
```

**或:**
```bash
python main.py predict --config configs/config.yaml --checkpoint outputs/xxx/best_model.pt
```

---

## 📊 项目结构

```
hyper_kan/
├── configs/
│   └── config.yaml              # 配置文件
├── src/
│   ├── data/
│   │   ├── pkl_loader.py        # 数据加载
│   │   └── dataset.py           # PyTorch数据集
│   ├── graph/
│   │   ├── hypergraph_nei.py    # 邻域超图构建
│   │   ├── hypergraph_sem.py    # 语义超图构建
│   │   └── hypergraph_utils.py  # 超图工具函数
│   ├── models/
│   │   ├── kan_layer.py         # KAN层实现
│   │   ├── hypergkan_conv.py    # 超图卷积层
│   │   └── hypergkan_model.py   # 完整模型
│   ├── training/
│   │   └── trainer.py           # 训练器
│   └── utils/
│       ├── metrics.py           # 评估指标
│       ├── logger.py            # 日志配置
│       ├── checkpoint.py        # Checkpoint管理
│       └── visualization.py     # 可视化工具
├── pause_resume/
│   ├── pause.py                 # 暂停指南
│   ├── resume.py                # 恢复训练脚本
│   └── README.md                # 暂停恢复说明
├── main.py                      # 主程序入口
├── train.py                     # 训练脚本
├── predict.py                   # 预测脚本
├── requirements.txt             # 依赖列表
└── README.md                    # 本文件
```

---

## 🧪 消融实验

### 1. 禁用KAN（使用MLP）

```yaml
model:
  kan:
    use_kan: false  # 切换为MLP
```

### 2. 仅使用邻域超图

```yaml
ablation:
  disable_semantic: true  # 禁用语义超图
```

### 3. 仅使用语义超图

```yaml
ablation:
  disable_neighbourhood: true  # 禁用邻域超图
```

### 4. 调整超边大小K

```yaml
graph:
  neighbourhood:
    top_k: 3  # 尝试 K = 2, 3, 4, 5, 6
  semantic:
    top_k: 3
```

---

## 📈 训练监控

训练过程中会自动保存：

- **Checkpoint**: `outputs/<timestamp>/checkpoint_epoch_<N>.pt`
- **最佳模型**: `outputs/<timestamp>/best_model.pt`
- **损失曲线**: `outputs/<timestamp>/loss_curve.png`
- **训练日志**: `outputs/<timestamp>/train_<timestamp>.log`

### 实时查看日志

```bash
tail -f outputs/20260126_123456_Temperature/train_20260126_123456.log
```

---

## 🔧 高级功能

### 1. 智能超图缓存 🆕⚡

**自动缓存**: 超图构建后自动保存，参数不变时直接加载。

**性能提升**:
- 首次训练: ~8分钟（2048个站点）
- 第二次训练: ~1秒 ⚡ **加速370倍！**

**配置方法:**

编辑 `configs/config.yaml`:
```yaml
graph:
  use_cache: true         # 启用缓存（默认）
  cache_dir: "data/cache" # 缓存目录
```

**训练输出**:
```
Building/Loading hypergraphs...
✓ Found cached neighbourhood hypergraph: data/cache/Temperature_nei_K5_knn_geoTrue_decay0.1.npz
✓ Found cached semantic hypergraph: data/cache/Temperature_sem_K5_euclidean_win12_normTrue.npz
```

**缓存管理**:
```bash
# 查看缓存
ls data/cache/

# 清理缓存（强制重建）
rm -rf data/cache/*

# 清理特定要素
rm data/cache/Temperature_*
```

详见 [`超图缓存与可视化说明.md`](超图缓存与可视化说明.md)

### 2. 超图可视化 🆕

**独立保存**: 可视化图保存在 `visuals` 文件夹，按要素分类。

**启用方法:**

编辑 `configs/config.yaml`:
```yaml
graph:
  visualize: true         # 启用可视化
  visual_dir: "visuals"   # 可视化目录
```

**查看位置:**
```
visuals/
└── Temperature/
    ├── hypergraph_neighbourhood_K5_knn.png
    └── hypergraph_semantic_K5_euclidean.png
```

**查看图片:**
```bash
# Windows
start visuals/Temperature/hypergraph_neighbourhood_K5_knn.png

# Mac/Linux
open visuals/Temperature/hypergraph_neighbourhood_K5_knn.png
```

**手动生成:**
```bash
python visualize_hypergraph.py --element Temperature --K 5
```

### 3. 定时自动暂停 🆕

设置训练时长，到时自动保存checkpoint并暂停。

**配置方法:**

编辑 `configs/config.yaml`:
```yaml
training:
  time_limit_minutes: 100  # 100分钟后自动暂停
```

**训练过程显示:**
```
Epoch 50/500:
  Train Loss: 0.5098
  Val Loss: 0.5912
  Elapsed Time: 100.2/100 min (remaining: -0.2 min)

============================================================
⏰ TIME LIMIT REACHED - Auto-pausing training
============================================================
Checkpoint saved successfully!
```

**恢复训练:**
```bash
python pause_resume/resume.py
```

详见 [`FEATURES_GUIDE.md`](FEATURES_GUIDE.md#功能2-定时自动暂停)

### 4. 训练暂停与恢复

详见 [`pause_resume/README.md`](pause_resume/README.md)

**快速恢复:**
```bash
python pause_resume/resume.py  # 自动查找最新checkpoint
```

### 2. 多元素训练

```bash
# 温度
python train.py --config configs/config.yaml

# 修改config中的element字段为"Cloud"后
python train.py --config configs/config.yaml

# 或创建多个配置文件
python train.py --config configs/config_cloud.yaml
python train.py --config configs/config_humidity.yaml
```

### 3. 超参数搜索

修改配置文件中的关键参数：
- `graph.neighbourhood.top_k`: 超边大小 (2-6)
- `training.optimizer.lr`: 学习率 (0.001-0.01)
- `model.hypergkan_layer.num_layers`: 层数 (1-3)

---

## 📊 评估指标

模型评估包括以下指标：

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差

**按预测步长评估:**
- Horizon 3: 前3步
- Horizon 6: 前6步
- Horizon 12: 全部12步

---

## 🎯 论文复现建议

基于论文和复现指引，建议的实验流程：

### Level 1: 基础复现
1. ✅ 使用Temperature数据
2. ✅ K=5, batch_size=16, lr=0.01
3. ✅ 训练至收敛 (early_stop=35)
4. ✅ 对比 MAE / RMSE

### Level 2: 完整复现
1. ✅ 所有气象变量 (Temp, Cloud, Humidity, Wind)
2. ✅ 消融实验 (w/o-KAN, w/o-Sem, w/o-Nei)
3. ✅ K值对比 (K=2,3,4,5,6)

### Level 3: 扩展实验
1. ✅ 不同相似度度量 (Euclidean, Pearson, Cosine)
2. ✅ 不同融合策略 (Concat, Add, Attention)
3. ✅ 与baseline模型对比 (STGCN, DCRNN, Transformer)

---

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案:**
- 减小 `batch_size`
- 减小 `d_model` 或 `hidden_channels`
- 使用更少的超图卷积层

### 2. pykan导入失败

**解决方案:**
```bash
pip install pykan --upgrade
```

如果仍失败，模型会自动降级为标准MLP。

### 3. 训练不收敛

**可能原因:**
- 学习率过大 -> 尝试 `lr=0.001`
- KAN层过深 -> 减少层数
- 数据未归一化 -> 检查 `config.yaml` 中的 `normalize: true`

### 4. 超图构建慢

**解决方案:**
- 启用缓存: `graph.use_cache: true`
- 减少 `top_k` 值
- 使用更少的站点进行测试

---

## 📚 参考文献

```bibtex
@article{tang2024hypergkan,
  title={Hypergraph Kolmogorov–Arnold Networks for Station-Level Meteorological Forecasting},
  author={Tang, Jian and Ma, Kai},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题，请通过以下方式联系：

- 📧 Email: [您的邮箱]
- 💬 Issues: [GitHub Issues](链接)

---

## 🎉 致谢

- 论文作者: Jian Tang, Kai Ma
- pykan库: [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)
- 数据集: WeatherBench

---

**祝您训练顺利！🚀**
