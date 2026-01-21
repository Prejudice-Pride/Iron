# Iron - 地震地表运动追踪与分析系统 (Seismic Surface Tracking System)

本项目是一个用于轨迹追踪的工具集。它集成了深度学习（CoTracker）与传统视觉算法（LK 光流），旨在从视频中提取高精度的运动位移和波形数据。

## 核心构成

项目采用模块化设计，主要由以下几部分构成：

### 1. 追踪引擎 (Core Engines)

提供了三种不同的追踪模式，满足不同场景下的性能与精度需求：

- **CoTracker Offline (`core.engine`)**: 基于 Facebook 的 CoTracker 模型，适用于需要高精度、全局一致性的离线视频处理。
- **CoTracker Online (`core.online_engine`)**: 针对实时或流式场景设计的在线追踪版本。
- **LK Optical Flow (`core.lk_engine`)**: 基于传统的 Lucas-Kanade 算法，无需深度学习模型权重，运行极其迅速，适合背景简单、运动较小的场景。

### 2. 采样管理 (Sampler)

- **`core.sampler`**: 负责确定追踪点位。它支持交互式框选感知区域 (ROI)，并将 ROI 信息保存为 JSON 文件以便重复使用。它能根据图像特征在指定区域内自动生成初始追踪点。

### 3. 数据分析与后处理 (Analyzer)

- **`core.analyzer`**: 负责对原始轨迹数据进行处理。
  - 提取目标物体的相对位移，计算平均波形 (Waveform)。
  - 支持将轨迹数据导出为 `.npz`（压缩数组）和 `.csv`（结构化表格）格式。

### 4. 辅助工具 (Utilities)

- **`utils.video_io`**: 统一的视频帧读取策略。
- **`utils.visualizer`**: 提供结果可视化功能，将追踪轨迹实时渲染回视频中，方便直观评估。

---

## 快速开始

### 环境配置

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 确保 `co-tracker` 子模块已正确配置，并下载相应的模型权重文件（如 `scaled_offline.pth` 和 `scaled_online.pth`）至 `co-tracker/checkpoints/` 目录下。

### 使用方法

根据需求运行对应的入口脚本：

- **CoTracker 离线追踪**:

  ```bash
  python main_surface.py [视频路径] [输出目录]
  ```

- **CoTracker 在线追踪**:

  ```bash
  python main_online.py [视频路径] [输出目录]
  ```

- **LK 光流快速追踪**:

  ```bash
  python main_lk.py [视频路径] [输出目录]
  ```

---

## 项目结构

```text
├── core/               # 核心逻辑模块 (Engine, Sampler, Analyzer)
├── co-tracker/         # CoTracker 依赖库及权重
├── utils/              # 视频 IO 与可视化工具
├── input_videos/       # 待处理视频输入目录
├── rois/               # 保存的感兴趣区域配置文件 (JSON)
├── output/             # 追踪轨迹与分析结果输出
└── main_*.py           # 不同追踪模式的启动入口
```

## 贡献与参考

本项目深度集成了 [CoTracker](https://github.com/facebookresearch/co-tracker) 算法。

## 后续更新

未来将添加、适配更多的深度学习算法或传统光流追踪算法，敬请期待。
