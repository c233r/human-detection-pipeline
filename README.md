# 人体目标检测 · 关键点解析 · 持绳二分类流水线

一条基于 **Kafka 解耦 + Ray 流水线并行 + TensorRT/ONNX 加速** 的多阶段视频推理流水线，
目标是从原始视频中产出**带人体框、人体关键点、以及"持绳 / 非持绳"二分类标签的结构化标注**。

> 本仓库是「人体目标检测」项目，是整条数据生产链路的第二环。
> 上游是 **`01-bilibili-crawler`**（B站视频采集与 VLM 初筛），
> 下游是 **`03-clip-clustering`**（CLIP 后处理聚类切分）。
> 三者合起来构成一条完整的数据生产链路。

---

## 目录

- [项目简介](#项目简介)
- [流水线总览](#流水线总览)
- [三个阶段](#三个阶段)
- [目录结构](#目录结构)
- [上游依赖与目录约定](#上游依赖与目录约定)
- [环境与安装](#环境与安装)
- [快速开始](#快速开始)
- [数据契约](#数据契约)
- [关键工程设计](#关键工程设计)
- [脚本演进对照](#脚本演进对照)
- [注意事项](#注意事项)
- [License](#license)

---

## 项目简介

把视频变成可训练的标注数据，需要连续回答三个问题：

1. **画面里有没有人？人在哪？** —— 目标检测（EdgeCrafter，`person / face / ball` 三类）
2. **人的姿态是什么？手在哪？** —— 人体关键点解析（mmpose RTMPose）
3. **这个人有没有握着绳子？** —— 手部裁剪 + 二分类（ConvNeXt，`hold_rope` 0/1）

三个阶段各自是一个独立的 GPU 服务，用 **Kafka Topic 串成一条流水线**。
这样做的好处是每个阶段可以**单独扩容、单独重启、单独重放**：

- 检测服务想换模型权重 → 只重启检测服务，Kafka 里堆积的任务不会丢；
- 关键点阶段慢 → 只给它加 worker，不影响上游已经产出的结果；
- 分类阈值想调 → 直接重放 `hold_rope_keypoints` topic，不用重跑前面两个阶段。

### 功能特性

| 能力 | 说明 |
| --- | --- |
| 三阶段解耦 | Kafka Topic 作为阶段间缓冲区，阶段可插拔、可独立重放 |
| 多卡多进程 | 一个服务实例管理多张 GPU，每卡多个 worker 进程 |
| 动态扩缩容 | 按 Kafka consumer lag 自动增减 worker（阈值/上下限可配） |
| Ray 流水线并行 | 预处理(CPU) → GPU 推理 → 后处理(CPU) 拆成三级 pipeline，用满 CPU 与 GPU |
| ONNX + TensorRT | 导出 ONNX 并转 FP16 TensorRT engine，显著降低推理延迟 |
| JPEG 传输优化 | 阶段间用 JPEG 字节流而非原始 ndarray 传递，减少对象存储压力与序列化开销 |
| 结果校验修复 | `validate_labels.py` 交叉核对 label 完整性，自动把缺失任务重投到对应 topic |
| 分类服务合并 | 手部裁剪与分类合并为单个服务，减少一次跨阶段数据传输 |

---

## 流水线总览

```
                  ┌──────────────────────────────────────────────────────┐
                  │  上游：01-bilibili-crawler（VLM 初筛后的视频）          │
                  └──────────────────────────────────────────────────────┘
                                      │
                    Flume Spooldir / Kafka Producer / HTTP 上传
                                      │
                                      ▼
   ╔══════════════════════════════════════════════════════════════════════╗
   ║  阶段 1 · 人体目标检测                        service/                ║
   ║  topic: video_tasks ──▶ EdgeCrafter (ecdet_l) ──▶ topic: video_results║
   ║  产出: 每帧的 person / face / ball 检测框 + track_id                   ║
   ╚══════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
   ╔══════════════════════════════════════════════════════════════════════╗
   ║  阶段 2 · 人体关键点解析                      mmpose_scripts/          ║
   ║  topic: video_results ──▶ RTMPose (rtmpose-l) ──▶ hold_rope_keypoints ║
   ║  产出: 人体 17/133 点关键点 + 手部框                                  ║
   ╚══════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
   ╔══════════════════════════════════════════════════════════════════════╗
   ║  阶段 3 · 手部裁剪 + 持绳二分类          classification_scripts/      ║
   ║  topic: hold_rope_keypoints ──▶ ConvNeXt ──▶ topic: final_videos      ║
   ║  产出: hold_rope (0/1) + hand_left_hold / hand_right_hold + 置信度      ║
   ╚══════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                  ┌──────────────────────────────────────────────────────┐
                  │  utils/validate_labels.py                             │
                  │  交叉核对 final_videos 与 temp 下的 labels 是否完整      │
                  │  缺失任务自动重投到对应 topic 重算                       │
                  └──────────────────────────────────────────────────────┘
                                      │
                                      ▼
                  ┌──────────────────────────────────────────────────────┐
                  │  下游：03-clip-clustering（CLIP 嵌入 → 层次聚类 → 切分）  │
                  └──────────────────────────────────────────────────────┘
```

所有阶段共享同一个输出根目录（默认 `/data2/caimoyao/api_outputs`），
中间态的 labels JSON 落在 `api_outputs/temp/`，供 `validate_labels.py` 校验。

---

## 三个阶段

### 阶段 1 · 人体目标检测（`service/`）

基于 **EdgeCrafter** 的轻量检测器（`ecdet_l`），检测类别为 `["person", "face", "ball"]`。

服务形态从简单到复杂：

| 文件 | 形态 |
| --- | --- |
| `api_service.py` | 纯 FastAPI，HTTP 上传视频 → 抽帧 → 推理 → 返回/落盘 JSON |
| `api_service_kafka.py` | 消费 Kafka `video_tasks`，结果投递 `video_results`（多进程 + 多卡） |
| `api_service_kafka_ray.py` | 引入 Ray Actor 管理 GPU worker |
| `api_service_kafka_ray_multi.py` | 多 GPU + 每卡多 worker，按空闲显存动态计算 worker 数 |
| `api_service_kafka_ray_multi_jpeg.py` | **最新版**：流水线并行 + JPEG 传输 + 动态扩缩容 |

处理流程：

```
视频上传 / Kafka 消息
   │
   ├─ 抽帧 (OpenCV / ffmpeg)，默认 FPS=0.5
   │
   ├─ 推理：调用 EdgeCrafter 的 tools/inference/torch_inf.py
   │        或 ONNX/TensorRT engine（edgecrafter_tools/onnx_inf.py）
   │
   ├─ 后处理：NMS、置信度过滤、跨帧 track_id 关联
   │
   └─ 落盘 labels JSON + 投递 Kafka video_results
```

### 阶段 2 · 人体关键点解析（`mmpose_scripts/`）

基于 **mmpose RTMPose-L**，从阶段 1 的检测框裁出人体区域，回归关键点。

| 文件 | 形态 |
| --- | --- |
| `monitor_hold_rope.py` | 目录轮询版：盯着一个目录，有新结果就推理 |
| `monitor_hold_rope_kafka.py` | Kafka 消费版 |
| `monitor_hold_rope_kafka_stream.py` | 流式处理，边读视频边推理，避免全量落盘 |
| `monitor_hold_rope_kafka_stream_ray.py` | Ray Actor 管理 GPU |
| `monitor_hold_rope_kafka_stream_ray_multi.py` | 多 GPU 多 worker + 动态扩缩容 |
| `monitor_hold_rope_kafka_stream_ray_multi_jpeg.py` | **最新版**：三级 pipeline（preprocess → GPU → postprocess）+ JPEG 传输 |

配套的独立推理脚本（用于单测 / 基准 / 导出验证）：

```
inference_onnx_multi_rtmpose_l.py                        # ONNX 多路推理
inference_tensorRT_multi_rtmpose_l.py                    # TensorRT 多路推理
inference_tensorRT_multi_rtmpose_l_stream.py             # 流式 + TensorRT
inference_tensorRT_multi_rtmpose_l_stream_ray.py         # 流式 + TensorRT + Ray
```

Ray 流水线的三级结构：

```
video_results ──▶ kps_preprocess (CPU) ──▶ KpsGPUInferActor (GPU) ──▶ kps_postprocess (CPU) ──▶ hold_rope_keypoints
                     │                            │                            │
                 解码/缩放/归一化              RTMPose 前向                  热图→坐标
                 可与 GPU 推理重叠执行 ────────┘                             + 手部框推导
```

关键配置（`Config` 类）：

```python
KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
KAFKA_CONSUME_TOPIC     = "video_results"
KAFKA_PRODUCE_TOPIC     = "hold_rope_keypoints"
KAFKA_FINAL_TOPIC       = "final_videos"
GPU_IDS                 = [6]
WORKERS_PER_GPU         = 4
RAY_OBJECT_STORE_MEMORY = 16 * 1024**3
PIPELINE_DEPTH          = 3       # 流水线并行深度

# 动态扩缩容
LAG_CHECK_INTERVAL      = 600     # 每 600s 检查一次消费积压
SCALE_UP_THRESHOLD      = 5       # 积压 > 5 个分区 → 扩容
SCALE_DOWN_THRESHOLD    = 1
MIN_TOTAL_WORKERS       = 3
MAX_TOTAL_WORKERS       = 4
```

### 阶段 3 · 手部裁剪 + 持绳二分类（`classification_scripts/`）

基于 **ConvNeXt-Base** 做二分类：`持绳` / `非持绳`。

| 文件 | 形态 |
| --- | --- |
| `hand_crop_service.py` | 手部裁剪 + 分类（目录版） |
| `hand_crop_service_kafka.py` | Kafka 消费版 |
| `hand_crop_service_kafka_onnx.py` | **改用 ONNX 推理**，解决 PyTorch 推理慢的问题 |
| `hand_crop_service_kafka_onnx_ray.py` | Ray 管理 GPU |
| `hand_crop_service_kafka_onnx_ray_multi.py` | 多 GPU 多 worker |
| `hand_crop_service_kafka_onnx_ray_multi_jpeg.py` | **最新版**：+ JPEG 传输 |
| `hold_rope_classify_service.py` | 纯分类服务（不含裁剪），便于单独调试 |
| `export_onnx.py` | 把 ConvNeXt 权重导出为 ONNX（含 FP16） |
| `inference_list2.py` | 对文件列表批量推理，用于验证导出正确性 |
| `draw_labels_on_video.py` | 把分类结果画回视频，人工抽检用 |
| `shwoHandCrop.py` | 可视化手部裁剪区域，调裁剪策略用 |
| `processJumpRopeVideos.py` | 跳绳视频批量处理入口 |

**为什么合并裁剪与分类？** 早期裁剪和分类是两个独立服务，
中间要经过一次 Kafka 往返（裁剪后的手部图落盘 → 分类服务再读），
既增加延迟又占用磁盘。合并后在同一进程内完成，少一次序列化与 IO。

阶段 3 消费 `hold_rope_keypoints`，利用阶段 2 给出的手部框做裁剪，
推理后把 `hold_rope` 等字段合并进原有 label，投递到 `final_videos`（终态 topic）。

---

## 目录结构

```
02-human-detection/
├── README.md
├── .gitignore
│
├── service/                       # 阶段 1：EdgeCrafter 检测服务
│   ├── api_service.py
│   ├── api_service_kafka.py
│   ├── api_service_kafka_ray.py
│   ├── api_service_kafka_ray_multi.py
│   └── api_service_kafka_ray_multi_jpeg.py      ← 最新
│
├── client/                        # 阶段 1 的调用方
│   ├── client_example.py                        # HTTP 客户端 + 可视化
│   ├── client_example_kafka.py                  # 往 Kafka 投递视频任务
│   └── client_example_kafka_test.py
│
├── pipeline/                      # 数据准备与基准
│   ├── data_pipeline.py                         # 检测结果 → YOLO/COCO 数据集
│   ├── extract_frames_batch.py                  # 批量抽帧
│   ├── video_utils.py                           # 视频读取（OpenCV / ffmpeg 双实现）
│   ├── compress_service_kafka.py                # 结果压缩服务
│   └── benchmark_onnx_inf.py                    # ONNX 推理基准测试
│
├── kafka/                         # 消息接入
│   ├── kafka_bridge.py                          # 目录 → Kafka 桥接
│   ├── watch_video_directory.py                 # 监听视频目录，自动入队
│   ├── flume_bridge_video_watcher.py            # Flume 幂等桥接
│   ├── flume-video-agent.conf                   # Flume Agent 配置
│   ├── check_topics.sh / check_partitions.py    # topic / 分区巡检
│   ├── reset_offsets.sh                         # 重置消费位点（重放用）
│   └── cleanup_kafka_test_topics.sh             # 清理测试 topic
│
├── ray_scripts/                   # Ray 集群
│   ├── ray_object_keeper.py / ray_object_keeper_mutli.py   # 对象常驻保活
│   ├── ray_start_head.sh
│   └── start_ray_multi.sh
│
├── scripts/                       # 服务启动脚本
│   ├── start_all_services.sh                    # 一键起停三个服务
│   ├── start_api.sh
│   ├── start_video_watcher.sh
│   └── start_flume_video_watcher.sh
│
├── edgecrafter_tools/             # EdgeCrafter 推理脚本（在官方脚本上改写）
│   ├── torch_inf.py                             # PyTorch 推理 + RAG 聚类钩子
│   ├── onnx_inf.py                              # ONNX / TensorRT 推理
│   ├── export_ecdet_onnx.py                     # 导出 ONNX
│   ├── convert_onnx_fp16.py                     # FP16 转换
│   └── debug_compile_graph_break.py             # 定位 ONNX 图打断点
│
├── mmpose_scripts/                # 阶段 2：关键点
│   ├── monitor_hold_rope*.py                    # 6 个版本，见上文对照
│   └── inference_*_rtmpose_l*.py                # 4 个独立推理脚本
│
├── classification_scripts/        # 阶段 3：持绳二分类
│   ├── hand_crop_service*.py                    # 6 个版本
│   ├── hold_rope_classify_service.py
│   ├── export_onnx.py / inference_list2.py
│   └── draw_labels_on_video.py / shwoHandCrop.py / processJumpRopeVideos.py
│
├── label_tools/                   # 标注数据处理
│   ├── extract_images.py                        # 从结果中导出目标 crop 图片
│   ├── process_hold_rope.py                     # 生成持绳标注
│   └── process_hold_rope_kafka.py               # Kafka 版
│
├── visualization/                 # 可视化与人工抽检
│   ├── angle_plot.py                            # 关节角度分布绘图
│   ├── draw_angles.py                           # 把角度画在图上
│   └── visualize_labels_video.py                # labels 叠加到视频
│
├── utils/                         # 校验工具
│   ├── validate_labels.py                       # 标签完整性校验 + 自动修复
│   ├── check_hold_rope_labels.py                # 持绳标签专项检查
│   ├── compress_labels.py                       # 标签压缩
│   └── verify_object_keeper_ref.py              # 校验对象保活引用
│
└── requirements/
    ├── requirements_api.txt                     # 检测服务依赖
    └── cmy_py310_requirements.txt               # 完整环境（含 ray / onnx / TRT）
```

---

## 上游依赖与目录约定

> **本仓库只包含我自己编写/改写的脚本，不包含第三方框架源码。**
> 下面这些上游项目需要你自行从官方仓库获取，并保持原有的相对目录结构。

| 上游项目 | 用途 | 获取方式 |
| --- | --- | --- |
| **EdgeCrafter** | 阶段 1 的检测模型与推理框架 | 官方仓库（`ecdetseg/` 含 `configs/`、`tools/inference/`、`models/`） |
| **mmpose** | 阶段 2 的 RTMPose 关键点模型 | <https://github.com/open-mmlab/mmpose> |
| **Awesome-Backbones** | 阶段 3 的 ConvNeXt 分类训练框架 | <https://github.com/Fafa-DL/Awesome-Backbones> |
| **TensorRT** | ONNX → FP16 engine，推理加速 | NVIDIA 官方 |

期望的目录布局（脚本中的 `sys.path` 与相对路径按此约定）：

```
<项目根>/
├── EdgeCrafter-main/
│   └── ecdetseg/
│       ├── configs/ecdet/ecdet_l.yml
│       ├── models/last.pth
│       └── tools/inference/{torch_inf.py, onnx_inf.py, rag_db.py, ...}
├── mmpose/
│   └── scripts_cmy/            ← 本仓库 mmpose_scripts/ 的内容放这里
├── Jump_rope/
│   └── classification/         ← Awesome-Backbones
│       ├── logs/ConvNeXt/<时间戳>/{*.py, *_fp16.onnx}
│       ├── datas/annotations.txt
│       └── scripts_cmy/        ← 本仓库 classification_scripts/ 的内容放这里
└── TensorRT-10.16.1.11/
```

**需要按你的环境修改的路径**（脚本中硬编码了作者服务器的绝对路径）：

| 位置 | 变量 | 原值 |
| --- | --- | --- |
| `service/*.py` | `Config.PROJECT_ROOT` | `/data2/caimoyao/EdgeCrafter-main/ecdetseg` |
| `service/*.py` | `Config.OUTPUT_DIR` | `/data2/caimoyao/api_outputs` |
| `service/*_jpeg.py` | `TRT_PATH` | `/data2/caimoyao/TensorRT-10.16.1.11` |
| `mmpose_scripts/*.py` | `Config.TEMP_DIR` | `/data2/caimoyao/api_outputs/temp` |
| `utils/validate_labels.py` | `TEMP_DIR` / `LOG_FILE` | 同上 |
| `scripts/*.sh` | `PID_DIR` / `LOG_DIR` | `/data2/caimoyao/{pids,logs}` |
| `scripts/*.sh` | conda 环境名 | `cmy_py310` / `mmpose` / `hold_rope_classify` |

建议做法：把这几个路径抽成环境变量或统一的 `config.py`，再在各脚本里读取。
（本仓库保持原样，是为了忠实保留当时的运行形态。）

---

## 环境与安装

- **Python** 3.10（检测服务）/ 3.11~3.12（mmpose 服务）/ 3.12（分类服务）—— 原环境按 conda 分环境隔离
- **CUDA** 与 **TensorRT** 版本需与 onnxruntime-gpu 匹配（作者环境为 TensorRT 10.16）
- **Kafka** 集群（原环境 `localhost:9094`，单机 KRaft 模式）
- **Ray** ≥ 2.x
- **ffmpeg**

```bash
# 检测服务
pip install -r requirements/requirements_api.txt

# 完整环境（含 ray / onnxruntime / confluent-kafka / TensorRT wheel）
pip install -r requirements/cmy_py310_requirements.txt
```

> `cmy_py310_requirements.txt` 是一份 `pip freeze` 快照，包含 CUDA 13 / TensorRT 10.16
> 相关的 wheel。如果你的 CUDA 版本不同，请只把它当作**参考清单**，按需安装。

---

## 快速开始

### 1. 起 Kafka 与 Ray

```bash
# Kafka（按你自己的安装方式）
<KAFKA_HOME>/bin/kafka-server-start.sh -daemon <KAFKA_HOME>/config/kraft/server.properties

# Ray head + worker
bash ray_scripts/ray_start_head.sh
bash ray_scripts/start_ray_multi.sh
```

### 2. 起三个阶段的服务

```bash
# 一键起停（改好脚本里的 conda 环境名与路径）
bash scripts/start_all_services.sh start     # 启动全部
bash scripts/start_all_services.sh status    # 查看状态
bash scripts/start_all_services.sh stop      # 停止全部

# 或单独控制
bash scripts/start_all_services.sh start api_service
bash scripts/start_all_services.sh start monitor_hold_rope
bash scripts/start_all_services.sh start hand_crop
```

### 3. 投递视频

```bash
# 方式 A：HTTP 直传
python client/client_example.py --infer /path/to/video.mp4 --visualize

# 方式 B：投递整个目录到 Kafka（由流水线自动消费）
python client/client_example_kafka.py --send-dir /path/to/videos

# 方式 C：让 watcher 监听目录，新文件自动入队
python kafka/watch_video_directory.py --watch-dir /path/to/videos
```

### 4. 校验结果

```bash
# 仅检查
python utils/validate_labels.py --topic-only

# 检查并自动重投缺失任务
python utils/validate_labels.py --repair

# 只查某个任务
python utils/validate_labels.py --task <TASK_ID>
```

---

## 数据契约

三个阶段通过 **同一份 labels JSON** 传递与累积字段。
文件名对应 `{task_id}_{track_id}.json`，内容是一个 entry 数组：

```jsonc
[
  {
    // ── 阶段 1 产出 ──
    "label": 1,                    // 类别 id（0=person, 1=face, 2=ball 按 CLASSES 顺序）
    "score": 0.92,                 // 检测置信度
    "box": [x1, y1, x2, y2],       // 人体框
    "track_id": 7,                 // 跨帧跟踪 id

    // ── 阶段 2 产出（label == 1 时必需）──
    "keypoints": [[x, y, conf], ...],   // 人体关键点
    "hand_left_hold":  ...,
    "hand_right_hold": ...,
    "hand_left_score":  ...,
    "hand_right_score": ...,

    // ── 阶段 3 产出 ──
    "hold_rope": 1                 // 1=持绳, 0=非持绳
  }
]
```

完整性规则（由 `validate_labels.py` 强制执行）：

```python
BASE_FIELDS = {"label", "score", "box", "track_id"}

REQUIRED_FIELDS_LABEL1 = {
    "keypoints",
    "hold_rope",
    "hand_left_hold", "hand_right_hold",
    "hand_left_score", "hand_right_score",
}
```

即：**任何 entry 都必须有 `BASE_FIELDS`；`label == 1` 的 entry 还必须补齐 `REQUIRED_FIELDS_LABEL1`。**
缺哪个字段，`validate_labels.py` 就把该 `task_id` 重新投递到产出该字段的 topic 上重算：

| 缺失字段 | 重投到 |
| --- | --- |
| `box` / `score` / `track_id` | `video_tasks` |
| `keypoints` / `hand_*` | `video_results` |
| `hold_rope` | `hold_rope_keypoints` |

这套机制让流水线具备**自愈能力**：任意阶段挂掉重启后，跑一遍校验就能把漏算的补回来。

---

## 关键工程设计

### 1. 用 Kafka 做阶段解耦，而不是直接函数调用

直接串行调用的话，任一阶段崩溃都要从头重跑整条链路（检测 + 关键点 + 分类全都重算）。
用 Topic 隔离后，每个阶段的产物都是持久的、可重放的：

- 换检测模型 → 只重放 `video_tasks`；
- 调分类阈值 → 只重放 `hold_rope_keypoints`；
- 加一张 GPU → 新 worker 加入同一消费组，自动分担分区。

代价是要处理**恰好一次/幂等**问题，因此有 `flume_bridge_video_watcher.py` 的幂等检查和
`ray_object_keeper.py` 的对象保活引用校验。

### 2. 动态扩缩容由 consumer lag 驱动

固定 worker 数要么浪费 GPU，要么在高峰期积压。这里用 Kafka 的消费积压量做信号：

```python
LAG_CHECK_INTERVAL   = 600    # 检查周期
SCALE_UP_THRESHOLD   = 5      # 积压超阈值 → 扩
SCALE_DOWN_THRESHOLD = 1      # 积压归零 → 缩
MIN_TOTAL_WORKERS    = 3      # 留底，避免冷启动抖动
MAX_TOTAL_WORKERS    = 4
```

扩容时新建 Ray Actor 抢占 GPU 资源；缩容时优雅退出，等在途任务跑完再释放。

### 3. Ray pipeline 把 CPU 和 GPU 都喂满

朴素实现是「读帧 → 送 GPU → 等结果 → 处理后处理 → 读下一帧」，
GPU 在 CPU 处理时是空闲的。Ray 的三级 pipeline 让三段重叠执行：

```
t0: [preprocess f1] [GPU f1 ] [postprocess f1]
t1: [preprocess f2] [GPU f2 ] [postprocess f2]     ← f2 的预处理与 f1 的 GPU 推理并行
t2: [preprocess f3] [GPU f3 ] [postprocess f3]
```

`PIPELINE_DEPTH = 3` 控制流水线中在途的批次数。

### 4. ONNX + TensorRT FP16

ConvNeXt 用 PyTorch 直接推理时，分类阶段成为整条流水线的瓶颈
（原话是"onnx解决Convnext推理缓慢问题"）。因此：

```
export_onnx.py / export_ecdet_onnx.py   →   .onnx
convert_onnx_fp16.py                    →   *_fp16.onnx
TensorRT                                →   FP16 engine
benchmark_onnx_inf.py                   →   量化收益对比
```

服务侧通过 `TRT_PATH` + `LD_LIBRARY_PATH` 让 onnxruntime 走 TensorRT Execution Provider。
**注意 `TRT_PATH` 必须在 `import onnxruntime` 之前设置**，否则 provider 注册失败。

### 5. JPEG 传输而非原始 ndarray

阶段间传递的是图像数据。原始 BGR ndarray 经 Ray object store 序列化后体积很大，
容易打满 `RAY_OBJECT_STORE_MEMORY` 并触发 spill（磁盘换出）。
改为在生产者侧编码成 JPEG 字节流、消费者侧解码，用一点 CPU 换取对象存储压力的显著下降。

### 6. 合并裁剪与分类，减少一次跨阶段往返

裁剪出的手部图原本要落盘再由分类服务读取，产生一次磁盘 IO 与一次 Kafka 往返。
`hand_crop_service_*` 把两步合进同一进程，直接内存传递。

---

## 脚本演进对照

如果你想理解每一步优化的动机，按下面的顺序读代码即可：

| 环节 | 演进路径 | 每次解决什么 |
| --- | --- | --- |
| 检测服务 | `api_service` → `_kafka` → `_kafka_ray` → `_ray_multi` → `_ray_multi_jpeg` | HTTP → 消息队列解耦 → Ray 管 GPU → 多卡多进程 → 流水线 + JPEG + 动态扩容 |
| 关键点 | `monitor_hold_rope` → `_kafka` → `_kafka_stream` → `_ray` → `_ray_multi` → `_ray_multi_jpeg` | 目录轮询 → Kafka → 流式避免全量落盘 → Ray → 多卡 → 三级 pipeline |
| 分类 | `hand_crop_service` → `_kafka` → `_kafka_onnx` → `_ray` → `_ray_multi` → `_ray_multi_jpeg` | 目录版 → Kafka → ONNX 提速 → Ray → 多卡 → 流水线并行 |

对照关系：**同后缀的三个脚本属于同一次迭代**，即 `*_ray_multi_jpeg` 是检测/关键点/分类
三个服务的当前最优形态，可以配套使用。

---

## 注意事项

### 关于第三方代码与版权

本仓库**不含** EdgeCrafter、mmpose、Awesome-Backbones、TensorRT 的任何源码或权重，
仅包含作者在其之上编写的服务与调度脚本。使用前请分别获取上游项目并遵守其许可证：

- mmpose：Apache-2.0（OpenMMLab）
- Awesome-Backbones：见其仓库 LICENSE
- EdgeCrafter：见其仓库 LICENSE
- TensorRT：NVIDIA 专有许可，**不要**把 SDK 二进制或生成的 engine 提交到公开仓库

深度学习模型权重（`.pth` / `.onnx` / `.engine`）通常体积巨大且可能带训练数据痕迹，
已通过 `.gitignore` 排除。

### 关于凭据与内部路径

- 脚本中保留了作者服务器的绝对路径（`/data2/caimoyao/...`）。这不是凭据，但**暴露了目录约定**，
  按[上游依赖与目录约定](#上游依赖与目录约定)一节改成你自己的路径即可。
- 本仓库**不含**任何 API Key、密码、Token 或 Cookie。
- Kafka 默认监听 `localhost`，**没有任何认证**。请勿把 9094 端口暴露到公网。

### 关于运行成本

- 原环境为多张 A100 80G。`WORKERS_PER_GPU` 会根据空闲显存动态计算（每 10G 开 1 个 worker），
  小显存机器上务必调低，否则会 OOM。
- 三个服务若共用同一张卡，需要错开 `GPU_IDS`，否则会互相抢显存。

---

## License

本仓库中作者编写的脚本采用 MIT License。
第三方依赖与框架遵循其各自许可证，详见[注意事项](#注意事项)。
