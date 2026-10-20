# world-hand-gen-next

这是一个面向世界坐标系手部生成的运行时项目。HOT3D 数据、训练检查点和外部模型资产需要单独准备，不包含在源码压缩包中。

## 部署流程

从源码压缩包解压后的项目根目录执行以下步骤。所有机器相关路径通过 `WHG_*` 环境变量配置，不需要修改 Python 源码：

```bash
export PROJECT_ROOT="$(pwd)"
export WHG_DATA_ROOT="<data-root>/hot3dclip"
export WHG_EGODEX_ROOT="<data-root>/egodex"             # 使用 EgoDex 时设置
export WHG_CACHE_ROOT="<cache-root>/world-hand-gen"
export WHG_EXPERIMENT_ROOT="<experiment-root>/world-hand-gen"
```

第三方源码由引导脚本准备。默认目录为：

```text
<project-root>/third_party/hamer/
<project-root>/third_party/vggt-omega/
<project-root>/third_party/mano/assets/mano/
```

执行：

```bash
./scripts/bootstrap_third_party.sh
./scripts/setup_main_env.sh
./scripts/setup_hamer_env.sh
./scripts/setup_vggt_omega_env.sh
```

`bootstrap_third_party.sh` 只获取第三方源代码和创建目录骨架，不覆盖已有的非空目录，也不自动下载受许可证保护的 MANO、模型 checkpoint、检测器权重或 ViTPose 权重。仓库地址和版本可以通过以下环境变量覆盖：

- `WHG_HAMER_REPO_URL`、`WHG_HAMER_REPO_REF`
- `WHG_VGGT_REPO_URL`、`WHG_VGGT_REPO_REF`
- `WHG_VITPOSE_REPO_URL`、`WHG_VITPOSE_REPO_REF`

### HaMeR 安装前置

运行 `scripts/setup_hamer_env.sh` 前，请先阅读 HaMeR 上游安装问题说明，并按目标机器的 CUDA、PyTorch 和依赖版本处理其中的前置要求：

<https://github.com/geopavlakos/hamer/issues/12>

该脚本会安装 HaMeR、ViTPose 和 Python 依赖，但不会替用户下载或接受需要单独许可的模型资产。

### 外部资产与解释器

将许可允许使用的资产放置在统一路径，或者用环境变量指定其他位置：

```bash
export WHG_MANO_ROOT="<project-root>/third_party/mano/assets"
export WHG_HAMER_CHECKPOINT="<hamer-assets>/hamer.ckpt"
export WHG_VGGT_CHECKPOINT="<vggt-assets>/vggt_omega_1b_512.pt"
export WHG_DETECTOR_CHECKPOINT="<detector-assets>/model_final_f05665.pkl"
export WHG_HAMER_PYTHON="<hamer-python>"
export WHG_DETECTOR_PYTHON="<detector-python>"
export WHG_VGGT_PYTHON="<vggt-python>"
```

如果使用脚本创建的 Conda 环境，可以使用脚本输出的解释器路径设置上述变量。也可以通过 `CONDA_EXE`、`CONDA_ENV_ROOT`、`MAIN_ENV_NAME`、`HAMER_ENV_NAME` 和 `VGGT_ENV_NAME` 调整 Conda 位置和环境名称。

### 环境检查

安装依赖并放置资产后运行：

```bash
./scripts/check_environment.sh
```

检查脚本会显示每个 `WHG_*` 配置解析后的实际路径，并检查：

- 主环境、HaMeR 环境和 VGGT 环境；
- HaMeR、ViTPose、VGGT 和 MANO 源码目录；
- MANO 左右手文件；
- HaMeR checkpoint、配置、均值参数和 ViTPose 权重；
- 检测器权重和 VGGT checkpoint；
- 三个外部 Python 解释器是否可执行；
- HOT3D 数据根目录和 `train_aria` 子集。

缺失项会同时显示对应的环境变量或修复脚本。数据目录必须只读，缓存和实验输出必须位于数据目录之外。

## 路径与数据要求

HOT3D 数据根目录应包含：

```text
<data-root>/hot3dclip/clip_splits.json
<data-root>/hot3dclip/train_aria/clip-<clip-id>/
```

每个 HOT3D clip 的每帧至少需要 `*.image_214-1.jpg`、`*.hands.json`、`*.cameras.json` 和 `*.info.json`。EgoDex 需要配对的 `<episode>.mp4` 和 `<episode>.hdf5` 文件。详细 adaptor 输入、统一 `WindowSample` 字段和坐标/单位约定见 [`src/data/README.md`](src/data/README.md)。

数据检查命令必须从项目根目录执行：

```bash
python -m src.train.validate_data --split train
```

第三方目录、数据目录、缓存目录、实验目录和模型权重相互分离。不要把数据或大模型权重复制到源码压缩包中。

## 四卡训练与独立条件设备

下面的命令使用物理 GPU 0–3 作为训练进程，GPU 4–7 作为 HaMeR、检测器和 VGGT 条件处理进程。根据机器 GPU 数量修改 `CUDA_VISIBLE_DEVICES`、`--nproc_per_node`、`--condition-device-offset` 和 `--global-batch-size`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=4 \
  -m src.train.train \
  --device cuda \
  --condition-device-offset 4 \
  --global-batch-size 16 \
  --expected-world-size 4 \
  --save-every 2000 \
  --eval-after-checkpoint
```

训练和条件提取的仓库、权重、解释器及缓存路径由 `src/config.py` 从 `WHG_*` 环境变量读取。`--global-batch-size` 必须能被进程数整除。

## 并行评估

评估固定使用 `test244` 划分，并对每个片段使用两个 64 帧窗口。训练检查点和评估输出目录由命令行指定，其他数据、缓存和外部模型路径由 `WHG_*` 环境变量提供：

```bash
python -m src.eval.run \
  --checkpoint "<checkpoint>" \
  --output-dir "<evaluation-output>" \
  --device cuda:0
```

命令会在指定输出目录写入 `request.json`、`predictions.pt`、`records.csv` 和 `summary.json`。如需不重新推理、只使用已保存的预测结果重算指标：

```bash
python -m src.eval.recompute \
  --predictions "<evaluation-output>/predictions.pt" \
  --output-dir "<recomputed-output>"
```

评估和训练都使用适配器构造好的规范 `target_camera`，不会直接使用原始 `camera_extrinsics`。

## 数据适配器与损失函数的训练命令

训练通过 `--dataset` 选择适配器，通过可以重复传入的 `--dataset-config key=value` 传递数据集专属配置；默认数据、缓存和实验路径从 `WHG_*` 环境变量读取。

### HOT3D 完整流匹配训练

`hot3d` 是默认数据集。不提供 `--losses` 时，使用以下默认权重：

```text
flow=1.0,camera_flow=1.0,joints3d=0.5,acceleration=0.02,camera_pose=0.1,camera_trajectory=0.1
```

```bash
python -m src.train.train \
  --dataset hot3d \
  --split train \
  --losses flow=1.0,camera_flow=1.0,joints3d=0.5,acceleration=0.02,camera_pose=0.1,camera_trajectory=0.1 \
  --max-steps 40000 \
  --global-batch-size 16
```

各损失项的监督要求如下：

| 损失项 | 作用 | 正权重启用时适配器必须提供 |
| --- | --- | --- |
| `flow` | MANO 平移、根部旋转和局部旋转的流匹配状态误差 | `target_mano` |
| `camera_flow` | 相机 6D 旋转和平移的流匹配误差 | 规范的 `target_camera` |
| `joints3d` | 解码 MANO 后的三维关节误差 | `target_mano`，或 `hand_joints3d`/`target_joints3d` 和 `joint_valid` |
| `acceleration` | MANO 关节单位时间二阶差分误差 | `target_mano` |
| `camera_pose` | 相机姿态的测地线旋转误差与平移误差 | 规范的 `target_camera` |
| `camera_trajectory` | 相邻帧相机姿态增量误差 | 规范的 `target_camera` |

### 非 HOT3D 数据适配器的三维关节训练

当前 `egodex` 适配器提供外部三维关节监督。非 HOT3D 入口仍使用 MMDiT 预测 `ManoFlowState`，只允许不依赖 MANO 真值的正权重 `joints3d`；该损失比较解码后的 MANO joints 与外部关键点：

```bash
python -m src.train.train \
  --dataset egodex \
  --dataset-config root="${WHG_EGODEX_ROOT}" \
  --dataset-config split=train \
  --dataset-config episode=<part>/<task>/<episode> \
  --losses joints3d=1.0 \
  --max-steps 100 --global-batch-size 1 --window 16 \
  --device cpu \
  --experiment-dir "${WHG_EXPERIMENT_ROOT}/egodex-smoke"
```

### 从 3D joints 拟合 MANO 参数

`src.model.wrappers.mano_fit.fit_joints_to_mano` 用已经初始化的 MANO 解码器和 3D 关节标注拟合 MANO pose/translation，适合只有 3D keypoint 标注、需要生成统一 `target_mano` 的数据适配器。该接口不读取或写入数据集文件，也不改变训练流程。

输入和输出遵循项目统一的 MANO 格式：

- 输入 `joints3d`：`[B, 2, T, 16, 3]`，手顺序为 `left, right`；16 点必须使用 `JOINT_ORDER`：`wrist`，然后依次是 `thumb`、`index`、`middle`、`ring`、`little` 的各 3 个关节。
- 可选 `initial_state`：`ManoFlowState`，形状为 translation `[B,2,T,3]`、root rotation `[B,2,T,6]`、local rotation `[B,2,T,15,6]`，作为优化初值。
- 输出：`ManoFlowState`，rotation 使用连续 6D 表示，形状与 `initial_state` 相同。
- 可选 `joint_valid`：`[B,2,T,16]`；无效或 NaN 关节不会参与拟合。没有任何有效点时接口会报错。
- 默认优化参数为 `steps=20`、`lr=0.05`，也可以通过函数参数覆盖；推荐的单手批量配置为 `batch_size=64`。
- MANO shape/betas 不参与拟合，沿用输入模型的固定 shape；输出保持输入 3D joints 的坐标系和单位。

```python
import torch
from src.model.flow.state import ManoFlowState
from src.model.wrappers import ManoWrapper, fit_joints_to_mano

mano = ManoWrapper()  # MANO layer 在首次 forward 时按输入设备和 dtype 加载
joints3d = torch.zeros(1, 2, 16, 16, 3)  # [B, 2, T, 16, 3]
target_mano: ManoFlowState = fit_joints_to_mano(
    mano,
    joints3d,
    steps=20,
    lr=0.05,
)
```

拟合内部使用 MANO 原生 16 点顺序与公共 `JOINT_ORDER` 的显式映射，并优化 root rotation、15 个局部 rotation 和 translation；不要把公共顺序直接当作 MANO 原生顺序传入模型。

## 运行时源码压缩包

使用当前工作树构建源码压缩包：

```bash
./scripts/package_runtime.sh
```

压缩包包含源码、测试、文档和安装脚本，但明确不包含：

- `third_party/` 第三方源代码；
- 数据目录；
- 训练或评估 checkpoint；
- 缓存、实验输出、`.git/`、Python 字节码和符号链接。

解压后先运行 `scripts/bootstrap_third_party.sh`，再运行三个环境安装脚本。这样源码包可以在不同用户、不同服务器和不同项目目录下部署，不依赖打包机器的本地路径。
