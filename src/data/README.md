# 数据集与 Adaptor

本目录把不同数据集转换成训练代码统一消费的窗口样本。训练循环不直接读取数据集文件，而是通过 `src/data/adapters.py` 中的 adaptor 创建器 `make_adapter(name, **config)` 选择数据集，再调用：

```python
sample = adapter.sample(seed=seed, batch_size=batch_size, window=window)
```

当前已注册：

| adaptor 名称 | 实现 | 数据根路径 | 默认划分 | 适合的训练入口 |
| --- | --- | --- | --- | --- |
| `hot3d` | `Hot3DAdapter` | `src.config.DATA_ROOT` | `train` | `python -m src.train.train`，完整 MANO + 相机 flow 训练 |
| `egodex` | `EgoDexAdapter` | `src.config.EGODEX_ROOT` | `test` | `python -m src.train.configured` 或 `src.train.train --dataset egodex`，当前为 3D joints 目标验证 |

数据集路径、缓存和实验输出统一在 `src/config.py` 配置，并通过 `WHG_*` 环境变量覆盖；不要把机器相关的绝对路径写进适配器、训练循环或文档。数据目录只读；预处理缓存应放在 `CACHE_ROOT` 等数据目录之外。第三方源码由 `scripts/bootstrap_third_party.sh` 获取，源码压缩包不包含第三方 checkout。

## 统一窗口契约

所有 adaptor 都必须返回 `src.data.contract.WindowSample`。除 `metadata` 外，字段必须保持下面的形状约定；`B` 是 batch size，`T` 是窗口帧数：

| 数据 / 字段 | 必需性 | 形状与类型 | 标准含义 |
| --- | --- | --- | --- |
| `hamer_tokens` | 必需 | `float[B, 2, T, 1024]` | 每只手、每帧的 HaMeR 条件 token；不存在时用同形状零张量，并在 `hamer_valid` 中标记无效。 |
| `hamer_valid` | 必需 | `bool[B, 2, T]` | HaMeR token 是否可用；顺序为 `left, right`。 |
| `vggt_tokens` | 必需 | `float[B, T, 17, 2048]` | 每帧 VGGT 相机条件 token；`17` 是 VGGT token 数。 |
| `vggt_valid` | 必需 | `bool[B, T]` | 该帧 VGGT token 是否可用。 |
| `keypoints` | 必需 | `float[B, 2, T, 21, 3]` | 每只手的 2D 关键点，最后一维为 `(x, y, confidence)`；坐标必须与 adaptor 的图像像素约定一致，训练前通常归一化到 `[0, 1]`。 |
| `bbox` | 必需 | `float[B, 2, T, 4]` | 每只手的 `xyxy` 框，顺序为 `(x1, y1, x2, y2)`；与 `keypoints` 使用同一图像坐标系，训练前通常按图像边长归一化。 |
| `bbox_valid` | 必需 | `bool[B, 2, T]` | 手框是否存在且有效。缺失框的数值可以为零，但必须置为 `False`。 |
| `images` | 可选 | `float[B, T, 3, H, W]` | RGB 图像，通道优先；数值范围通常为 `[0, 1]`。图像旋转或矫正属于数据集 adaptor 的责任。 |
| `camera_extrinsics` | 可选 | `float[B, T, 4, 4]` | 原始相机外参，统一语义为 `x_world = E_t @ x_source_camera`（`world_from_camera`）；保留数据集 source camera 轴，不做首帧对齐。它与 `target_camera` 不是同一字段语义。 |
| `camera_intrinsics` | 可选 | `float[B, T, 3, 3]` | 与 `images` 像素坐标和图像方向匹配的内参；HOT3D 是 rectified/upright pixels，EgoDex 是 original video pixels。 |
| `camera_valid` | 可选 | `bool[B, T]` | 相机标注是否有效。 |
| `hand_joints3d` / `target_joints3d` | 按目标需要 | `float[B, 2, T, 16, 3]` | 外部 3D 手部关节监督；两个字段是兼容别名，至少设置一个。关节顺序必须是 `JOINT_ORDER`：`wrist`，随后按 `thumb/index/middle/ring/little` 各 3 个非指尖关节。必须与 `target_mano` 使用同一窗口首帧 model-camera 坐标系；单位在 metadata 中声明。 |
| `joint_valid` | 与 3D 关节同时提供 | `bool[B, 2, T, 16]` | 每个 3D 关节点是否有效。无效点可以保留 NaN，但必须在 mask 中置为 `False`。 |
| `joint_confidence` | 可选 | `float[B, 2, T, 16]` | 数据集原生关节置信度；没有原生置信度时使用 `None`，不要伪造置信度。 |
| `target_mano` | 按 `flow` / MANO 目标需要 | `ManoFlowState`：translation `[B,2,T,3]`、root rotation `[B,2,T,6]`、local rotation `[B,2,T,15,6]` | MANO flow 监督。旋转使用连续 6D 表示；坐标系、单位和缺失帧语义必须在 `metadata` 说明。 |
| `target_camera` | 按相机 loss 需要 | `CameraFlowState`：rotation `[B,T,6]`、translation `[B,T,3]` | 唯一统一的相机 flow 监督：`U_t` 满足 `x_first_model_camera = U_t @ x_current_model_camera`，首帧 `U_0 = I`、平移为零。无轴变换时 `U_t = inverse(E_0) @ E_t`；有固定模型相机轴变换 `F` 时 `U_t = F @ inverse(E_0) @ E_t @ inverse(F)`。rotation 为 `U_t` 的连续 6D，translation 使用同一 adaptor 的相机单位。 |
| `metadata` | 建议 | `dict[str, Any]` | 数据集名称、split、样本/帧 ID、坐标系、单位、外参语义、图像变换和 fallback 信息。它是复现实验和排查坐标错误的唯一公开说明入口。 |

`WindowSample.__post_init__` 会把 `hand_joints3d` 与 `target_joints3d` 互相补成别名，但 adaptor 仍应优先填充规范字段 `hand_joints3d`。`camera_extrinsics` 始终是 raw `world_from_camera`；所有 camera loss 只消费 canonical `target_camera`，不会自动从 raw 外参重新对齐。

## 接入一个新数据集

1. 在 `src/data/<dataset>/` 增加只读加载器，负责文件配对、连续帧窗口、缺失标注和原始坐标读取。加载器不要修改挂载的数据文件。
2. 在 `src/data/adapters.py` 增加一个 dataclass adaptor。它负责 split、episode/clip 选择、随机窗口和数据集专属的图像/标注变换；训练循环不应包含这些分支。相机目标必须调用 `src/data/camera.py` 的统一变换，不能自行定义另一套首帧方向。
3. 将 adaptor 注册到 `make_adapter()`：

   ```python
   if name == "my_dataset":
       return MyDatasetAdapter(**config)
   ```

4. 返回完整的 `WindowSample`。没有某类条件时返回正确形状的零张量和全 `False` mask；没有监督目标时返回 `None`，不要用零目标冒充标注。
5. 在 `describe()` 和 `metadata` 中写清楚 split、样本 ID、坐标系、单位、外参方向、图像变换和关节顺序。
6. 为加载器、窗口边界、缺失标注、坐标变换和 adaptor 注册增加 `tests/data/` 下的测试。
7. 先用单窗口采样验证形状，再启动训练：

   ```bash
   python -m src.train.configured \
     --dataset my_dataset \
     --dataset-config root=/path/to/dataset \
     --losses joints3d=1.0 \
     --steps 1 --batch-size 1 --window 8 \
     --device cpu --output-dir /tmp/my-dataset-smoke
   ```

`parse_key_values()` 支持 `key=value`，整数会解析为 `int`，`none`/`null` 会解析为 `None`，其他值保留为字符串；复杂配置应放在 `src/config.py` 或 adaptor 自己的配置结构中。

## HOT3D adaptor

`Hot3DAdapter` 使用 `DATA_ROOT/train_aria/clip-<clip_id>/`，每帧至少需要：

- `*.image_214-1.jpg`：RGB 相机图像；
- `*.hands.json`：左右手框和 `mano_pose`；完整 flow 训练要求窗口内左右手都有 MANO 标注；
- `*.cameras.json`：RGB 相机标定和 `T_world_from_camera`；
- `*.info.json`：原始帧附加信息；
- 根目录 `clip_splits.json`：官方 Aria clip 列表。

当前只使用 `train_aria`。图像先经 FISHEYE624 到线性针孔相机的矫正，再顺时针旋转 90 度；图像、手框和内参使用 rectified/upright 像素坐标。raw `camera_extrinsics` 仍是 source-camera 轴上的 `world_from_camera`，不能直接和 upright 内参配套投影。模型相机目标统一为 `current_model_camera_to_first_model_camera`：`U_t = F @ inverse(E_0) @ E_t @ inverse(F)`，其 reference frame 是窗口首帧 upright camera；MANO 和 3D joints 也使用该首帧模型相机坐标，单位为 `m`。

默认 `train` / `test` split 由 `src/data/hot3d/splits.py` 对挂载的 Aria clip 做确定性 1272/244 划分；也可以用 `clip_id=000001` 固定单个 clip，用 `window_start=<int>` 固定窗口起点。

## EgoDex adaptor

`EgoDexAdapter` 使用 `EGODEX_ROOT` 下的配对文件：

```text
<root>/<part>/<task>/<episode>.mp4
<root>/<part>/<task>/<episode>.hdf5
```

`part1`–`part5` 是 `train`，`test` 是 `test`。HDF5 至少需要 `transforms/camera`、手部 `transforms/<side><joint>`；可选的 `confidences/<side><joint>` 会进入 `joint_confidence`。视频帧数必须与标注一致，窗口必须是连续帧。EgoDex 保留原始 RGB、原始 ARKit world 坐标和数据集原始单位，不做图像旋转、投影或 MANO 拟合；raw 相机矩阵是 `world_from_camera`，模型相机目标统一为 `current_model_camera_to_first_model_camera`：`U_t = inverse(E_0) @ E_t`，reference frame 是窗口首帧 source camera。3D joints 同样先转换到该首帧 camera 坐标。没有可靠内参时使用代码中的 fallback，并在 metadata 标出 `intrinsic_fallback`。

该 adaptor 当前提供 3D 关节和相机几何、但条件 token 是显式的零张量，且不提供 `target_mano`。因此当前训练入口对非 HOT3D 数据集只允许正权重 `joints3d`，例如：

```bash
python -m src.train.train \
  --dataset egodex \
  --dataset-config root="${WHG_EGODEX_ROOT}" \
  --dataset-config split=train \
  --dataset-config episode=part1/pick/0 \
  --losses joints3d=1.0 \
  --max-steps 100 --global-batch-size 1 --window 16 \
  --device cpu --experiment-dir /tmp/egodex-smoke
```

如果新数据集要启用 MANO、相机或条件模型 loss，必须先在 adaptor 中提供对应的 target/condition，并同步修改训练入口的能力检查和测试；不能只在命令行添加 loss 名称。
