# 数据与全局不变量

- 所有数据集、缓存、实验输出、处理进程解释器和第三方资产路径统一从 `src/config.py` 读取；机器相关路径通过 `WHG_*` 环境变量配置，禁止写入用户或服务器绝对路径。
- 数据集挂载目录只读，不能修改任何原始数据内容；预处理缓存必须放在数据目录之外。
- 当前已接入 `hot3d` 和 `egodex` 两个数据适配器。HOT3D 位于 `DATA_ROOT/train_aria`，只使用 `train_aria` 子集；EgoDex 位于 `EGODEX_ROOT`。
- `train_aria` 图像必须经过顺时针 90 度旋转才能正常显示；手框、内参和几何标注必须同步执行相同的图像坐标变换。
- 所有数据适配器的原始 `camera_extrinsics` 统一表示 `x_world = E_t @ x_source_camera`，即 `world_from_camera`，不做首帧对齐。
- 训练和评估使用的 `target_camera` 统一表示窗口首帧模型相机坐标系中的当前相机变换：`x_first_model_camera = U_t @ x_current_model_camera`。首帧 `U_0` 为单位变换、平移为零；没有固定轴变换时 `U_t = inverse(E_0) @ E_t`，有固定轴变换 `F` 时 `U_t = F @ inverse(E_0) @ E_t @ inverse(F)`。
- `hand_joints3d` 必须与 `target_mano`（如果提供）使用同一首帧模型相机坐标和声明单位；图像/内参与原始外参的相机轴不一致时，必须在元数据中显式标记。
- 数据适配器的接入规范、统一 `WindowSample` 字段表和各数据集文件标准见 [`src/data/README.md`](src/data/README.md)。第三方源代码由 `scripts/bootstrap_third_party.sh` 获取，源码压缩包不包含第三方 checkout。

# 项目结构

功能代码统一存放在 `src/`，主要结构如下：

- `config.py`：数据、缓存、实验输出、MANO/HaMeR/VGGT 仓库、检查点和外部处理进程解释器路径的唯一配置入口；外部路径由 `WHG_*` 环境变量覆盖。
- `data/`：数据集原始读取、窗口采样和数据适配器。
  - `contract.py`：数据集无关的 `WindowSample` 和 `JOINT_ORDER` 公共契约。
  - `camera.py`：原始 `world_from_camera` 到统一首帧模型相机目标的变换，以及三维点坐标转换。
  - `adapters.py`：`Hot3DAdapter`、`EgoDexAdapter` 和 `make_adapter()` 注册表；训练循环只依赖这里暴露的数据契约。
  - `hot3d/`：HOT3D Aria 原始帧、相机/手部标注、矫正、手框和缓存。
    - `__init__.py`：原始帧加载、FISHEYE624 矫正、顺时针 90 度正向变换和 MANO 延迟加载。
    - `dataset.py`：`Hot3DDataset` 延迟加载矫正图像、手框和相机/MANO 目标。
    - `cache.py`：HaMeR 逐帧特征、手框和评估固定窗口 VGGT 特征缓存。
    - `splits.py`：基于 `train_aria` 的确定性 1272/244 训练/测试片段划分。
    - `handbox.py`：检测器手框和关键点的批处理与缓存。
  - `egodex/`：EgoDex 的 MP4/HDF5 配对校验、连续帧窗口以及 OP-16 关节/相机读取。
  - `README.md`：数据适配器接入规范、统一字段表、数据文件标准和冒烟测试命令。
- `model/`：生成模型和外部模型封装器。
  - `wrappers/`：HaMeR、检测器、VGGT、MANO 等外部模型封装器。
  - `flow/`：流匹配状态、条件编码、手部/相机生成器、联合生成器和采样器。
- `loss/losses.py`：流匹配、三维关节、加速度、相机姿态和相机轨迹损失。
- `geometry/rotation.py`：6D 旋转、轴角、四元数转换与测地线距离。
- `train/`：训练入口和数据/特征处理进程。
  - `train.py`：HOT3D 完整流匹配训练、分布式数据并行、AdamW、检查点恢复和检查点后评估。
  - `configured.py`：数据集无关的三维关节适配器冒烟测试和验证入口。
  - `runner.py`：`key=value` 数据适配器配置和 `name=weight` 损失配置解析。
  - `sample.py`：HOT3D 窗口采样、HaMeR/VGGT 条件和训练目标构造。
  - `flow_sampling.py`：噪声和流匹配时间采样。
  - `step.py`：单步前向计算和加权损失；反向传播由外层训练循环执行。
  - `feature_worker.py` / `detector_worker.py`：在外部 conda 环境中常驻加载条件模型。
  - `validate_data.py`：训练前 `train_aria` 连续性检查，不在训练循环中执行。
- `eval/`：固定 `test244` 评估入口、预测结果协议、离线指标重算、评估队列以及手部/相机/运动指标。
  - `run.py`：执行固定评估窗口并保存预测结果文件。
  - `recompute.py`：从 `predictions.pt` 离线重新计算指标，不重新推理。
  - `metric/`：相机、世界关节、运动、对齐和通用指标实现。
- `visualization/`：预测、关节、相机轨迹、网格和图像叠加可视化工具。
- `scripts/`：第三方源码获取、运行时压缩包构建、环境检查和外部模型环境安装辅助脚本。
  - `bootstrap_third_party.sh`：按可覆盖的仓库地址和版本获取第三方源码，保留已有 checkout，不下载受许可保护的模型权重。
  - `package_runtime.sh`：构建并校验不含第三方源码、数据、检查点和符号链接的源码压缩包。
  - `check_environment.sh`：检查主环境和外部资产是否满足运行条件。
  - `setup_main_env.sh`、`setup_hamer_env.sh`、`setup_vggt_omega_env.sh`：安装主环境及外部模型环境。
- `docs/`：补充设计、运行记录和交付说明，不替代根 README 中的可执行命令。
  - `metric/README.md`：评估指标和相机/关节误差的解释。

# README 与文档位置

- 根目录 [`README.md`](README.md)：部署与路径配置、数据检查、训练命令、数据适配器和损失函数的启用方式、多卡参数、评估和运行时压缩包说明。
- [`src/data/README.md`](src/data/README.md)：每个数据集适配器的接入方法、所需数据、统一窗口字段及每条数据的标准含义。
- [`docs/`](docs/)：补充设计、运行记录和交付说明。
- 本文件 [`AGENTS.md`](AGENTS.md)：面向代码协作者的项目结构、数据不变量、统一相机语义和文档索引。

# 测试约定

测试存放在 `tests/`，目录结构对应 `src/`：

- `tests/data`：数据加载器、相机变换、窗口契约和数据适配器。
- `tests/train`：采样、flow time、训练 step、joint flow 和训练入口。
- `tests/loss`：MANO、关节、相机和运动损失。
- `tests/model`：流匹配、注意力、条件编码、MANO 封装器和生成模型。
- `tests/eval`：评估协议、指标、队列、重算和运行入口。
- `tests/geometry`：旋转和几何变换。
- `tests/visualization`：图像叠加、轨迹和可视化几何。

新增数据适配器时，至少要增加窗口边界、缺失标注、`E_0`/`E_t` 相机变换方向、首帧单位变换、三维关节与目标坐标系一致性以及注册入口测试。测试不应修改真实数据挂载目录。

# 编码规范

- 未复用的逻辑不要提取成函数。
- 一次性的可视化和验证命令不要写进文件，直接通过命令行运行。
- 数据集、缓存、实验输出和第三方资产路径不要散落在业务代码中，统一从 `src/config.py` 获取。
- 数据适配器负责数据集专属的划分、窗口、图像变换、标注变换和元数据；训练循环不应增加数据集分支。
- 不要把缺失标注伪造成零目标；缺失条件使用正确形状的零张量和 `False` mask，缺失监督目标使用 `None`。
