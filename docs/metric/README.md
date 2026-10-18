# 评估指标规范（Metrics Specification）

本文档以论文级严格性定义 `src/eval` 中全部评估指标。每条指标给出：输入约定、对齐算子、数学定义、单位。公式与实现一一对应（实现位置随条目标注）。

---

## 1. 记号与约定

设评估窗口长度 $T=64$，双手集合 $\mathcal{H}=\{\mathrm{L},\mathrm{R}\}$，MANO 关节数 $J=16$，局部关节数 $K=15$。

| 符号 | 含义 | 形状 |
|---|---|---|
| $\hat{X}_{h,t,j}\in\mathbb{R}^3$ | 预测 MANO 关节位置（米） | $[2,T,J,3]$ |
| $X_{h,t,j}\in\mathbb{R}^3$ | 真值 MANO 关节位置（米） | $[2,T,J,3]$ |
| $\hat{p}_{h,t},\,p_{h,t}\in\mathbb{R}^3$ | 预测 / 真值最终 MANO 解码腕关节（`joints[...,0,:]`）位置（米） | $[2,T,3]$ |
| $\hat{R}_{h,t},\,R_{h,t}\in SO(3)$ | 预测 / 真值 root 全局旋转（由 6D 表示经 Gram–Schmidt 恢复） | $[2,T,3,3]$ |
| $\hat{R}^{\mathrm{loc}}_{h,t,k},\,R^{\mathrm{loc}}_{h,t,k}\in SO(3)$ | 预测 / 真值第 $k$ 个局部关节旋转 | $[2,T,15,3,3]$ |
| $\hat{C}_t,\,C_t\in SO(3)$ | 预测 / 真值相机旋转（相对窗口首帧相机的位姿） | $[T,3,3]$ |
| $\hat{c}_t,\,c_t\in\mathbb{R}^3$ | 预测 / 真值相机平移（米） | $[T,3]$ |
| $x_{h,t},\,\hat{x}_{h,t}$ | 手部关节集合的简写：$\{X_{h,t,j}\}_j$、$\{\hat{X}_{h,t,j}\}_j$ | $[J,3]$ |

所有量均定义在**窗口首帧相机 upright 参考系**中（数据集顺时针 90° 直立约定后，以第一帧相机位姿为参考）。位置单位为米；报告值乘以 $10^3$ 转为毫米（mm），旋转以度（deg）报告。约定 $\hat{(\cdot)}$ 为预测、无帽为真值；$t=0$ 为窗口首帧。实现中 $p_{h,t}$ 固定取外部 MANO 前向得到的 `joints[...,0,:]`，不是 `ManoFlowState.translation` 或 MANO 的 `transl` 输入偏置。

**旋转测地距离**（`src/geometry/rotation.py: geodesic_distance`）：

$$
d_{\mathrm{SO(3)}}(R_1,R_2)\;=\;\arccos\!\left(\frac{\operatorname{tr}\!\left(R_1^{\top}R_2\right)-1}{2}\right)\in[0,\pi],
\qquad
d_{\mathrm{deg}}(R_1,R_2)=\frac{180}{\pi}\,d_{\mathrm{SO(3)}}(R_1,R_2).
\tag{1}
$$

---

## 2. 首帧对齐算子（`metric/align.py`）

所有 "absolute" 类指标对预测与真值**各自**施加首帧对齐（两侧对称、无跨序列拟合），使两条序列从统一原点出发：

**定义 1（平移对齐）**　对任意序列 $y_{h,t}\in\mathbb{R}^3$ 及其首帧值 $y_{h,0}$：

$$
\mathcal{T}(y)_{h,t} \;=\; y_{h,t}-y_{h,0}.
\tag{2}
$$

**定义 2（刚体对齐）**　给定首帧旋转 $Q_{h,0}\in SO(3)$ 与平移 $q_{h,0}$：

$$
\mathcal{R}(y;q)_{h,t} \;=\; Q_{h,0}^{\top}\bigl(y_{h,t}-q_{h,0}\bigr).
\tag{3}
$$

**定义 3（正最小二乘尺度）**　对同型张量 $(P,G)$（元素视为向量集合），拟合单一正标量

$$
s^*(P,G)\;=\;\frac{\langle P,G\rangle}{\lVert P\rVert_2^{2}+\varepsilon},\qquad \varepsilon=10^{-12},
\tag{4}
$$

其中 $\langle\cdot,\cdot\rangle$ 与 $\lVert\cdot\rVert_2$ 为全体坐标元素上的内积与范数。$s^*$ 用于 "scaled" 变体，衡量预测运动幅度与真值的相对偏差；它**不是**逐帧或逐关节的拟合，而是全窗口单一标量。

---

## 3. 手部世界系指标（`metric/world.py`）

以下均值域：关节类 $\frac{1}{|\mathcal{H}|TJ}$，root 平移类 $\frac{1}{|\mathcal{H}|T}$，旋转类同理（对 $h,t$ 或 $h,t,k$ 求均值）。

**(M1) MPJPE**　对关节序列施加平移对齐 (2)（对全部关节减去各自手的最终解码腕根首帧位置）：

$$
\mathrm{MPJPE}
=\frac{10^3}{|\mathcal{H}|TJ}\sum_{h,t,j}\Bigl\lVert\bigl(\hat{X}_{h,t,j}-\hat{p}_{h,0}\bigr)-\bigl(X_{h,t,j}-p_{h,0}\bigr)\Bigr\rVert_2
\quad[\mathrm{mm}].
\tag{5}
$$

**(M2) MPJPE-scaled**　先按 (5) 得到对齐序列 $\tilde{X}=\hat{X}-\hat{p}_{h,0}$、$\tilde{X}^{\ast}=X-p_{h,0}$，拟合 $s^*=s^*(\tilde{X},\tilde{X}^{\ast})$ 后：

$$
\mathrm{MPJPE\text{-}s}
=\frac{10^3}{|\mathcal{H}|TJ}\sum_{h,t,j}\Bigl\lVert s^*\,\tilde{X}_{h,t,j}-\tilde{X}^{\ast}_{h,t,j}\Bigr\rVert_2
\quad[\mathrm{mm}].
\tag{6}
$$

**(M3) W-MPJPE**　对关节序列施加刚体对齐 (3)，即 $\hat{X}^{r}_{h,t,j}=\hat{R}_{h,0}^{\top}(\hat{X}_{h,t,j}-\hat{p}_{h,0})$，$X^{r}_{h,t,j}=R_{h,0}^{\top}(X_{h,t,j}-p_{h,0})$：

$$
\mathrm{W\text{-}MPJPE}
=\frac{10^3}{|\mathcal{H}|TJ}\sum_{h,t,j}\bigl\lVert\hat{X}^{r}_{h,t,j}-X^{r}_{h,t,j}\bigr\rVert_2
\quad[\mathrm{mm}].
\tag{7}
$$

**(M4) W-MPJPE-scaled**　在 (7) 的刚体对齐序列上按 (4) 拟合 $s^*$ 后同 (6) 计算。

**(M5) Root translation error**　对最终解码腕根序列施加平移对齐 (2)：

$$
\mathrm{RTE}
=\frac{10^3}{|\mathcal{H}|T}\sum_{h,t}\Bigl\lVert\bigl(\hat{p}_{h,t}-\hat{p}_{h,0}\bigr)-\bigl(p_{h,t}-p_{h,0}\bigr)\Bigr\rVert_2
\quad[\mathrm{mm}].
\tag{8}
$$

即预测与真值之间**不拟合任何变换**；对齐仅为各自去除首帧偏移。

**(M6) Root translation error-scaled**　在 (8) 的对齐序列上拟合 $s^*$（作用于 $2T$ 个平移向量全体）后按 (6) 形式计算。

**(M7) Root rotation error（absolute）**　直接比较旋转矩阵，无对齐：

$$
\mathrm{RRE}
=\frac{1}{|\mathcal{H}|T}\sum_{h,t}d_{\mathrm{deg}}\bigl(\hat{R}_{h,t},\,R_{h,t}\bigr)
\quad[\mathrm{deg}].
\tag{9}
$$

**(M8) W-Root rotation error**　施加首帧旋转对齐（比较相对旋转）：

$$
\mathrm{WRRE}
=\frac{1}{|\mathcal{H}|T}\sum_{h,t}d_{\mathrm{deg}}\bigl(\hat{R}_{h,0}^{\top}\hat{R}_{h,t},\;R_{h,0}^{\top}R_{h,t}\bigr)
\quad[\mathrm{deg}].
\tag{10}
$$

**(M9) Local rotation error**　局部关节旋转对全局刚体变换不变，直接比较：

$$
\mathrm{LRE}
=\frac{1}{|\mathcal{H}|TK}\sum_{h,t,k}d_{\mathrm{deg}}\bigl(\hat{R}^{\mathrm{loc}}_{h,t,k},\,R^{\mathrm{loc}}_{h,t,k}\bigr)
\quad[\mathrm{deg}].
\tag{11}
$$

**(M10) W-Local rotation error**　当前实现与 (11) 恒等（`world.py` 中两键取同一计算结果）；列出仅为保持接口完备。

---

## 4. Procrustes 对齐指标（`metric/aligned.py`）

本节对齐变换**由预测拟合到真值**（跨序列拟合），且拒绝反射。

**定义 4（Kabsch 刚体拟合）**　给定点集 $P,G\in\mathbb{R}^{N\times 3}$（行为点），去质心 $\bar{P},\bar{G}$，令 $H=P_c^{\top}G_c$，SVD 分解 $H=U\Sigma V^{\top}$，则

$$
D=\operatorname{diag}\bigl(1,1,\operatorname{sign}\det(VU^{\top})\bigr),\qquad
R=VDU^{\top}\in SO(3),
\tag{12}
$$

$$
\tau=\bar{G}-\bar{P}R^{\top},\qquad
\mathcal{A}_{\mathrm{rigid}}(P)=P\,R^{\top}+\tau.
\tag{13}
$$

（行向量约定；$\det(VU^{\top})<0$ 时以 $D$ 翻转最后一列，保证 proper rotation。）

**定义 5（Umeyama 相似拟合）**　在 (12) 记号下，记奇异值 $\sigma_1\ge\sigma_2\ge\sigma_3$（对应 $H=U\Sigma V^\top$），则

$$
s=\frac{\sum_i d_i\,\sigma_i}{\lVert P_c\rVert_F^{2}+\varepsilon}\ \ge 0,\qquad d_i = \bigl(D\bigr)_{ii},
\tag{14}
$$

$$
\mathcal{A}_{\mathrm{sim}}(P)=s\,P\,R^{\top}+\tau,
\qquad
\tau=\bar{G}-s\,\bar{P}R^{\top}.
\tag{15}
$$

**(M11) PA-MPJPE**　**逐姿态**（每个 $(h,t)$ 独立）拟合相似变换：

$$
\mathrm{PA\text{-}MPJPE}
=\frac{10^3}{|\mathcal{H}|TJ}\sum_{h,t,j}
\bigl\lVert\mathcal{A}^{(h,t)}_{\mathrm{sim}}(\hat{x}_{h,t})_j-X_{h,t,j}\bigr\rVert_2
\quad[\mathrm{mm}],
\tag{16}
$$

其中 $\mathcal{A}^{(h,t)}_{\mathrm{sim}}$ 由该帧该手的 $J$ 个关节对拟合。

**(M12) WA-MPJPE**　**全局刚体**：每个 batch item（一个窗口）以**单一** Kabsch 变换覆盖全部 $2TJ$ 个点：

$$
\mathrm{WA\text{-}MPJPE}
=\frac{10^3}{|\mathcal{H}|TJ}\sum_{h,t,j}
\bigl\lVert\mathcal{A}^{\mathrm{glob}}_{\mathrm{rigid}}(\hat{X})_{h,t,j}-X_{h,t,j}\bigr\rVert_2
\quad[\mathrm{mm}].
\tag{17}
$$

**(M13) WA-MPJPE-scaled**　同 (17)，全局变换换为单一相似拟合 $\mathcal{A}^{\mathrm{glob}}_{\mathrm{sim}}$（式 14–15）。

---

## 5. 时间运动指标（`metric/motion.py`）

约定帧间隔 $dt=1$（帧为单位；若采集帧率为 $f$ Hz，物理速度为 $f$ 倍数值）。先转毫米再做差分。

**(M14) Velocity error**　一阶前向差分 $v_{h,t,j}=X_{h,t+1,j}-X_{h,t,j}$（$t=0,\dots,T-2$）：

$$
\mathrm{VelErr}
=\frac{1}{|\mathcal{H}|(T-1)J}\sum_{h,t,j}\bigl\lVert\hat{v}_{h,t,j}-v_{h,t,j}\bigr\rVert_2
\quad[\mathrm{mm/frame,~标记为\ mms}].
\tag{18}
$$

**(M15) Acceleration error**　二阶前向差分 $a_{h,t,j}=X_{h,t+2,j}-2X_{h,t+1,j}+X_{h,t,j}$（$t=0,\dots,T-3$）：

$$
\mathrm{AccErr}
=\frac{1}{|\mathcal{H}|(T-2)J}\sum_{h,t,j}\bigl\lVert\hat{a}_{h,t,j}-a_{h,t,j}\bigr\rVert_2
\quad[\mathrm{mm/frame^2,~标记为\ mms^2}].
\tag{19}
$$

**(M16) Endpoint drift**　比较首末相对位移（无对齐）：

$$
\mathrm{Drift}
=\frac{10^3}{|\mathcal{H}|J}\sum_{h,j}
\Bigl\lVert\bigl(\hat{X}_{h,T-1,j}-\hat{X}_{h,0,j}\bigr)-\bigl(X_{h,T-1,j}-X_{h,0,j}\bigr)\Bigr\rVert_2
\quad[\mathrm{mm}].
\tag{20}
$$

---

## 6. 相机指标（`metric/camera.py`）

与第 3 节同构，作用于相机相对位姿 $(\hat{C}_t,\hat{c}_t)$ 与 $(C_t,c_t)$；均值域对 $t$（$B=1$）。

**(C1) Camera rotation error（absolute）**

$$
\mathrm{CRE}=\frac{1}{T}\sum_{t}d_{\mathrm{deg}}\bigl(\hat{C}_t,\,C_t\bigr)\quad[\mathrm{deg}].
\tag{21}
$$

**(C2) W-Camera rotation error**

$$
\mathrm{WCRE}=\frac{1}{T}\sum_{t}d_{\mathrm{deg}}\bigl(\hat{C}_0^{\top}\hat{C}_t,\;C_0^{\top}C_t\bigr)\quad[\mathrm{deg}].
\tag{22}
$$

**(C3) Camera translation error**　平移对齐 (2)：

$$
\mathrm{CTE}=\frac{10^3}{T}\sum_{t}\Bigl\lVert\bigl(\hat{c}_t-\hat{c}_0\bigr)-\bigl(c_t-c_0\bigr)\Bigr\rVert_2\quad[\mathrm{mm}].
\tag{23}
$$

**(C4) Camera translation error-scaled**　在 (23) 的对齐序列上按 (4) 拟合 $s^*$ 后计算。

**(C5) W-Camera translation error**　刚体对齐 (3)：$\hat{c}^{r}_t=\hat{C}_0^{\top}(\hat{c}_t-\hat{c}_0)$，$c^{r}_t=C_0^{\top}(c_t-c_0)$，

$$
\mathrm{WCTE}=\frac{10^3}{T}\sum_{t}\bigl\lVert\hat{c}^{r}_t-c^{r}_t\bigr\rVert_2\quad[\mathrm{mm}].
\tag{24}
$$

**(C6) W-Camera translation error-scaled**　在 (24) 上拟合 $s^*$ 后计算。

**(C7) Camera trajectory rotation error**　相邻帧相对旋转增量本身不含全局偏移：

$$
\mathrm{CTrajR}
=\frac{1}{T-1}\sum_{t=0}^{T-2}d_{\mathrm{deg}}\bigl(\hat{C}_t^{\top}\hat{C}_{t+1},\;C_t^{\top}C_{t+1}\bigr)
\quad[\mathrm{deg}].
\tag{25}
$$

**(C8) Camera trajectory translation error**　相邻帧位移差：

$$
\mathrm{CTrajT}
=\frac{10^3}{T-1}\sum_{t=0}^{T-2}\Bigl\lVert\bigl(\hat{c}_{t+1}-\hat{c}_t\bigr)-\bigl(c_{t+1}-c_t\bigr)\Bigr\rVert_2
\quad[\mathrm{mm}].
\tag{26}
$$

**(C9) Camera trajectory translation error-scaled**　在 (26) 的位移序列上按 (4) 拟合 $s^*$ 后计算。

---

## 7. 评估协议（`eval/run.py`）

1. **数据划分**：官方 train/Aria clip 全集按 `partition_clips(seed=2026)` 确定性划分为本地 train 1272 / test 244；仅评估 test244。
2. **窗口**：每个 clip 固定取两个 64 帧窗口 $[0,63]$ 与 $[64,127]$；每 clip 150 帧的末尾 22 帧丢弃。任何不满足 150 连续帧的 clip 报错。
3. **采样**：Heun ODE 积分，步数 $10$（NFE $=2\times10-1=19$）。窗口随机种子
   $$
   \mathrm{seed}_{\mathrm{win}}=(\mathrm{seed}+2\,i_{\mathrm{clip}}+i_{\mathrm{win}})\ \mathrm{mod}\ 2^{63},
   \tag{27}
   $$
   与处理顺序、截断及全局 RNG 消耗无关。
4. **缺失真值策略**：窗口内任一手任一帧缺 `mano_pose` ⇒ 该窗口**全部**手部指标（M1–M16）置 `null`；窗口与相机指标（C1–C9）照常计算。不丢窗口、不做有效子集平均。
5. **非有限值**：任一指标计算结果非有限 ⇒ 该值置 `null` 并在 `records.csv` 记录指标名。
6. **聚合**：`summary.json` 对每个指标做**等权窗口均值**
   $$\bar{m}=\frac{1}{N}\sum_{w=1}^{N}m_w,$$
   且只要存在任一请求窗口的 $m_w$ 为 `null`/非有限，$\bar{m}$ 即为 `null`。CSV 空单元格表示 `null`。

---

## 8. 可视化中的对齐（现状备注，非评估指标）

- **3D mesh/轨迹动画与 mesh 投影视频**：不做任何对齐，GT 与预测各自以首帧相机 upright 参考系原始状态渲染。
- **`--scale-align` 轨迹图**（`src/visualization/geometry.py: align_scene_scale`）：每窗口拟合单一共享正尺度
  $$s=\frac{\sum_{h,t,j}\langle\hat{X}_{h,t,j},X_{h,t,j}\rangle+\sum_t\langle\hat{c}_t,c_t\rangle}{\sum_{h,t,j}\lVert\hat{X}_{h,t,j}\rVert^2+\sum_t\lVert\hat{c}_t\rVert^2},$$
  同时作用于预测关节、顶点与相机平移；**不**拟合平移或旋转。该尺度在透视投影下被除法约去，不改变 mesh 的 2D 像素位置。
- 每窗口的 `alignment_scale` 数值保存于对应 `metrics.json` / `manifest.json`。

---

## 9. 实现对照表

| 公式 | 指标键 | 实现 |
|---|---|---|
| (5)–(6) | `mpjpe_mm`, `mpjpe_scaled_mm` | `world.py` |
| (7)–(8) | `w_mpjpe_mm`, `w_mpjpe_scaled_mm` | `world.py` |
| (8)–(6) | `root_translation_error_mm`, `..._scaled_mm` | `world.py` |
| (9)–(11) | `root_rotation_error_deg`, `w_root_rotation_error_deg`, `local_rotation_error_deg`, `w_local_rotation_error_deg` | `world.py` |
| (16) | `pa_mpjpe_mm` | `aligned.py` |
| (17) | `wa_mpjpe_mm` | `aligned.py` |
| (M13) | `wa_mpjpe_scaled_mm` | `aligned.py` |
| (18)–(20) | `velocity_error_mmps`, `acceleration_error_mmps2`, `endpoint_drift_mm` | `motion.py` |
| (21)–(26) | `camera_*` / `w_camera_*` / `camera_trajectory_*` | `camera.py` |
| (27) | 窗口种子 | `run.py` |

**注意事项**：(i) M14/M15 的单位注释为 mm/s、mm/s²，但实现假设 $dt=1$ 帧；(ii) M10 与 M9 当前恒等；(iii) `fit_similarity_transform` 为向后兼容的居中形式，返回 (rotation, scale, 两个质心) 而非平移。
