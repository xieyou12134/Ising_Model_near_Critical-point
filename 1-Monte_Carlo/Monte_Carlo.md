# 1. Monte Carlo：小尺寸临界 Ising 训练数据方案

## 1. 目标与实验边界

本阶段只为主线实验准备数据：在二维最近邻、零外场 Ising 模型的临界逆温度上生成可信的 Monte Carlo 父组态，再从父组态裁出多个小尺寸开放窗口，供 masked diffusion Transformer 训练。

固定物理条件为：

- 正方格，周期边界；
- $J=1$、$h=0$、$k_B=1$；
- 临界逆温度 $\beta_c=\frac{1}{2}\log(1+\sqrt{2})\approx0.44068679350977147$；
- 主父格点 $L_{\mathrm{parent}}=512$；
- 训练窗口 $L\in\{32,48,64\}$；
- 后续推理参考窗口 $L\in\{128,256\}$。

本阶段不混入其他 $\beta$。数据格式中保留 `beta` 字段，之后的高温负控制和多温度实验使用新的独立 Monte Carlo 链生成。

选择“大周期父场再裁窗”而不是分别模拟小周期格点，是因为主线要研究从有限观察窗口向更大开放窗口外推。训练、验证和推理参考数据必须采用同一种裁窗协议。

## 2. 技术原理

### 2.1 目标分布

构型 $\mathbf{s}\in\{-1,+1\}^{L^2}$ 的 Hamiltonian 为 $H(\mathbf{s})=-\sum_{x,y}(s_{x,y}s_{x+1,y}+s_{x,y}s_{x,y+1})$，坐标按 $L$ 取模，每条键只计算一次。目标 Gibbs 分布为 $p_{\beta,L}(\mathbf{s})=Z_{\beta,L}^{-1}\exp[-\beta H(\mathbf{s})]$。

在 $\beta_c$ 附近，局部 Metropolis 更新存在严重的 critical slowing down，因此使用 Wolff cluster 算法生成父组态。

### 2.2 Wolff cluster 更新

一次更新执行以下步骤：

1. 均匀选择一个 seed site，记录其自旋方向 $\sigma$；
2. 从 seed 开始检查四个周期最近邻；
3. 对尚未加入、且自旋同为 $\sigma$ 的邻居，以概率 $p_{\mathrm{add}}=1-\exp(-2\beta J)$ 加入 cluster；
4. 递归扩展，直到没有新格点加入；
5. 将 cluster 内所有自旋整体翻转。

该更新满足目标 Gibbs 分布的 detailed balance，并通过集体翻转显著减弱临界点的自相关。

### 2.3 Monte Carlo 时间与独立性

一次 cluster flip 更新的格点数 $|C|$ 不固定。定义累计 sweep-equivalent time 为 $\tau_{\mathrm{sweep}}=\sum_k |C_k|/L^2$，仅用于描述计算量和设置初始间隔。

不能在“累计翻转格点数刚好超过某个阈值”时保存样本，因为保存时刻依赖当前 cluster 大小，可能产生 stopping-time bias。正确做法是：先用不保留的 pilot 估计平均 cluster 大小 $\overline{|C|}$，再固定 burn-in 和相邻样本之间的 cluster flip 次数。若目标间隔为 $g$ 个等效 sweeps，则固定 $n_{\mathrm{gap}}=\lceil gL^2/\overline{|C|}\rceil$。

相邻保存组态不需要严格独立，但必须估计能量、$|m|$、$m^2$ 和低波数结构因子的 integrated autocorrelation time。对长度为 $n$ 的序列，按 $\mathrm{ESS}\approx n/(2\tau_{\mathrm{int}})$ 报告有效样本量。误差估计和数据划分的独立单位是 Monte Carlo 链及父组态，而不是 crop 数量。

### 2.4 裁窗协议

每个父组态是 $512\times512$ 周期场，训练样本是其中不跨边界的连续子数组。crop 自身按开放窗口处理，不把 crop 的左右边或上下边重新连接为周期近邻。

同一父组态的多个 crop 强相关，因此必须先按链划分 `train/val/reference_a/reference_b`，再在各 split 内裁窗。任何父组态及其 crop 不得跨 split。

`reference_a` 和 `reference_b` 是两份独立 MC 参考，用来估计“MC 与 MC 之间”的自然误差；二者都不能参与训练或超参数选择。

## 3. 首轮生产配置

| split | 独立链数 | 每链父组态数 | 用途 |
|---|---:|---:|---|
| `train` | 8 | 512 | 在线随机裁取 $32/48/64$ 窗口 |
| `val` | 4 | 512 | 固定小窗口验证与模型选择 |
| `reference_a` | 4 | 512 | 固定 $32/48/64/128/256$ MC 参考 |
| `reference_b` | 4 | 512 | 独立 MC 基线和最终不确定性校准 |

每个 split 使用不同的 `base_seed`。每条链的 seed 由 `SeedSequence([base_seed, split_id, chain_id])` 派生，不手工填写连续 seed。各 split 分开运行和落盘。

推荐的 sampler 起始参数如下，pilot 诊断不通过时再延长，不通过诊断的运行不能进入数据集：

| 参数 | 起始值 |
|---|---:|
| $L_{\mathrm{parent}}$ | 512 |
| $\beta$ | 0.44068679350977147 |
| adaptation | 5 equivalent sweeps |
| cluster-size pilot | 5 轮，每轮 256 flips |
| burn-in | 50 equivalent sweeps |
| 保存间隔 | 2 equivalent sweeps 对应的固定 flips 数 |
| 保存样本 | 每链 512 |
| 自旋 dtype | `int8` |

各 split 中一半链从随机场开始，另一半从全 $+1$ 场开始，用于检查不同初态是否收敛到相同分布。生产阶段保持每条链的更新和随机数流独立；并行只发生在链之间。

## 4. 实现流程

### 4.1 冻结运行配置

每次运行先写入只读配置，至少包含：

```yaml
run_name: critical_L512_train
split: train
beta: 0.44068679350977147
parent_size: 512
n_chains: 8
n_samples_per_chain: 512
adaptation_sweeps: 5.0
pilot_cluster_steps: 256
pilot_rounds: 5
burnin_sweeps: 50.0
sweeps_between_samples: 2.0
base_seed: 2026080901
dtype: int8
```

运行开始后不原地修改配置；若参数变化，使用新的 `run_name` 和输出目录。

### 4.2 实现并单测 sampler

用 Python + Numba 或等价的编译实现保存 `int8[L,L]` 自旋。每次 cluster 更新使用预分配的 stack、cluster index 和 visitation marker，避免在循环内创建 Python 对象。

在 $L=4$ 上先完成以下测试：

- 所有自旋始终属于 $\{-1,+1\}$；
- 周期邻居索引正确，每个格点有四个邻居；
- cluster 内格点只加入一次，返回的 cluster size 等于实际翻转数；
- 固定 seed 时输出可复现，不同链 seed 不重复；
- Monte Carlo 的能量和磁化分布与 $L=4$ 精确枚举在统计误差内一致。

上述测试通过后才运行大格点。

### 4.3 Smoke run

先运行 $L=32$、2 条链、每链 32 个保存样本，只检查完整链路：初始化、Wolff 更新、保存、读取、观测量计算、manifest 和 checksum。Smoke 数据不得并入正式数据集。

### 4.4 Pilot 与 burn-in 检查

对 $L=512$ 运行独立 pilot：

1. 完成 5 个等效 sweeps 的 adaptation；
2. 用固定次数的 cluster flips 估计 $\overline{|C|}$；
3. 将 50 个等效 sweeps 换算成固定的 burn-in flip 数；
4. burn-in 后再次估计 $\overline{|C|}$；
5. 固定正式生产的 $n_{\mathrm{gap}}$，此后不根据链状态动态修改；
6. 保存短序列，计算 $e$、$|m|$、$m^2$、$m^4$、$S(k_{\min})$ 的 $\tau_{\mathrm{int}}$、ESS 和 split-$\hat R$。

若随机初态和有序初态的链均值仍有系统差异，或 split-$\hat R>1.05$，优先延长 burn-in；若 burn-in 已通过但 ESS 太低，增加保存样本数，必要时再增大固定 gap。

### 4.5 正式生成父组态

按 `train`、`val`、`reference_a`、`reference_b` 顺序分别运行。每条链执行：

1. 从指定初态初始化；
2. 完成不保存的 adaptation、pilot 和 burn-in；
3. 使用固定 $n_{\mathrm{gap}}$ 推进链；
4. 每隔固定 flips 保存一个完整父组态；
5. 同时记录能量、磁化、cluster size、累计更新格点数和 RNG seed；
6. 每条链单独写一个 `.npy` shard，形状为 `[512,512,512]`、dtype 为 `int8`；
7. 原子地完成文件写入后计算 SHA-256，并更新 manifest。

不使用单个压缩 `.npz` 保存整个 split；逐链 `.npy` shard 可以 memory-map，失败时也只需重跑对应链。

### 4.6 物理与数值验收

对每条链和每个 split 分别计算以下量：

- 每自旋能量 $e=H/L^2$；
- 磁化 $m=L^{-2}\sum_i s_i$、$|m|$、$m^2$、$m^4$；
- Binder 累积量 $U_4=1-\langle m^4\rangle/(3\langle m^2\rangle^2)$；
- 周期两点关联 $G(\mathbf r)=L^{-2}\langle\sum_i s_i s_{i+\mathbf r}\rangle$；
- 结构因子 $S(\mathbf k)=L^{-2}\langle|\sum_j s_j e^{i\mathbf k\cdot\mathbf r_j}|^2\rangle$；
- $S(0)$、$S(k_{\min})$ 和 $\xi_2/L$；
- 局部 Gibbs 校准 $P(s_i=+1\mid q)=\operatorname{sigmoid}(2\beta q)$，其中 $q\in\{-4,-2,0,2,4\}$ 是四邻居自旋和。

必须通过的实现检查：

- $G(0)=1$，数值误差小于 $10^{-10}$；
- FFT/Parseval、能量—最近邻关联和 $S(0)=L^2\langle m^2\rangle$ 恒等式误差小于 $10^{-10}$；
- 所有样本 shape、dtype、取值、seed、sample ID 和 checksum 正确；
- 不存在父组态跨 split 或 seed 重复。

生产质量检查：

- energy 和 $|m|$ 的 split-$\hat R\le1.05$；
- 每条链在 $e$、$|m|$、$m^2$、$m^4$、$S(k_{\min})$ 上的最小 ESS 至少为 30，目标为 100；
- $\langle m\rangle$ 与 0 的偏差不超过链级标准误的 3 倍；
- $\langle e\rangle$ 应接近临界无限体精确值 $-\sqrt{2}$，要求偏差不超过 $\max(5\,\mathrm{MCSE},0.02)$；
- 计数不少于 500 的局部 Gibbs bin 中，经验条件概率与精确值的最大绝对误差不超过 0.05；
- $G(r)$ 在中间距离呈与 $r^{-1/4}$ 相容的趋势；该项是诊断，不单独作为硬阈值。

统计误差按整条独立链重采样；若分析 crop，则在链下继续以 parent ID 为 cluster。不得把所有 crop 当作独立样本计算标准误。

### 4.7 构造训练和参考窗口

父组态是唯一的 canonical data，crop 通过 manifest 和确定性 RNG 生成。

训练集：

1. 每个 batch 先从 $\{32,48,64\}$ 均匀选择一个尺寸；
2. 先均匀选择链，再均匀选择父组态，避免不同链被不等权采样；
3. 在父组态内部均匀选择不跨边界的左上角；
4. 每次只裁一个连续开放窗口；
5. 在线随机应用 $D_4$ 旋转/反射和全局 spin flip；
6. 返回 `spin`, `beta`, `split`, `chain_id`, `parent_id`, `crop_top`, `crop_left`, `transform_id`。

验证集与参考集：

- 使用冻结的 crop manifest，不在不同 checkpoint 之间改变 crop；
- `val` 固定生成 $32/48/64$ crop；
- `reference_a` 和 `reference_b` 固定生成 $32/48/64/128/256$ crop；
- 计算 crop 关联时只统计窗口内有效点对，不做周期 wrap；
- $256$ crop 的主要关联分析限制在 $r\le64$，与后续生成模型的评估区间一致。

同一父组态可以贡献多个训练 crop，但不能因此增加统计上的独立样本数。验证和参考分析默认每个 parent、每个尺寸只取一个固定 crop；若需要多个 crop，bootstrap 时必须保留 parent 层级。

### 4.8 父尺寸效应确认

主数据验收后，再用 $L_{\mathrm{parent}}=1024$、至少 4 条独立链生成一个小型确认集。比较 $512$ 与 $1024$ 父场裁出的 $128/256$ 窗口在 $r\le64$ 上的 $G(r)$。

若两者差异不大于 `reference_a` 与 `reference_b` 的 MC—MC 自然差异，则 $L_{\mathrm{parent}}=512$ 可作为主实验近似无限体参考；否则应扩大父场或缩短最终评价距离。

## 5. 数据目录与契约

推荐输出结构：

```text
1-Monte_Carlo/
├── configs/
│   ├── critical_L512_train.yaml
│   ├── critical_L512_val.yaml
│   ├── critical_L512_reference_a.yaml
│   └── critical_L512_reference_b.yaml
├── data/
│   └── critical_L512/
│       ├── train/chain_000.npy
│       ├── val/chain_000.npy
│       ├── reference_a/chain_000.npy
│       └── reference_b/chain_000.npy
├── manifests/
│   ├── parents.csv
│   ├── val_crops.csv
│   └── reference_crops.csv
├── reports/
│   ├── chain_diagnostics.csv
│   ├── observables.npz
│   └── validation.md
└── Monte_Carlo.md
```

`parents.csv` 至少包含：`sample_id, split, chain_id, index_in_chain, beta, parent_size, seed, initial_state, gap_cluster_flips, realized_gap_sweeps, energy, magnetization, shard_path, sha256`。

所有实验产物同时记录 git commit、Python/NumPy/Numba 版本、配置文件 checksum、运行时间和主机信息。模型训练只读取验收状态为 `pass` 的 manifest。

## 6. 本阶段退出条件

只有同时满足以下条件，Monte Carlo 阶段才完成：

1. sampler 通过小系统精确枚举测试和全部数值恒等式检查；
2. 四个 split 均由互不重叠的独立链生成；
3. burn-in、split-$\hat R$、ESS、精确能量和局部 Gibbs 校准通过；
4. 固定的训练、验证和双参考 manifest 已生成并可复现；
5. $512$ 父场在目标距离 $r\le64$ 上的有限父尺寸误差已被量化；
6. 数据可由模型 dataloader 读取，并能追溯到 chain、parent、crop 和 seed。

完成后，主线模型只使用 `train` crop 优化参数，使用 `val` 选择 checkpoint；`reference_a` 和 `reference_b` 只用于最终物理比较。

## 7. 对应实现

本方案的生产代码、冻结配置、测试和 AutoDL 使用流程见 [`README.md`](README.md)。命令行入口为 `ising-mc`，实现位于 `src/critical_ising_mc/`；正式运行前必须先通过 `pytest` 和 `ising-mc smoke`。
