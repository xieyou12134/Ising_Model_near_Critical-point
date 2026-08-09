# 临界 Ising Monte Carlo 数据流水线

本目录实现 [`Monte_Carlo.md`](Monte_Carlo.md) 中的数据方案。生产数据使用二维周期父场和 Wolff cluster 更新；训练时再裁成开放窗口。所有随机数、父构型、crop 和诊断结果都能追溯到 split、chain、parent、配置 checksum 与 Git commit。

## 1. 安装与本地验收

建议使用 Python 3.10–3.12。Monte Carlo 只使用 CPU，不需要 GPU。

```bash
cd 1-Monte_Carlo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
```

先运行不会并入正式数据的 `L=32` smoke test：

```bash
ising-mc smoke --config configs/smoke_L32.yaml --workers 2
```

输出位于 `data/smoke_L32/` 和 `manifests/smoke/`。它检查初始化、JIT、采样、原子落盘、读取、manifest、checksum 和基本物理/FFT 恒等式链路，但不把小样本的 ESS 或 R-hat 当作生产验收条件。

## 2. 正式生成顺序

每个 split 可以单独断点续跑。建议 `workers` 不超过物理 CPU 核数；每个并行 worker 通常需要约 150–250 MiB 内存。以下示例使用8个 worker：

```bash
mkdir -p logs

ising-mc generate --config configs/critical_L512_train.yaml --workers 8
ising-mc generate --config configs/critical_L512_val.yaml --workers 4
ising-mc generate --config configs/critical_L512_reference_a.yaml --workers 4
ising-mc generate --config configs/critical_L512_reference_b.yaml --workers 4
```

也可以顺序完成全部四个 split：

```bash
ising-mc generate-all --config-dir configs --workers 8
```

已完成且 checksum 正确的链会显示 `reused`，因此命令中断后可直接重跑。`--force` 会重新生成并覆盖目标链，不应在正常续跑时使用。

运行开始时，配置会复制为对应 split 下只读的 `run_config.yaml`。若生产参数发生变化，必须修改 `run_name` 和输出目录，不能把新参数混入已有运行。

## 3. 诊断、crop 与最终验收

四个 split 完成后运行物理和数值验收：

```bash
ising-mc diagnose --config-dir configs
```

该命令检查 shape、dtype、自旋取值、checksum、能量和磁化复算、FFT 恒等式、链级 split-R-hat、ESS、临界能量、零磁场对称性和局部 Gibbs 条件概率。只在 `reports/validation.md` 总体状态为 `PASS` 后构造正式 crop：

```bash
ising-mc make-crops --config-dir configs --spec configs/crops.yaml
ising-mc verify --config-dir configs
```

也可用一个命令执行生产、诊断、crop 和验收；若物理诊断失败，它会以非零状态退出且不创建 crop manifest：

```bash
ising-mc run-all --config-dir configs --workers 8
```

### 父尺寸效应确认

主数据通过后，再生成独立的小型 `L=1024` 确认集：

```bash
ising-mc generate --config configs/critical_L1024_parent_size_check.yaml --workers 4
ising-mc parent-size-check --config-dir configs --spec configs/crops.yaml --max-distance 64
```

程序分别比较 `W=128/256`、`r<=64` 的开放窗口关联函数，并以 `reference_a`–`reference_b` 的 MC–MC 差异作为自然误差基线。结果写入 `reports/parent_size_check.md` 和 `reports/parent_size_check.npz`。

## 4. 目录和数据契约

生产后目录如下；大文件均被 `.gitignore` 排除，不会推送到 GitHub：

```text
1-Monte_Carlo/
├── configs/
│   ├── critical_L512_train.yaml
│   ├── critical_L512_val.yaml
│   ├── critical_L512_reference_a.yaml
│   ├── critical_L512_reference_b.yaml
│   ├── critical_L1024_parent_size_check.yaml
│   └── crops.yaml
├── data/
│   ├── critical_L512/{train,val,reference_a,reference_b}/chain_*.npy
│   └── critical_L1024/parent_size_check/chain_*.npy
├── manifests/
│   ├── parents.csv
│   ├── val_crops.csv
│   └── reference_crops.csv
├── reports/
│   ├── chain_diagnostics.csv
│   ├── observables.npz
│   └── validation.md
├── src/critical_ising_mc/
└── tests/
```

每个 `chain_NNN.npy` 是 `[n_samples, L, L]`、`int8`、取值严格为 `{-1,+1}` 的可 memory-map 数组。旁边的 `.meta.json` 保存配置和 shard checksum、随机数 seed、pilot、burn-in、固定 gap 及运行环境；`.metrics.npz` 保存逐构型能量、磁化、每次 cluster size、累计更新格点数和实际 gap。

`parents.csv` 至少包含：

```text
sample_id, split, chain_id, index_in_chain, beta, parent_size,
seed, initial_state, gap_cluster_flips, realized_gap_sweeps,
energy, magnetization, shard_path, sha256, config_sha256
```

## 5. 模型侧读取

训练数据使用链均匀、父构型均匀的在线随机 crop，并在每次读取时应用 D4 和全局 spin flip：

```python
from critical_ising_mc.crops import OnlineTrainingCropDataset

train_data = OnlineTrainingCropDataset(
    parent_manifest="manifests/parents.csv",
    monte_carlo_root=".",
    sizes=(32, 48, 64),
    epoch_size=100_000,
    base_seed=2026080950,
)
train_data.set_epoch(0)
sample = train_data[0]
```

验证和参考数据必须读取冻结的 crop manifest：

```python
from critical_ising_mc.crops import FixedCropDataset

val_data = FixedCropDataset("manifests/val_crops.csv", monte_carlo_root=".")
reference_data = FixedCropDataset("manifests/reference_crops.csv", monte_carlo_root=".")
```

两个类都返回 `spin, beta, split, chain_id, parent_id, crop_top, crop_left, transform_id`，并可直接交给 PyTorch `DataLoader`。crop 保持 `{-1,+1}`；若模型使用0/1 token，应在模型 dataloader 中显式转换为 `(spin + 1) // 2`。

## 6. AutoDL 建议流程

```bash
git clone https://github.com/xieyou12134/Ising_Model_near_Critical-point.git
cd Ising_Model_near_Critical-point/1-Monte_Carlo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
ising-mc smoke --workers 2
ising-mc run-all --config-dir configs --workers 8
```

长任务建议在 `tmux`/`screen` 中运行。每条链独立原子落盘，单条链失败不会破坏其他链；重启相同命令即可恢复。
