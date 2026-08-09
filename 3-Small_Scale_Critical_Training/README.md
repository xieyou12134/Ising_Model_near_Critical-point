# 3. Small-Scale Critical Training

本目录实现 `2-Model_Design` 中的小尺度临界 Ising 训练基线。训练宽度为 $W\in\{32,48,64\}$，目标是在进入大尺度 context extrapolation 之前，先验证模型、扩散目标、恢复机制和训练尺度内的生成能力。

## 1. 当前实验边界

本阶段固定物理逆温度

$$
\beta=\beta_c=\frac{1}{2}\log(1+\sqrt{2})\approx0.44068679350977147.
$$

`beta` 仍保留在数据 batch、模型 `forward` 和采样接口中，但正式配置使用：

```yaml
condition_on_beta: false
fixed_beta: 0.44068679350977147
```

因此当前网络只条件化扩散时间 $t$，不会把 $\beta$ 输入 conditioner。若后续研究多温度模型，将 `condition_on_beta` 改为 `true` 即可启用已有的可选 $\beta$ 分支；数据、objective 和 sampler 接口不需要改变。

配置中的 `training.adam_betas` 是 AdamW 的一阶/二阶矩系数，与物理逆温度 $\beta$ 无关。

## 2. 实现内容

```text
3-Small_Scale_Critical_Training/
├── configs/
│   ├── train_smoke.yaml
│   └── train_critical.yaml
├── scripts/
│   ├── train.py
│   └── sample.py
├── src/ising_scale_diffusion/
│   ├── spec.py          # 配置、固定 beta 协议和校验
│   ├── rng.py           # 稳定的逻辑样本 seed
│   ├── data.py          # chain-uniform crop、D4、spin flip、多宽度 batch
│   ├── model.py         # 条件化共享 row/column axial Transformer + 2D RoPE
│   ├── objective.py     # Bernoulli absorbing corruption 与 1/t NELBO
│   ├── system.py        # 模型和 objective 的职责边界
│   ├── trainer.py       # AdamW、AMP、验证、日志和原子 checkpoint
│   ├── evaluator.py     # 固定 t 网格与固定 crop 的 likelihood ruler
│   ├── sampler.py       # schedule-consistent absorbing reverse sampler
│   ├── observables.py   # 能量、磁化、Binder 和开放边界关联
│   ├── artifacts.py     # 环境、manifest hash 与运行产物
│   └── cli.py           # `ising-train` / `ising-sample`
└── tests/
```

模型输入为部分 mask 的 `int64 [B,H,W]` token，取值为 `0/1/2`，其中 `2` 是 `[MASK]`。输出为 clean spin 的 logits：

$$
\ell_\theta(X_t,t)\in\mathbb R^{B\times H\times W\times2}.
$$

训练目标为 fixed-site NELBO：

$$
\mathcal L
=\frac{1}{B}\sum_b\frac{1}{H_bW_b}
\sum_i\frac{M_{b,i}}{t_b}
\left[-\log p_\theta((X_0)_{b,i}\mid X_t,t_b)\right].
$$

分母始终是固定的 $H_bW_b$，不是实际 mask 数。每个 optimizer step 只使用一个宽度；宽度序列、crop、增强、$t$ 和 mask 都由稳定 seed 派生，恢复训练后可重放。

## 3. 数据准备

正式训练依赖相邻目录 `1-Monte_Carlo` 的以下文件：

```text
1-Monte_Carlo/manifests/parents.csv
1-Monte_Carlo/manifests/val_crops.csv
1-Monte_Carlo/data/critical_L512/...
```

先按 `1-Monte_Carlo/README.md` 完成：

```bash
cd ../1-Monte_Carlo
ising-mc run-all --config-dir configs --workers 8
```

`train` 与 `val` 已在 Monte Carlo 阶段按独立 chain 划分。训练在线裁取并应用 $D_4$ 与全局 spin flip；validation 只读取冻结的 `val_crops.csv`。

只运行训练 smoke test 时，先生成较小数据：

```bash
cd ../1-Monte_Carlo
ising-mc smoke --config configs/smoke_L32.yaml --workers 2
```

## 4. 安装

建议使用 Python 3.10–3.12 和 PyTorch 2.2 以上版本。GPU 环境应先按 CUDA 版本安装 PyTorch，再安装本项目：

```bash
cd 3-Small_Scale_Critical_Training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

AutoDL 镜像若已安装 PyTorch，可直接执行最后一条命令。确认环境：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 5. 正确性测试与 smoke 训练

正式运行前执行：

```bash
pytest -q
ising-train --config configs/train_smoke.yaml
```

smoke 配置使用 $W=16$、2 层、$d=64$、20 steps，目标是检查完整 forward/backward、validation、checkpoint 和 resume 链路，不用于科学结论。

临时验证而不占用配置中的正式输出目录时，可增加 `--output outputs/my_smoke_check`。

恢复 smoke checkpoint：

```bash
ising-train \
  --config configs/train_smoke.yaml \
  --resume outputs/critical_small_scale_smoke/checkpoints/last.pt
```

## 6. 正式小尺度临界训练

前台运行：

```bash
ising-train --config configs/train_critical.yaml
```

AutoDL 后台运行：

```bash
mkdir -p outputs/critical_small_scale_baseline
nohup ising-train --config configs/train_critical.yaml \
  > outputs/critical_small_scale_baseline/launcher.log 2>&1 </dev/null &
echo $! > outputs/critical_small_scale_baseline/job.pid
```

随时查看：

```bash
tail -f outputs/critical_small_scale_baseline/launcher.log
tail -f outputs/critical_small_scale_baseline/logs/train.jsonl
ps -p "$(cat outputs/critical_small_scale_baseline/job.pid)" -o pid,etime,stat,cmd
```

断点恢复：

```bash
ising-train \
  --config configs/train_critical.yaml \
  --resume outputs/critical_small_scale_baseline/checkpoints/last.pt
```

checkpoint 保存模型、optimizer、scheduler、AMP scaler、global step、累计格点数、CPU/CUDA RNG、配置 hash 和 best validation NELBO。写入使用临时文件加原子替换。

## 7. 训练尺度内采样

在 $W=32$ 上从全 mask 生成：

```bash
ising-sample \
  --config configs/train_critical.yaml \
  --checkpoint outputs/critical_small_scale_baseline/checkpoints/best.pt \
  --width 32 \
  --batch-size 16 \
  --steps 64 \
  --seed 2026080930
```

固定-$\beta_c$ checkpoint 可以省略 `--beta`。若显式传入不同值，程序会拒绝运行，避免把未训练的温度误写成条件生成结果。

sampler 从 $X_1=\mathrm{MASK}^{H\times W}$ 开始。由 $t$ 移动到 $s<t$ 时，每个仍为 mask 的位置以

$$
p_{\mathrm{reveal}}=1-\frac{s}{t}
$$

揭示，已经揭示的位置不会被修改。产物包括：

```text
samples/W32/samples_seed*.npz
samples/W32/samples_seed*_trace.jsonl
samples/W32/samples_seed*_manifest.json
```

manifest 记录 checkpoint hash、固定 $\beta_c$、网格尺寸、reverse steps、sampling temperature、seed 和基础物理观测量。

## 8. 运行产物

```text
outputs/<run_id>/
├── resolved_config.yaml
├── environment.json
├── data_manifest_snapshot.json
├── checkpoints/
│   ├── last.pt
│   ├── best.pt
│   └── step_*.pt
├── logs/train.jsonl
├── validation/nelbo_by_t.csv
└── samples/
```

checkpoint 只根据训练尺度内的固定 validation NELBO 选择。$W>64$ 的外推结果不得用于选择 checkpoint、调整 RoPE 或改变 reverse steps。

## 9. 当前范围与下一阶段

本目录刻意聚焦单 GPU、固定 batch、固定 $\beta_c$ 的可审计 baseline。后续阶段可以在不改变 objective 的前提下加入：

- DDP 与 site-aware 动态 batch adapter；
- `condition_on_beta: true` 的多温度训练；
- $W=96,128,192,256$ 的冻结模型 context extrapolation；
- 独立的物理评估、置信区间和 patch-shuffle 负控制。

“模型能在更大张量上运行”只代表 shape extrapolation；只有 $G_c(r)$、磁化分布、Binder 和低波数结构在未见距离上接近独立 Monte Carlo 参考，才能支持物理尺度外推结论。
