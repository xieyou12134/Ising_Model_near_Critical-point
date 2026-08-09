# 当前模型设计中的 Infra 总结

本文只提炼同目录 `README.md` 中与 infra 有关的内容，不重复模型公式与物理背景。该 infra 目前是设计方案，还没有实测吞吐和显存数据。

## 1. 总体组成

整体运行链路为：

```text
Monte Carlo 数据缓存
-> chain-level 数据划分
-> 可重放的 crop、增强与 mask
-> IsingDiffusionSystem
-> nanoinfra Trainer
-> checkpoint
-> 多尺寸采样
-> 物理评估与结果归档
```

各组件的职责如下：

| 组件 | 主要职责 |
|:---|:---|
| 数据缓存 | 只读保存 Monte Carlo parent fields、链信息和物理参数 |
| Dataset | 按 chain 划分数据，在线生成 crop、$D_4$ 增强和 spin flip |
| Model | 只负责从带 mask 的构型输出 clean-spin logits |
| Objective/System | 生成 corruption，并计算 $1/t$ 加权 NELBO |
| Trainer | 负责优化、梯度累积、DDP、日志和 checkpoint |
| Sampler | 从全 mask 状态逐步生成完整 Ising 构型 |
| Evaluator | 计算固定 validation loss 与物理观测量 |
| Artifacts | 保存配置、样本、指标、图表和可追溯 manifest |

## 2. 数据与可复现性

- Parent fields 使用 memmap 或 chunked array，只保存一次，不预先物化大量重复 crop。
- `manifest.json` 记录边界条件、$\beta$、Monte Carlo 配置、chain seed、文件 hash 和数据划分。
- crop、增强、扩散时间和 mask 由稳定的逻辑样本身份派生，可在 resume 后重放。
- train、validation 和 test 先按 Monte Carlo chain 划分，避免重叠 crop 造成数据泄漏。

## 3. 训练组织

模型目标通过

```text
IsingDiffusionSystem.loss(batch) -> scalar loss
```

暴露给通用 Trainer。固定 batch baseline 可直接复用 nanoinfra；按尺寸动态改变 batch size 时，只增加一个 site-aware adapter，负责梯度累积和格点数量统计，不修改 Ising objective。

不同宽度使用同尺寸 microbatch，并由确定性的 width schedule 交替训练。基础方案使用固定 batch，便于审计；优化方案按近似固定 site budget 设置不同宽度的 batch size，提高小尺寸任务的 GPU 利用率。

## 4. GPU 与分布式设计

- BF16 前向和反向，optimizer state 与 loss reduction 保持 FP32。
- 使用 fused SDPA/Flash attention，不显式保存完整 attention matrix。
- 按 $W=32,64,128$ 分别缓存编译图，避免动态 shape 反复编译。
- 优先采用 DDP/NanoDDP；模型参数单卡放不下时才考虑 FSDP。
- 大模型训练可按 block 使用 activation checkpointing，推理时关闭。
- 单卡小模型优先扩大 batch；多卡更适合并行不同 seed 或实验，而不是强行切分很小的模型。
- 日志同时记录 optimizer steps、样本数、格点数、耗时和峰值显存。

## 5. Checkpoint 与恢复

Checkpoint 计划保存：

- model、optimizer、scheduler 和 gradient scaler；
- global step、累计处理格点数和 wall time；
- dataloader/sampler 状态；
- CPU/CUDA RNG 状态；
- resolved config、数据 manifest hash、git commit 和工作区状态；
- validation ruler 版本与 best metric。

写入采用临时路径加原子 rename，并保留 `last`、周期快照和预先定义的 `best`。外推尺寸的结果不能反向参与 checkpoint 选择。

## 6. 运行产物与正确性门禁

每次运行保存配置、环境信息、数据 manifest、训练日志、checkpoint、逐尺寸样本、指标、图表和最终报告。所有结果都能追溯到 checkpoint、采样 seed 与 MC reference。

正式实验前需验证：中断恢复一致性、单卡与 DDP 梯度等价、compiled/eager 等价、各宽度不 OOM、固定 seed 可重复，以及 sampler 最终不残留 mask。

## 7. 设计重点

这套 infra 的核心不是增加更多框架层，而是确保三件事：

1. 数据、噪声和多尺寸调度能够确定性重放；
2. 训练、采样和物理评估职责分离；
3. GPU 优化不改变实验的数据权重、checkpoint 选择和科学问题。
