# Ising 离散扩散 Transformer：数学模型与数据流

本文只描述模型本身：它试图学习什么概率分布、接收什么输入、信息如何在网络中流动、如何定义训练目标，以及如何从全 mask 状态生成 Ising 构型。本文不讨论代码组织、具体模型尺寸、计算复杂度、训练硬件、并行方式、缓存或 checkpoint。

## 1. 模型要学习的对象

在边长为 $L$ 的二维方格上，一个 Ising 构型写作

$$
X=\{s_{x,y}\}_{x,y=0}^{L-1},
\qquad
s_{x,y}\in\{-1,+1\}.
$$

给定物理逆温度 $\beta$，目标分布是

$$
p_{\beta,L}(X)
=\frac{e^{-\beta H_L(X)}}{Z_L(\beta)}.
$$

Monte Carlo 提供来自这个分布的样本。生成模型的任务是学习一个近似分布

$$
p_{\theta,L}(X\mid\beta)
\approx p_{\beta,L}(X),
$$

其中 $\theta$ 表示 Transformer 的可学习参数。

本研究还要求同一组参数能够作用于不同的空间尺寸。若模型只在一组较小尺寸上训练，我们希望检验

$$
p_{\theta,L_{\mathrm{test}}}(X\mid\beta)
\approx p_{\beta,L_{\mathrm{test}}}(X),
\qquad
L_{\mathrm{test}}>L_{\mathrm{train}},
$$

是否仍然成立。这里的核心不是生成更大的数组，而是更大构型中的统计关系是否仍然正确。

## 2. 两种“温度”必须区分

模型中同时存在两个不同的连续变量：

- $\beta$ 是 Ising 模型的物理逆温度，决定目标数据分布；
- $t$ 是离散扩散的噪声时间，决定一个构型有多少位置被 mask。

它们回答的是完全不同的问题：

$$
\beta:
\quad
\text{我们希望生成哪一个物理分布？}
$$

$$
t:
\quad
\text{当前输入距离完整构型还有多远？}
$$

因此，模型在一般形式下学习的是条件分布

$$
p_\theta(X_0\mid X_t,t,\beta).
$$

若整个实验只研究临界点 $\beta=\beta_c$，$\beta$ 是常数，可以不作为显式输入；$t$ 仍然必须输入模型，因为模型需要区分轻度 mask 与全 mask 等不同去噪阶段。

## 3. 总体数据流

模型的数据流可以概括为

$$
X_0
\xrightarrow{\ q_t\ }
X_t
\xrightarrow{\ \text{Transformer}(t,\beta)\ }
p_\theta(X_0\mid X_t,t,\beta).
$$

其中：

1. $X_0$ 是完整的 Monte Carlo 构型；
2. $q_t$ 是只会把自旋替换成 mask 的前向过程；
3. $X_t$ 是部分信息可见、部分位置被 mask 的构型；
4. Transformer 根据可见自旋、二维位置、$t$ 和可选的 $\beta$，预测每个被 mask 位置原本的自旋；
5. 训练时用真实 $X_0$ 监督该预测；
6. 生成时从全 mask 开始，多次使用同一个条件分布逐步得到完整构型。

## 4. 离散状态空间

对神经网络而言，每个位置有三个可能的输入状态：

$$
\mathcal V_{\mathrm{input}}
=\{-1,+1,\mathrm{MASK}\}.
$$

输出空间只有两个 clean spin：

$$
\mathcal V_{\mathrm{output}}
=\{-1,+1\}.
$$

`MASK` 是前向扩散引入的吸收态，不是物理系统中的第三种自旋，也不应该成为最终生成结果。

对位置 $i=(x,y)$，模型接收的信息可以抽象写成

$$
\mathcal I_i
=\left((X_t)_i,\,x,\,y,\,t,\,\beta\right).
$$

整个网络输入是所有位置的信息集合

$$
\mathcal I
=\{\mathcal I_i\}_{i\in\Lambda_L},
$$

其中 $\Lambda_L$ 表示二维格点集合。

模型不把目标尺寸 $L$ 作为额外条件。空间尺寸只通过输入中实际存在的坐标集合体现，从而避免模型依赖一个显式的“你现在应该生成多大”的标签。

## 5. 前向 absorbing diffusion

令 $X_0$ 表示干净构型。对每个位置 $i$，前向过程独立地以概率 $t$ 将自旋替换为 `MASK`：

$$
q_t((X_t)_i\mid(X_0)_i)
=(1-t)\,\delta_{(X_t)_i,(X_0)_i}
+t\,\delta_{(X_t)_i,\mathrm{MASK}}.
$$

整个构型的条件分布为

$$
q_t(X_t\mid X_0)
=\prod_i q_t((X_t)_i\mid(X_0)_i).
$$

该过程有三个重要极限：

$$
t=0:
\quad
X_t=X_0,
$$

$$
0<t<1:
\quad
X_t\text{ 同时包含可见自旋和 mask},
$$

$$
t=1:
\quad
X_t=\mathrm{MASK}^{L\times L}.
$$

前向过程不会把 $+1$ 直接变成 $-1$，也不会把 $-1$ 直接变成 $+1$。它只删除信息，因此是 absorbing-state diffusion。

## 6. 条件去噪分布

模型对每个位置输出一个二分类分布：

$$
p_\theta((X_0)_i=s\mid X_t,t,\beta),
\qquad
s\in\{-1,+1\}.
$$

一次前向传播中，输出可以写为

$$
p_\theta(X_0\mid X_t,t,\beta)
=\prod_{i\in\mathcal M_t}
p_\theta((X_0)_i\mid X_t,t,\beta),
$$

其中 $\mathcal M_t$ 是当前被 mask 的位置集合。

这个乘积并不意味着最终生成的所有格点彼此独立。每个位置的条件概率都由同一个完整 $X_t$ 计算，因此已经依赖其他可见格点；而多步反向过程中，较早揭示的自旋还会成为后续位置的新条件。最终的联合依赖由“全局条件化 + 多步生成”共同形成。

## 7. 输入表示

### 7.1 自旋状态表示

每种输入状态对应一个向量：

$$
e_i
=E_{\mathrm{token}}((X_t)_i).
$$

`MASK` embedding 表示“该位置的自旋未知”，而不是表示自旋为零。它必须与 $+1$ 和 $-1$ 的表示分开。

### 7.2 扩散时间表示

连续变量 $t$ 经过连续特征映射得到

$$
c_t=\Phi_t(t).
$$

$c_t$ 告诉模型当前处于怎样的去噪阶段。相同的局部 mask 图案在不同 $t$ 下具有不同的概率含义，因此不能只让模型从实际 mask 数量猜测时间。

### 7.3 物理逆温度表示

当模型覆盖多个物理温度时，$\beta$ 被映射为

$$
c_\beta=\Phi_\beta(\beta).
$$

它告诉模型当前应生成高温短程关联、临界幂律关联，还是低温长程有序分布。

扩散条件与物理条件合并为

$$
c=\Psi(c_t,c_\beta).
$$

固定 $\beta_c$ 的模型则使用

$$
c=\Psi(c_t).
$$

### 7.4 初始隐状态

每个格点的初始隐状态由 token 表示和全局条件构成：

$$
h_i^{(0)}
=e_i+C(c).
$$

二维坐标不以普通向量直接加到 $h_i^{(0)}$ 中，而是在 attention 的 query 与 key 关系中发挥作用。这避免模型依赖一个固定长度的绝对位置表。

## 8. 二维空间表示

### 8.1 为什么不能使用 raster 一维位置

如果按行把二维方格展平为一维序列，相邻序号不总是对应物理近邻。例如一行末尾和下一行开头在序列中相邻，但在开放二维方格中并不是最近邻。

因此模型始终保留坐标

$$
i=(x_i,y_i),
$$

并分别处理行方向与列方向的相对位置。

### 8.2 Axial 2D RoPE

对 row attention，旋转相位只由列坐标 $y$ 决定；对 column attention，旋转相位只由行坐标 $x$ 决定。于是 attention score 能感知相对位移

$$
\Delta y=y_i-y_j
$$

或

$$
\Delta x=x_i-x_j,
$$

而不需要为每个绝对坐标学习独立 embedding。

这种表示具有两个重要性质：

1. 坐标关系直接对应二维方格的几何方向；
2. 同一个位置规则可以应用到训练时未出现的更大坐标集合。

第二点只保证模型在数学形式上可以接收更大网格，并不保证它一定学会正确外推；是否保持正确长程关联仍然必须通过生成分布检验。

## 9. 二维轴向 Transformer

### 9.1 行方向的信息传播

对格点 $i=(x_i,y_i)$，row attention 只汇总同一行的格点：

$$
a_{i,\mathrm{row}}^{(l)}
=\sum_{j:x_j=x_i}
\alpha_{ij,\mathrm{row}}^{(l)}v_j^{(l)}.
$$

权重由当前隐状态和列方向相对位置共同决定：

$$
\alpha_{ij,\mathrm{row}}^{(l)}
=\operatorname{softmax}_{j:x_j=x_i}
\left(
\frac{
\langle R(y_i)q_i^{(l)},R(y_j)k_j^{(l)}\rangle
}{\sqrt d}
\right).
$$

### 9.2 列方向的信息传播

column attention 以相同方式汇总同一列的格点：

$$
a_{i,\mathrm{col}}^{(l)}
=\sum_{j:y_j=y_i}
\alpha_{ij,\mathrm{col}}^{(l)}v_j^{(l)},
$$

$$
\alpha_{ij,\mathrm{col}}^{(l)}
=\operatorname{softmax}_{j:y_j=y_i}
\left(
\frac{
\langle R(x_i)q_i^{(l)},R(x_j)k_j^{(l)}\rangle
}{\sqrt d}
\right).
$$

行列两个方向使用相同的 attention 参数，并将输出对称地合并：

$$
a_i^{(l)}
=\frac{a_{i,\mathrm{row}}^{(l)}+a_{i,\mathrm{col}}^{(l)}}{\sqrt2}.
$$

共享参数的目的不是宣称网络自动拥有完整的旋转对称性，而是避免从架构上给横向与纵向定义两套无关的物理规律。方格旋转、反射和全局 spin flip 对称性还需要通过数据分布与最终评估共同约束。

### 9.3 跨行列的全局信息

一次 row attention 可以在一整行中传播信息，一次 column attention 可以在一整列中传播信息。两个不同行且不同列的格点可以通过它们的行列交点形成信息路径：

$$
(x_1,y_1)
\longrightarrow
(x_1,y_2)
\longrightarrow
(x_2,y_2).
$$

因此，多层轴向 Transformer 可以在保持二维结构的同时建立跨越整个网格的依赖，而不是把每个 crop 当作彼此独立的小块。

## 10. 条件化 Transformer block

令 $h^{(l)}$ 表示第 $l$ 层输入。全局条件 $c$ 在每一层调制归一化和 residual 更新：

$$
u^{(l)}
=\operatorname{Norm}(h^{(l)};c),
$$

$$
\widetilde h^{(l)}
=h^{(l)}
+g_{\mathrm{attn}}^{(l)}(c)
\odot
\operatorname{AxialAttn}^{(l)}(u^{(l)}),
$$

$$
h^{(l+1)}
=\widetilde h^{(l)}
+g_{\mathrm{mlp}}^{(l)}(c)
\odot
F^{(l)}
\left(
\operatorname{Norm}(\widetilde h^{(l)};c)
\right).
$$

这里：

- $\operatorname{AxialAttn}$ 负责不同空间位置之间的信息交换；
- $F^{(l)}$ 是逐格点的非线性特征变换；
- $g_{\mathrm{attn}}^{(l)}(c)$ 和 $g_{\mathrm{mlp}}^{(l)}(c)$ 控制不同 $t$ 与 $\beta$ 下每类更新的强度；
- residual connection 保留前一层已经形成的局部与长程信息。

条件 $c$ 在每层都出现，而不是只在输入端加入一次。这使模型可以随去噪阶段改变内部信息处理方式：全 mask 时需要从全局先验建立结构，mask 较少时则更接近局部条件补全。

## 11. Binary clean-spin head

最后一层为每个位置产生两个 logits：

$$
(\ell_{i,-},\ell_{i,+})
=W_{\mathrm{out}}\operatorname{Norm}(h_i^{(K)}).
$$

对应概率为

$$
p_{i,-}
=\frac{e^{\ell_{i,-}}}
{e^{\ell_{i,-}}+e^{\ell_{i,+}}},
$$

$$
p_{i,+}
=\frac{e^{\ell_{i,+}}}
{e^{\ell_{i,-}}+e^{\ell_{i,+}}}.
$$

模型在该位置预测的条件平均自旋为

$$
\widehat s_i
=p_{i,+}-p_{i,-}.
$$

训练和生成使用完整的二分类概率，而不是只使用 $\widehat s_i$ 或 argmax。保留概率信息对于表示临界点附近的强涨落非常重要。

## 12. 训练目标

令 $M_i=1$ 表示位置 $i$ 被 mask，定义单点负对数概率

$$
\ell_i
=-\log
p_\theta((X_0)_i\mid X_t,t,\beta).
$$

训练目标为

$$
\mathcal L(\theta)
=\mathbb E_{X_0,t,X_t}
\left[
\frac{1}{N}
\sum_{i=1}^{N}
\frac{M_i}{t}\ell_i
\right],
\qquad
N=HW.
$$

因为

$$
\mathbb E[M_i\mid t]=t,
$$

所以 $M_i/t$ 修正了一个位置在不同 $t$ 下被选中监督的概率。分母使用固定格点数 $N$，使目标始终表示每格点的平均贡献，而不会随着一次随机抽到多少 mask 而改变归一化方式。

只在 masked 位置计算监督。可见位置已经直接包含真实自旋，如果也把它们作为主要预测目标，模型可以通过复制输入得到大量无意义的低 loss。

## 13. 物理对称性如何进入数据流

零外场 Ising 分布满足全局 spin-flip 对称性：

$$
p_{\beta,L}(X)=p_{\beta,L}(-X).
$$

正方格还具有旋转和反射对称性。训练数据流对构型施加这些保持物理分布不变的变换：

$$
X_0
\longrightarrow
gX_0,
\qquad
g\in D_4\times\mathbb Z_2.
$$

然后再对变换后的构型执行 forward masking：

$$
gX_0
\xrightarrow{\ q_t\ }
X_t.
$$

这种顺序保证被 mask 的坐标、可见自旋和监督目标始终属于同一个变换后的构型。对称增强鼓励模型把方向或正负自旋视为等价物理描述，但是否真正满足对称性仍需要检查生成分布。

## 14. 反向生成过程

### 14.1 初态

生成不需要一个已有的 Ising 构型。初态为

$$
X_{t_K}=\mathrm{MASK}^{H\times W},
\qquad
t_K=1.
$$

给定目标物理逆温度 $\beta$，模型首先从完全未知的输入中给出每个位置的 clean-spin posterior。

### 14.2 单步反演

令相邻两个反向时间满足

$$
t>s.
$$

对当前仍为 mask 的位置，先从

$$
p_\theta((X_0)_i\mid X_t,t,\beta)
$$

抽取候选自旋，再以概率

$$
p_{\mathrm{reveal}}
=1-\frac{s}{t}
$$

将该位置从 mask 变为候选自旋。未被揭示的位置继续保持 mask，已经揭示的位置保持不变。

于是一次反向转换可以表示为

$$
X_t
\xrightarrow{
p_\theta(X_0\mid X_t,t,\beta)
}
X_s.
$$

### 14.3 完整生成轨迹

选择递减时间序列

$$
1=t_K>t_{K-1}>\cdots>t_1>t_0=0,
$$

得到

$$
X_{t_K}
\longrightarrow
X_{t_{K-1}}
\longrightarrow
\cdots
\longrightarrow
X_{t_1}
\longrightarrow
X_{t_0}=\widehat X_0.
$$

最终

$$
\widehat X_0\in\{-1,+1\}^{H\times W}
$$

不再包含 `MASK`。

在轨迹早期，可见信息很少，模型主要依赖 $\beta$ 对应的整体分布先验；随着自旋被逐渐揭示，后续预测会条件化于越来越丰富的局部团簇和长程结构。生成因此是从全局不确定性到具体构型的逐步收缩过程。

## 15. 联合分布如何在多步过程中形成

单次网络输出对不同 masked 位置给出形式上分解的 posterior，但完整生成分布包含所有中间状态：

$$
p_\theta(\widehat X_0\mid\beta)
=\sum_{X_{t_1},\ldots,X_{t_{K-1}}}
\prod_{k=1}^{K}
p_\theta(X_{t_{k-1}}\mid X_{t_k},t_k,\beta).
$$

因此，两个远距离自旋之间的依赖可以通过三种路径建立：

1. 它们在同一次 attention 中共享可见上下文；
2. 一个位置先被揭示，成为另一个位置后续预测的条件；
3. 多个中间格点在不同反向步骤中传递空间信息。

模型的长程生成能力不等于某一个 attention head 的单次作用，而是整个条件网络与反向轨迹共同定义的性质。

## 16. 固定临界模型与多温度模型

### 16.1 固定临界模型

固定

$$
\beta=\beta_c
$$

时，模型学习

$$
p_\theta(X_0\mid X_t,t).
$$

它只需表示临界分布，研究问题最集中：在训练窗口之外，模型能否继续产生正确的幂律关联？

### 16.2 多温度条件模型

若训练数据包含多个 $\beta$，模型学习

$$
p_\theta(X_0\mid X_t,t,\beta).
$$

此时同一网络需要表示：

$$
\beta<\beta_c:
\quad
C(r)\text{ 具有有限关联长度并近似指数衰减},
$$

$$
\beta=\beta_c:
\quad
C(r)\propto r^{-1/4},
$$

$$
\beta>\beta_c:
\quad
G(r)\text{ 在长距离形成有序平台}.
$$

多温度模型因此同时学习“局部构型怎样依赖温度”和“空间关联形式怎样随温度改变”。它还允许进一步检验温度与尺度的组合泛化，但不会改变前向 mask 和反向生成的基本结构。

## 17. 尺度外推在模型中的含义

设训练最大尺寸为 $L_{\max}$。模型在更大尺寸 $L'>L_{\max}$ 上使用相同的：

- token 状态空间；
- 条件映射；
- attention 参数；
- 二维相对位置规则；
- clean-spin 输出分布；
- forward 与 reverse diffusion 定义。

改变的只有坐标集合

$$
\Lambda_{L_{\max}}
\longrightarrow
\Lambda_{L'}.
$$

如果模型只学到训练窗口中的有限尺度纹理，那么在未见距离上生成的关联函数会偏离目标形式。如果模型学到了可延伸的临界统计结构，则更大构型中仍应出现

$$
G_c(r)\propto r^{-1/4}
$$

的未见距离区间。

因此，尺度外推不是模型定义中的额外模块，而是同一条件概率模型在更大坐标集合上的行为。

## 18. 从输入到输出的完整训练数据流

训练时，一条数据依次经过：

$$
\text{Monte Carlo parent field}
\longrightarrow
\text{合法空间 crop}
\longrightarrow
\text{物理对称变换}
\longrightarrow
X_0,
$$

$$
(X_0,t)
\longrightarrow
q_t(X_t\mid X_0)
\longrightarrow
X_t,
$$

$$
(X_t,t,\beta)
\longrightarrow
\text{token and condition representation}
\longrightarrow
\text{axial Transformer},
$$

$$
\text{hidden field}
\longrightarrow
p_\theta((X_0)_i\mid X_t,t,\beta)
\longrightarrow
\mathcal L(\theta).
$$

整个训练数据流中，完整构型 $X_0$ 只出现在前向 corruption 的起点和 loss 的目标端，不进入去噪网络的可见输入。

## 19. 从输入到输出的完整生成数据流

这一节把生成过程中的每个符号、每次箭头变换，以及负责该变换的模型结构逐一展开。完整主线是

$$
(H,W,\beta)
\longrightarrow
X_1
\longrightarrow
(X_t,t,\beta)
\longrightarrow
\{h_i^{(0)}\}
\longrightarrow
\{h_i^{(K_{\mathrm{layer}})}\}
\longrightarrow
p_\theta(X_0\mid X_t,t,\beta)
\longrightarrow
X_s
\longrightarrow
\widehat X_0
\longrightarrow
\text{物理观测量}.
$$

这里 $K_{\mathrm{layer}}$ 表示 Transformer 层数；后面使用的 $K_{\mathrm{rev}}$ 表示反向扩散步数。两者是不同概念。

### 19.1 从生成条件到全 mask 初态

第一步为

$$
(H,W,\beta)
\longrightarrow
X_1=\mathrm{MASK}^{H\times W}.
$$

各个符号的含义是：

| 符号 | 含义 |
|---|---|
| $H$ | 目标方格的高度 |
| $W$ | 目标方格的宽度 |
| $\beta$ | 希望生成的 Ising 分布所对应的物理逆温度 |
| $X_1$ | 扩散时间 $t=1$ 时的场，而不是“第一个训练样本” |
| $\mathrm{MASK}^{H\times W}$ | 一个 $H\times W$ 的场，其中每个位置都处于未知状态 |

这一步不经过 Transformer。它只是根据目标几何形状建立生成初态：

$$
(X_1)_i=\mathrm{MASK},
\qquad
\forall i\in\Lambda_{H,W}.
$$

$\Lambda_{H,W}$ 表示全部二维坐标的集合。此时没有任何自旋被指定，模型只能依据 $\beta$ 对应的分布先验和二维位置关系提出第一批自旋概率。

### 19.2 从离散输入变成 Transformer 隐状态

任意一个反向步骤的模型输入是

$$
(X_t,t,\beta).
$$

其中：

| 符号 | 含义 |
|---|---|
| $t$ | 当前扩散时间，也等价于 forward process 中单点被 mask 的概率参数 |
| $X_t$ | 当前部分揭示的构型；每个位置是 $-1$、$+1$ 或 `MASK` |
| $(X_t)_i$ | 位置 $i=(x_i,y_i)$ 当前的离散状态 |
| $\beta$ | 物理条件；它不随反向扩散步骤改变 |

这三个输入沿不同分支进入模型。

#### 自旋状态分支

每个位置的离散状态通过 token embedding 变成向量：

$$
e_i
=E_{\mathrm{token}}((X_t)_i).
$$

负责该箭头的结构是 **token embedding**：

$$
(X_t)_i
\xrightarrow{\ E_{\mathrm{token}}\ }
e_i.
$$

这里的 $e_i$ 不是物理能量，而是位置 $i$ 的神经网络表示。物理能量在生成完成后才由 Hamiltonian 计算。

#### 扩散时间分支

$t$ 通过连续条件映射变成扩散时间表示：

$$
t
\xrightarrow{\ \Phi_t\ }
c_t.
$$

$c_t$ 告诉网络当前是“几乎全 mask 的全局构造阶段”，还是“只剩少量未知位置的条件补全阶段”。

#### 物理逆温度分支

$\beta$ 通过另一套连续条件映射变成物理条件表示：

$$
\beta
\xrightarrow{\ \Phi_\beta\ }
c_\beta.
$$

$c_\beta$ 控制模型应当偏向哪一种物理统计结构，例如高温短程关联、临界幂律关联或低温长程有序。

#### 条件融合

两个连续条件合并为

$$
(c_t,c_\beta)
\xrightarrow{\ \Psi\ }
c.
$$

随后，token 表示与全局条件形成初始隐状态：

$$
h_i^{(0)}
=e_i+C(c).
$$

因此，该阶段的完整数据流是

$$
(X_t)_i
\longrightarrow
e_i,
$$

$$
(t,\beta)
\longrightarrow
c,
$$

$$
(e_i,c)
\longrightarrow
h_i^{(0)}.
$$

二维坐标 $(x_i,y_i)$ 没有在这里直接加到 $h_i^{(0)}$ 上，而是在 attention 的 query 和 key 中通过二维 RoPE 进入模型。

### 19.3 隐状态怎样经过 attention 变成下一层

设第 $l$ 层输入是

$$
\{h_i^{(l)}\}_{i\in\Lambda_{H,W}}.
$$

它经过以下结构变成第 $l+1$ 层隐状态。

#### 第一步：条件化归一化

$$
u_i^{(l)}
=\operatorname{Norm}(h_i^{(l)};c).
$$

负责这一箭头的是 **conditioned normalization**。它一方面把不同位置的隐状态放到稳定的表示尺度，另一方面使用 $c$ 让同一 Transformer 层在不同 $t$ 和 $\beta$ 下采用不同的特征调制：

$$
(h_i^{(l)},c)
\xrightarrow{\ \operatorname{Norm}\ }
u_i^{(l)}.
$$

#### 第二步：产生 query、key 和 value

归一化后的隐状态分别投影为

$$
q_i^{(l)}=W_Q^{(l)}u_i^{(l)},
$$

$$
k_i^{(l)}=W_K^{(l)}u_i^{(l)},
$$

$$
v_i^{(l)}=W_V^{(l)}u_i^{(l)}.
$$

三个符号的角色分别是：

- $q_i^{(l)}$：位置 $i$ 正在寻找什么信息；
- $k_j^{(l)}$：位置 $j$ 能以什么特征被其他位置匹配；
- $v_j^{(l)}$：若位置 $j$ 被关注，它实际传递什么内容。

负责这一箭头的是 attention 内部的 **QKV projection**：

$$
u_i^{(l)}
\xrightarrow{\ W_Q,W_K,W_V\ }
(q_i^{(l)},k_i^{(l)},v_i^{(l)}).
$$

#### 第三步：二维 RoPE 注入相对位置

对 row attention，列坐标 $y_i$ 作用于 query 和 key：

$$
q_{i,\mathrm{row}}^{(l)}
=R(y_i)q_i^{(l)},
\qquad
k_{i,\mathrm{row}}^{(l)}
=R(y_i)k_i^{(l)}.
$$

对 column attention，行坐标 $x_i$ 作用于 query 和 key：

$$
q_{i,\mathrm{col}}^{(l)}
=R(x_i)q_i^{(l)},
\qquad
k_{i,\mathrm{col}}^{(l)}
=R(x_i)k_i^{(l)}.
$$

负责这一箭头的是 **axial 2D RoPE**。它不改变自旋 token，而是改变 query 与 key 的比较方式，使 attention score 同时依赖内容和二维相对位置。

#### 第四步：row attention 汇总同一行的信息

位置 $i$ 对同一行位置 $j$ 的权重为

$$
\alpha_{ij,\mathrm{row}}^{(l)}
=\operatorname{softmax}_{j:x_j=x_i}
\left(
\frac{
\langle q_{i,\mathrm{row}}^{(l)},
k_{j,\mathrm{row}}^{(l)}\rangle
}{\sqrt d}
\right).
$$

row attention 输出为

$$
a_{i,\mathrm{row}}^{(l)}
=\sum_{j:x_j=x_i}
\alpha_{ij,\mathrm{row}}^{(l)}v_j^{(l)}.
$$

这里 $\alpha_{ij,\mathrm{row}}^{(l)}$ 表示同一行的位置 $j$ 对位置 $i$ 当前更新的影响程度。负责该变换的是 **row self-attention**。

#### 第五步：column attention 汇总同一列的信息

同理，column attention 得到

$$
\alpha_{ij,\mathrm{col}}^{(l)}
=\operatorname{softmax}_{j:y_j=y_i}
\left(
\frac{
\langle q_{i,\mathrm{col}}^{(l)},
k_{j,\mathrm{col}}^{(l)}\rangle
}{\sqrt d}
\right),
$$

$$
a_{i,\mathrm{col}}^{(l)}
=\sum_{j:y_j=y_i}
\alpha_{ij,\mathrm{col}}^{(l)}v_j^{(l)}.
$$

负责该变换的是 **column self-attention**。它让一个位置读取同一列上可见自旋、mask 状态和已经形成的抽象特征。

#### 第六步：合并两个空间方向

$$
a_i^{(l)}
=\frac{
a_{i,\mathrm{row}}^{(l)}
+a_{i,\mathrm{col}}^{(l)}
}{\sqrt2}.
$$

负责这一箭头的是 **axial attention merge**。$a_i^{(l)}$ 是位置 $i$ 从整行和整列收到的综合空间消息。

#### 第七步：attention residual 更新

$$
\widetilde h_i^{(l)}
=h_i^{(l)}
+g_{\mathrm{attn}}^{(l)}(c)
\odot a_i^{(l)}.
$$

负责这一箭头的是 **conditioned residual gate**。它由 $t$ 和 $\beta$ 的联合条件控制空间消息写回当前隐状态的方式。加号表示保留旧状态并叠加新信息，而不是完全用 attention 输出覆盖旧状态。

#### 第八步：逐位置 MLP 更新

$$
h_i^{(l+1)}
=\widetilde h_i^{(l)}
+g_{\mathrm{mlp}}^{(l)}(c)
\odot
F^{(l)}
\left(
\operatorname{Norm}(\widetilde h_i^{(l)};c)
\right).
$$

负责这一箭头的是 **position-wise MLP 与第二个 residual connection**。attention 负责在格点之间搬运信息，MLP 则在每个格点内部组合这些信息并形成新的非线性特征。

经过全部 $K_{\mathrm{layer}}$ 层后，数据流为

$$
\{h_i^{(0)}\}
\xrightarrow{\ \text{axial Transformer layers}\ }
\{h_i^{(K_{\mathrm{layer}})}\}.
$$

各结构的职责可以概括为：

| 结构 | 输入 | 输出 | 在数据流中的作用 |
|---|---|---|---|
| token embedding | $(X_t)_i$ | $e_i$ | 把离散自旋或 mask 变成连续表示 |
| time conditioner | $t$ | $c_t$ | 标记当前去噪阶段 |
| beta conditioner | $\beta$ | $c_\beta$ | 指定目标物理分布 |
| conditioned normalization | $h_i^{(l)},c$ | $u_i^{(l)}$ | 用 $t,\beta$ 调制当前层输入 |
| QKV projection | $u_i^{(l)}$ | $q_i,k_i,v_i$ | 形成信息检索与传递表示 |
| axial 2D RoPE | Q、K 与坐标 | 带位置的 Q、K | 把二维相对位置放入 attention score |
| row attention | 同一行隐状态 | $a_{i,\mathrm{row}}$ | 沿水平方向交换信息 |
| column attention | 同一列隐状态 | $a_{i,\mathrm{col}}$ | 沿垂直方向交换信息 |
| residual gate | 旧状态与空间消息 | $\widetilde h_i$ | 保留旧信息并写入 attention 结果 |
| position-wise MLP | $\widetilde h_i$ | $h_i^{(l+1)}$ | 在单点内部非线性组合特征 |

### 19.4 从最终隐状态变成自旋概率

最终隐状态先经过输出归一化与 binary clean-spin head：

$$
h_i^{(K_{\mathrm{layer}})}
\xrightarrow{\ \operatorname{Norm}\ }
\bar h_i,
$$

$$
\bar h_i
\xrightarrow{\ W_{\mathrm{out}}\ }
(\ell_{i,-},\ell_{i,+}).
$$

$\ell_{i,-}$ 和 $\ell_{i,+}$ 分别是位置 $i$ 对 clean spin $-1$ 与 $+1$ 的 logits。它们再经过 softmax：

$$
p_{i,-}
=p_\theta((X_0)_i=-1\mid X_t,t,\beta),
$$

$$
p_{i,+}
=p_\theta((X_0)_i=+1\mid X_t,t,\beta).
$$

负责这一段的结构依次是 **final normalization、binary output head 和 softmax**：

$$
\{h_i^{(K_{\mathrm{layer}})}\}
\longrightarrow
\{\ell_{i,-},\ell_{i,+}\}
\longrightarrow
p_\theta(X_0\mid X_t,t,\beta).
$$

符号 $p_\theta$ 中：

- $p$ 表示条件概率分布；
- $\theta$ 表示 token embedding、conditioner、attention、MLP 和 output head 的全部可学习参数；
- 条件竖线 $\mid$ 表示右侧的 $X_t,t,\beta$ 是已知条件；
- $X_0$ 表示模型试图恢复的 clean field，不表示当前输入已经知道 $X_0$。

Transformer 到这里结束。它输出的是 clean spin 概率，还没有直接产生下一扩散状态 $X_s$。

### 19.5 从自旋概率变成下一状态 $X_s$

令 $s<t$ 是下一个更接近 clean field 的反向时间。先定义当前 mask 集合

$$
\mathcal M_t
=\{i:(X_t)_i=\mathrm{MASK}\}.
$$

对每个 $i\in\mathcal M_t$，从模型 posterior 抽取候选自旋

$$
Y_i
\sim
\operatorname{Categorical}(p_{i,-},p_{i,+}),
\qquad
Y_i\in\{-1,+1\}.
$$

然后产生 reveal 变量

$$
R_i
\sim
\operatorname{Bernoulli}
\left(1-\frac{s}{t}\right).
$$

下一状态逐位置定义为

$$
(X_s)_i
=
\begin{cases}
(X_t)_i,
& i\notin\mathcal M_t,\\
Y_i,
& i\in\mathcal M_t\ \text{且}\ R_i=1,\\
\mathrm{MASK},
& i\in\mathcal M_t\ \text{且}\ R_i=0.
\end{cases}
$$

因此

$$
p_\theta(X_0\mid X_t,t,\beta)
\longrightarrow
X_s
$$

并不是由另一个神经网络完成，而是由两部分共同完成：

1. Transformer posterior 决定被揭示位置应取 $-1$ 还是 $+1$；
2. absorbing reverse rule 决定哪些 mask 在本步被揭示。

已经揭示的位置直接从 $X_t$ 复制到 $X_s$，不会再次被 mask 或改写。尚未揭示的位置继续作为 `MASK` 进入下一次 Transformer 前向传播。

### 19.6 多次重复怎样得到最终构型

定义反向时间序列

$$
1=t_{K_{\mathrm{rev}}}
>t_{K_{\mathrm{rev}}-1}
>\cdots
>t_1
>t_0=0.
$$

其中：

- $K_{\mathrm{rev}}$ 是反向生成包含的状态转换次数；
- 下标 $k$ 表示反向轨迹中的阶段；
- $t_k$ 是第 $k$ 个阶段的扩散时间；
- 它们都不是 Transformer 层编号。

完整轨迹为

$$
X_{t_{K_{\mathrm{rev}}}}
=X_1
\longrightarrow
X_{t_{K_{\mathrm{rev}}-1}}
\longrightarrow
\cdots
\longrightarrow
X_{t_1}
\longrightarrow
X_{t_0}
=\widehat X_0.
$$

每个箭头内部都重复同一组结构：

$$
X_t
\xrightarrow{\ \text{token embedding}\ }
h^{(0)}
\xrightarrow{\ \text{conditioned axial Transformer}\ }
h^{(K_{\mathrm{layer}})}
\xrightarrow{\ \text{binary head}\ }
p_\theta(X_0\mid X_t,t,\beta)
\xrightarrow{\ \text{reverse reveal}\ }
X_s.
$$

最终帽子符号 $\widehat X_0$ 表示“模型生成的 clean field”，用来与 Monte Carlo 提供的真实 clean field $X_0$ 区分。二者都只包含 $-1$ 和 $+1$，但来源不同。

### 19.7 从最终构型变成物理观测量

最后一步为

$$
\widehat X_0
\longrightarrow
\left\{
e,m,U_4,G(r),G_c(r),S(k),\ldots
\right\}.
$$

这个箭头不属于 Transformer，也不属于 reverse diffusion。它表示在一组生成样本上进行外部物理测量。

令 $N=HW$，生成自旋记为 $\widehat s_i$。各符号含义如下。

#### 能量密度 $e$

$$
e
=\frac{1}{N}
\mathbb E_{\widehat X_0}
\left[H(\widehat X_0)\right].
$$

$e$ 衡量相邻格点整体上倾向同向还是反向。它由最终自旋代入 Hamiltonian 得到，不是前文 token embedding $e_i$。

#### 磁化密度 $m$

$$
m
=\frac{1}{N}
\sum_i\widehat s_i.
$$

$m$ 衡量一张构型整体偏向 $+1$ 还是 $-1$。评估分布时还会比较 $m$ 的完整样本分布和 $|m|$。

#### Binder 累积量 $U_4$

$$
U_4
=1-
\frac{\mathbb E[m^4]}
{3\,\mathbb E[m^2]^2}.
$$

$U_4$ 由许多生成构型的磁化矩计算，描述磁化分布形状，不是单张构型的局部量。

#### 两点关联 $G(r)$

$$
G(r)
=\mathbb E
\left[
\widehat s_i\widehat s_{i+r}
\right].
$$

$G(r)$ 衡量相距 $r$ 的两个自旋同向或反向的统计倾向，是判断长程结构的主要输出指标。

#### 连通关联 $G_c(r)$

$$
G_c(r)
=G(r)-\mathbb E[\widehat s]^2.
$$

$G_c(r)$ 中的减号用于去除仅由非零平均磁化造成的背景平台，使剩余部分更直接表示共同涨落。

#### 结构因子 $S(k)$

$$
S(k)
=\frac{1}{N}
\mathbb E
\left[
\left|
\sum_j
\widehat s_j e^{\mathrm i k\cdot r_j}
\right|^2
\right].
$$

$k$ 是空间波数，$r_j$ 是格点 $j$ 的二维坐标。$S(k)$ 把实空间的关联转换成不同空间尺度上的波动强度，低 $k$ 对应长波长、长距离结构。

这些物理量只读取 $\widehat X_0$，不会把信息反馈给已经完成的生成轨迹。除非另行定义物理正则项，否则它们也不参与模型训练 loss。

### 19.8 容易混淆的负号与减号

如果“负号”是指文中实际出现的负号，它们分别有以下含义：

1. 自旋 $-1$ 是 Ising 二值状态之一，与 $+1$ 对称；它不是负概率，也不是“错误 token”。
2. 铁磁 Hamiltonian 中

   $$
   H(X)=-J\sum_{\langle i,j\rangle}s_i s_j
   $$

   的负号表示当 $J>0$ 时，相邻同向自旋使能量降低。
3. 连通关联

   $$
   G_c(r)=G(r)-\mathbb E[s]^2
   $$

   的减号表示扣除平均磁化产生的非连通背景。
4. 临界幂律

   $$
   G_c(r)\propto r^{-1/4}
   $$

   中的负指数表示关联随距离增加而衰减；它不表示关联函数必然为负。
5. 输出 logit 的下标 $\ell_{i,-}$ 中，$-$ 是 clean spin 类别 $-1$ 的标签，不是对 logit 取负。

综上，生成数据流中真正把一个状态变成下一状态的核心链条是：**embedding 把离散状态变成向量，conditioner 注入 $t$ 和 $\beta$，二维 RoPE 与 row/column attention 交换空间信息，MLP 在单点内重组信息，binary head 产生自旋 posterior，absorbing reverse rule 再把 posterior 变成下一部分揭示的构型。**

## 20. 模型架构的核心假设

该设计建立在以下假设上：

1. mask reconstruction 能迫使模型学习 Ising 条件分布，而不只是无条件像素频率；
2. 显式 $t$ 条件能够让一个网络描述整个去噪轨迹；
3. 可选 $\beta$ 条件能够让一个网络区分不同物理分布；
4. 二维相对位置表示比 raster 一维位置更符合 Ising 几何；
5. 行列 attention 的多层组合能够传播局部与长程信息；
6. 多步 absorbing reverse 可以把逐位置 posterior 组合成具有非平凡空间关联的联合分布；
7. 不显式输入目标尺寸，可以把更大网格行为作为真正的 context/scale extrapolation 测试。

这些是假设而不是结论。模型是否真的学到临界长程物理，最终必须由训练尺寸内和未见尺寸上的生成分布共同验证。

## 21. 本文刻意不包含的内容

为了保持“数学模型与数据流”这一视角，本文不规定：

- Transformer 的具体层数、宽度或 head 数；
- batch size、learning rate 或训练步数；
- dense、Flash 或其他 attention kernel；
- 显存、吞吐量或复杂度估算；
- DDP、FSDP、编译和设备配置；
- 数据文件格式、代码目录与程序接口；
- checkpoint、日志和任务调度方式。

这些内容属于第二个视角——infra 设计，不影响本文定义的概率模型和数据流边界。
