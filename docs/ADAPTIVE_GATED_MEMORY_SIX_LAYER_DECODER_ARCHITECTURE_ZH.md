# 自适应门控事实记忆六层小型 Decoder 架构说明

## 1. 总体概述

本文设计的模型是一个带有固定局部状态和有界长期事实记忆的六层
decoder-only TinyLM。它不是标准的完整自注意力 Transformer，而是将不同时间尺度
的信息分配给不同模块：

- 局部滑动窗口注意力负责保存最近的逐 token 上下文；
- attention sink 负责稳定局部注意力分布；
- 内容门控事实记忆负责保存需要跨越长干扰上下文的事实；
- value-token 对齐路径负责从连续记忆表示中恢复具体答案词元；
- Dense FFN 或 MoE 负责对每个 token 进行非线性变换。

整体数据流可以表示为：

```text
输入 token
   │
   ├── Token Embedding + RoPE
   │
   ├── 第 1 层 Decoder Block
   ├── 第 2 层 Decoder Block
   ├── 第 3 层 Decoder Block + 长期记忆融合
   ├── 第 4 层 Decoder Block
   ├── 第 5 层 Decoder Block
   └── 第 6 层 Decoder Block + 长期记忆融合
   │
   ├── Final LayerNorm
   └── tied LM Head
          │
       token logits
```

长期记忆路径与六层 decoder 并行工作：

```text
已有长期记忆
   │
   ├── 构造查询向量
   ├── 计算 key 相似度
   ├── Top-K 稀疏读取
   └── 在第 3、6 层进行门控融合

当前用户输入
   │
   ├── 事实候选抽取
   ├── 写入门
   ├── 合并/更新门
   ├── 保留门
   └── 当前轮结束后提交到长期记忆
```

模型的核心思想是：局部 attention 不再承担无限期历史保存任务，长期槽位也不保存
全部历史，而是只保存经过内容判断后值得跨轮保留的少量事实。

## 2. 基本模型配置

TinyLM 主干的主要配置如下：

| 项目 | 设置 |
|---|---:|
| 模型类型 | decoder-only causal language model |
| Transformer 层数 | 6 |
| hidden size | 384 |
| 注意力头数 | 8 |
| 每个 head 的维度 | 48 |
| Dense FFN 隐层维度 | 1536 |
| 词表大小 | 50,257 |
| 位置编码 | RoPE |
| 最大位置 | 32,768 |
| dropout | 0.1 |
| 输入/输出嵌入 | 权重绑定 |
| 局部 KV 预算 | 128 |
| sink token 数 | 4 |
| recent token 数 | 124 |

因此，对于 batch 大小为 (B)、序列长度为 (T) 的输入，主隐藏状态的形状为：

\[
H\in\mathbb{R}^{B\times T\times384}.
\]

最终语言模型输出的形状为：

\[
\mathrm{logits}\in\mathbb{R}^{B\times T\times50257}.
\]

需要区分代码默认配置和论文压力测试配置。代码中的默认长期记忆为 64 个槽位、
reader Top-8；论文中的选择性事实记忆压力测试为了使容量竞争可审计，使用 8 个用户
事实槽位、Top-4 reader，并将每轮候选数限制为 1。

## 3. 输入嵌入与位置编码

给定 token 序列：

\[
x_1,x_2,\ldots,x_T,
\]

模型首先通过词嵌入矩阵得到初始向量：

\[
h_j^{(0)}=E[x_j],
\]

其中：

\[
E\in\mathbb{R}^{50257\times384},
\qquad
h_j^{(0)}\in\mathbb{R}^{384}.
\]

这些初始向量只表示 token 的词表嵌入。经过六层 Transformer 后，向量才会包含
上下文、局部语义和必要的历史记忆信息。

RoPE 作用于每一层注意力的 query 和 key。模型保留绝对位置编号，即使旧的非 sink
KV 被移出缓存，仍不会对剩余位置进行错误的重新编号。

## 4. Decoder Block 的基本结构

每个 Decoder Block 使用 Pre-LN 和残差连接。对第 \(\ell\) 层，可以概括写成：

\[
u^{(\ell)}=
h^{(\ell-1)}+
\operatorname{Dropout}
\left(
\operatorname{LocalAttn}^{(\ell)}
\left(
\operatorname{LN}_{\mathrm{local}}^{(\ell)}
\left(h^{(\ell-1)}\right)
\right)
\right).
\]

在第 3 层和第 6 层，局部注意力之后还会加入长期记忆残差：

\[
\tilde u^{(\ell)}=
u^{(\ell)}+
\operatorname{Dropout}
\left(
g^{(\ell)}F_{\ell}
\left(u^{(\ell)},\tilde m\right)
\right).
\]

最后经过 Dense FFN 或 MoE：

\[
h^{(\ell)}=
\tilde u^{(\ell)}+
\operatorname{Dropout}
\left(
\operatorname{FFN/MoE}^{(\ell)}
\left(
\operatorname{LN}_{\mathrm{ffn}}^{(\ell)}
\left(\tilde u^{(\ell)}\right)
\right)
\right).
\]

因此六层并不是六个完全不同的网络，而是六个结构相同的 decoder block；主要区别
在于第 3、6 层额外启用长期记忆融合。

## 5. 固定局部 KV 的 Sink-Sliding Attention

### 5.1 Query、Key、Value 投影

每层先将隐藏状态投影为 query、key 和 value：

\[
Q=hW_Q,
\qquad
K=hW_K,
\qquad
V=hW_V.
\]

每个投影结果的最后一维为 384，然后拆分为 8 个注意力头：

\[
Q,K,V\in\mathbb{R}^{B\times8\times T\times48}.
\]

RoPE 只作用于 query 和 key：

\[
(Q',K')=\operatorname{RoPE}(Q,K,\mathrm{position\_ids}).
\]

### 5.2 Sink cache

模型永久保留对话开头的 4 个位置：

\[
\mathcal C_{\mathrm{sink}}=\{0,1,2,3\}.
\]

这些 token 的主要作用是稳定注意力分布。它们不一定具有明确的语义，也不是专门
保存事实的 memory slot。

你的 sink token 与 Longformer 的 global token 不同：

- sink token 只是作为 key/value 被保留；
- sink query 不会因此获得双向注意力；
- sink token 不能访问整个序列；
- sink token 不负责实体、关系和值的显式存储。

### 5.3 Recent window

除 sink 之外，每层只保存最近的 124 个非 sink token：

\[
\mathcal C_{\mathrm{recent}}(t)
=
\text{最近的 124 个非 sink token}.
\]

当新 token 到来时，最旧的非 sink KV 被移出缓存。缓存填满后，单个 query 能看到的
唯一 key 数量最多为：

\[
|\mathcal C_t|
\leq4+124=128.
\]

每个 query 仍然受到 causal 约束：

\[
i\leq t.
\]

局部注意力可以写为：

\[
a_{t,i}=
\operatorname{softmax}_i
\left(
\frac{q_t^{\top}k_i}{\sqrt{48}}
\right),
\qquad i\in\mathcal C_t,
\]

\[
o_t=\sum_{i\in\mathcal C_t}a_{t,i}v_i.
\]

随后通过输出投影并加入残差：

\[
h_{\mathrm{local}}^{(\ell)}=
h^{(\ell-1)}+
\operatorname{Dropout}(W_Oo^{(\ell)}).
\]

### 5.4 局部状态的优势与边界

局部窗口使注意力状态不再随完整历史长度无限增长。对于固定预算 \(B=128\)，
其计算规模近似为：

\[
\mathcal O(TBD),
\]

而完整 causal attention 的注意力计算规模近似为：

\[
\mathcal O(T^2D).
\]

但局部窗口本身不具备内容级长期记忆能力。一个重要事实一旦离开 recent window，
除非通过多层局部传播间接保留，否则不能被当前 token 直接访问。因此，SWA-sink
只负责短期上下文，长期事实由独立的记忆槽位处理。

## 6. 第 3 层和第 6 层的长期记忆融合

代码中的配置为：

```python
memory_fusion_layers = (2, 5)
```

由于代码从 0 开始编号，索引 2 和 5 分别对应第 3 层和第 6 层。

将记忆融合放在这两层有三个目的：

1. 第 3 层已经经过局部注意力，当前 token 具有初步上下文表示；
2. 第 6 层靠近输出端，可以在答案生成前再次注入事实；
3. 其余层保持纯局部 decoder 计算，避免每层都受到外部记忆干扰。

## 7. 长期事实记忆的读取

长期记忆槽位可以表示为：

\[
\mathcal M=\{(k_r,v_r)\}_{r=1}^{M}.
\]

其中：

- \(k_r\) 是用于检索的 key；
- \(v_r\) 是保存事实语义的 value；
- 槽位还保存 payload、active、confidence、age、访问次数等元数据。

### 7.1 查询向量

新一轮用户输入开始时，模型从 prompt 边界构造查询表示 \(u_t\)，并投影为：

\[
q_t=\operatorname{Normalize}(W_qu_t).
\]

### 7.2 Top-K 稀疏读取

查询与每个活跃槽位的 key 计算相似度：

\[
s_r=q_t^{\top}k_r.
\]

失活槽位被屏蔽，最后只选择得分最高的 \(K\) 个槽位：

\[
\mathcal I_t=\operatorname{TopK}_r(s_r).
\]

论文压力测试中使用 \(K=4\)。即使长期记忆有 8 个用户槽位，当前一轮也最多将
4 个槽位交给融合模块。

这一步是“稀疏读取”，它回答：

> 当前问题需要哪几条历史事实？

reader 通常在一轮开始时执行一次。选中的槽位在本轮助手生成期间保持不变，避免
每个 token 重新检索导致路由抖动。

## 8. 记忆交叉注意力和融合门

在第 3 层和第 6 层，当前 token 与 Top-K 记忆进行小规模交叉注意力。

当前 token 的 query 为：

\[
q_t^{\mathrm{tok}}=W_qh_t.
\]

记忆 key/value 分别为：

\[
k_r^{\mathrm{mem}}=W_kk_r,
\qquad
v_r^{\mathrm{mem}}=W_vm_r.
\]

根据模型配置，记忆 value 可以是：

\[
m_r=v_r+\text{payload summary},
\]

也可以是：

\[
m_r=v_r+z_r,
\]

其中 \(z_r\) 是 value-token alignment 路径得到的词汇值表示。

记忆交叉注意力为：

\[
\alpha_{t,r}=
\operatorname{softmax}_r
\left(
\frac{(q_t^{\mathrm{tok}})^{\top}k_r^{\mathrm{mem}}}
{\sqrt{384}}
\right),
\]

\[
c_t=\sum_{r\in\mathcal I_t}
\alpha_{t,r}v_r^{\mathrm{mem}}.
\]

其中 \(c_t\) 是当前 token 对长期记忆的汇总上下文。

模型随后计算融合门：

\[
g_t=\sigma\left(W_g[h_t;c_t]+b_g\right).
\]

最终记忆残差为：

\[
r_t^{\mathrm{mem}}=g_tW_oc_t,
\]

并加入当前 token 状态：

\[
h_t'=h_t+r_t^{\mathrm{mem}}.
\]

因此：

- reader 决定“读取哪条记忆”；
- fusion gate 决定“记忆影响当前 token 多少”。

如果 \(g_t\) 接近 1，说明当前 token 强烈依赖长期事实；如果 \(g_t\) 接近 0，
说明当前 token 主要依赖局部上下文。

## 9. Dense FFN 和 MoE FFN

### 9.1 Dense FFN

Dense 模式下，每层的 FFN 为：

\[
f^{(\ell)}=
W_{\mathrm{out}}^{(\ell)}
\operatorname{GELU}
\left(
W_{\mathrm{in}}^{(\ell)}
\operatorname{LN}_{\mathrm{ffn}}^{(\ell)}(h^{(\ell)})
\right),
\]

其中：

\[
W_{\mathrm{in}}\in\mathbb{R}^{384\times1536},
\qquad
W_{\mathrm{out}}\in\mathbb{R}^{1536\times384}.
\]

Dense FFN 的每个 token 都通过同一组前馈参数。

### 9.2 MoE FFN

MoE 只替换 FFN 子层，不改变局部 attention、长期记忆、reader 或 value-token 路径。

路由器先计算专家概率：

\[
p(h)=\operatorname{softmax}(W_rh+b_r).
\]

假设有 4 个专家并使用 Top-2 路由，则每个 token 选择两个专家：

\[
\mathcal T(h)=\operatorname{Top2}(p(h)).
\]

每个专家都是一个独立的 Dense FFN：

\[
F_e(h)=
W_{e,\mathrm{out}}
\operatorname{GELU}(W_{e,\mathrm{in}}h).
\]

最终输出为：

\[
\operatorname{MoE}(h)=
\sum_{e\in\mathcal T(h)}
\tilde p_e(h)F_e(h),
\]

其中 \(\tilde p_e\) 是在选中专家内部重新归一化的路由权重。

当前实现使用 dropless dispatch：

- 有效 token 都会被送到其选中的专家；
- 不会因为专家容量上限而丢弃 token；
- padding token 不参与路由统计；
- 训练时加入负载均衡损失；
- 同时记录专家负载、dispatch fraction 和 routing entropy。

论文中的事实记忆压力测试使用 4 expert、Top-2 MoE；六种基础架构统一比较报告中
还存在 4 expert、Top-1 配置。这两类结果属于不同实验协议，不能直接当作同一个 MoE
设置解释。

## 10. 最终归一化与语言模型输出

六层计算完成后，模型进行最终层归一化：

\[
H_{\mathrm{final}}=\operatorname{LayerNorm}(H^{(6)}).
\]

padding 位置会通过 mask 被置零。

由于默认使用 tied embeddings，输出层直接使用输入词嵌入矩阵的转置：

\[
\operatorname{logits}=H_{\mathrm{final}}E^{\top}.
\]

在单个位置上：

\[
\operatorname{logits}_t
=h_tE^{\top}
\in\mathbb{R}^{50257}.
\]

最后通过 softmax 得到下一个 token 的概率：

\[
P(x_{t+1}\mid x_{\leq t})
=\operatorname{softmax}(\operatorname{logits}_t).
\]

这条 tied LM head 是旧 Gated 模型出现词级生成瓶颈的重要原因之一：语义记忆向量
最终必须落到正确答案 token 的词嵌入方向上。

## 11. 事实抽取和写入生命周期

### 11.1 事实候选抽取

当前用户输入经过六层 decoder 后得到最终隐藏状态：

\[
h_1,h_2,\ldots,h_T.
\]

事实抽取器在 user-role token 上预测：

- 事实候选起点；
- 事实长度；
- 写入概率；
- 事实置信度；
- 可选的 value 起点和 value 长度。

一个事实片段 \(S_i\) 的整体表示为：

\[
\bar h_i=
\frac{1}{|S_i|}\sum_{j\in S_i}h_j.
\]

实际代码使用 mask mean，只对片段内有效的 user token 求平均。

随后得到：

\[
k_i=\operatorname{Normalize}(W_k\bar h_i),
\qquad
v_i=W_v\bar h_i.
\]

如果启用 value-token alignment，还会对 value 子片段 \(V_i\) 单独聚合：

\[
z_i=W_{\mathrm{lex}}
\left(
\frac{1}{|V_i|}\sum_{j\in V_i}h_j
\right).
\]

### 11.2 写入、合并、更新和保留

写入门输出：

\[
p_i^{\mathrm{write}}
=\sigma(f_{\mathrm{write}}(\bar h_i)).
\]

超过写入阈值的候选才会被接受。

新事实先与已有 key 计算相似度。如果：

\[
\operatorname{sim}(k_i,k_r)
\geq\tau_{\mathrm{merge}},
\]

则将其视为已有事实的更新，并通过更新门插值：

\[
v_r'=(1-g_{i,r}^{\mathrm{upd}})v_r+
g_{i,r}^{\mathrm{upd}}v_i.
\]

如果没有足够相似的槽位，则优先写入空槽位；槽位已满时，根据保留分数选择淘汰
对象。

保留门计算：

\[
p_r^{\mathrm{ret}}
=\sigma(f_{\mathrm{ret}}(k_r,v_r,\operatorname{meta}_r)).
\]

元数据包括：

- active strength；
- confidence；
- age；
- access count；
- conflict flag；
- last access time；
- source role。

低于阈值的槽位会在下一次 commit 边界被释放。

三个主要控制器的职责是：

| 控制器 | 判断对象 | 作用 |
|---|---|---|
| 写入门 | 新事实候选 | 是否值得进入长期状态 |
| 更新门 | 新事实与目标槽位 | 新旧事实混合多少 |
| 保留门 | 已有记忆槽位 | 是否继续占用容量 |

### 11.3 轮次级提交

模型把一次用户输入和随后的助手回答视作一个 round。其生命周期为：

```text
用户输入
   │
   ├── 读取上一轮已经存在的记忆
   ├── 计算当前轮 logits
   ├── 当前用户事实进入 pending buffer
   ├── 生成助手回答
   └── 当前轮结束后提交新事实
```

当前轮候选事实不会立即进入可读记忆，而是先进入 pending buffer。只有在当前轮 logits
全部计算完成并收到 commit 信号后，才执行写入、保留、合并和更新。

因此：

\[
\text{当前轮使用 }\mathcal M_t,
\qquad
\text{当前轮结束后更新为 }\mathcal M_{t+1}.
\]

新事实只能影响下一轮，不能帮助模型预测创建该事实的当前回答。这一设计避免了
“答案先被写入、再帮助自己生成”的未来信息泄漏。

代码中还保留了 4 个 assistant summary slots，用于保存助手侧的轮次摘要；但当前
选择性事实压力测试主要研究 user fact memory，并通过较低的 assistant write bias
抑制了意外的助手槽位写入。

## 12. Value-token 对齐路径

旧 Gated 使用完整事实的平均表示作为 semantic value。例如：

```text
Remember Bob likes blue.
```

人物、关系、颜色、提示词和标点可能全部混合在一个连续向量中。这个向量足以支持
语义检索，却未必能让 `blue` 在词表中稳定排第一。

Value-token Gated 额外识别 value span，例如：

```text
blue
```

并得到词汇值表示：

\[
z_i=W_{\mathrm{lex}}
\left(
\frac{1}{|V_i|}\sum_{j\in V_i}h_j
\right).
\]

融合时使用：

\[
m_i=v_i+z_i.
\]

其中：

- \(v_i\) 负责完整事实和关系语义；
- \(z_i\) 负责答案值的词汇身份；
- value token ID 和 mask 用于保留原始词元；
- memory-to-token auxiliary head 直接监督值词元。

辅助头为答案位置加入可学习的位置表示，并与词嵌入矩阵共享输出投影。它只在训练
时提供约束，推理时仍由普通自回归 tied LM head 生成答案，没有使用 pointer/copy
机制。

## 13. 状态规模与计算复杂度

### 13.1 局部状态

每层局部状态包括：

\[
4\text{ 个 sink KV}+124\text{ 个 recent KV}.
\]

六层总局部状态近似为：

\[
\mathcal O(6\times128\times384).
\]

### 13.2 长期状态

长期状态包括：

- semantic key；
- semantic value；
- lexical value；
- payload token ID；
- value token ID；
- payload mask；
- active、confidence、age；
- access count、last access 和 conflict 标记。

若使用 (M) 个槽位、每个槽位最多保存 (P=12) 个 token，则长期状态近似为：

\[
\mathcal O(MD+MP),
\qquad D=384.
\]

整体状态规模可以概括为：

\[
\mathcal O(NBD)+\mathcal O(MD+MP),
\]

其中 (N=6)、(B=128)。因此，随着对话历史继续增长，状态上界保持不变。

但“逻辑活跃槽位少”不等于“物理显存已经减少”：当前实现按最大容量预分配槽位
张量。若要让活跃槽位真正带来显存收益，还需要进一步实现物理 packing 或动态分配。

## 14. 与普通六层 Transformer 的区别

普通六层 decoder 可以概括为：

```text
Token Embedding
→ Full Causal Attention
→ Dense FFN
→ 重复 6 层
→ LM Head
```

你的模型则是：

```text
Token Embedding
→ 六层局部 causal decoder
   ├── 每层使用 4 sink + 124 recent KV
   ├── 第 3、6 层读取长期事实
   ├── FFN 可替换为 MoE
   ├── 轮次边界更新事实槽位
   └── value-token 路径辅助精确恢复
→ Final LayerNorm
→ Tied LM Head
```

其职责分工可以总结为：

\[
\boxed{\text{局部 KV}\Rightarrow\text{短期逐 token 上下文}}
\]

\[
\boxed{\text{门控事实槽位}\Rightarrow\text{跨轮长期事实}}
\]

\[
\boxed{\text{value-token 路径}\Rightarrow\text{精确答案词元恢复}}
\]

\[
\boxed{\text{MoE}\Rightarrow\text{条件 FFN 容量}}
\]

最核心的设计不是简单缩小 Transformer，而是把不同时间尺度和不同信息类型分开：

- 最近 token 由局部注意力直接保存；
- 重要但久远的事实由内容门控槽位保存；
- 写入、更新和保留由不同控制器负责；
- reader 只读取少量相关槽位；
- fusion gate 决定记忆对每个 token 的影响强度；
- value-token alignment 负责把连续语义记忆映射回离散答案 token；
- MoE 只扩展前馈网络容量，不替代长期记忆机制。

