# Sliding-window beats linear attention：代码来源审计与基线定义

更新时间：2026-09-09  
论文：*Sliding-window beats linear attention*，arXiv:2608.28444v1

## 结论先行

截至上述日期，未找到该论文作者公开的、可确认的 GitHub 代码仓库。
因此，当前不能把任何下载的实现称为“论文原始代码复现”。这不是把论文
的 SWA 机制重新发明一遍，而是论文没有提供可下载的作者实现。

本项目已经下载并冻结了论文所依赖的、最接近且可审计的上游实现：

| 用途 | 官方仓库 | 冻结 revision | 本地目录 | 关键实现 |
|---|---|---|---|---|
| Attention Sinks / sink-cache 基线 | <https://github.com/mit-han-lab/streaming-llm> | `2e5042606d69933d88fbf909bd77907456b9b4dd` | `references/streaming-llm/` | `streaming_llm/kv_cache.py::StartRecentKVCache` |
| Google 官方局部注意力参考 | <https://github.com/google/gemma_pytorch> | `014acb7ac4563a5f77c76d7ff98f31b568c16508` | `references/gemma_pytorch/` | `gemma/model.py` 中 `LOCAL_SLIDING` |

前者是本项目的主 SWA 基线来源；后者只用于核对 Google Gemma 的局部窗口
实现，不能称为本文论文代码，也不等同于本文的纯 SWA(w, 4)。完整来源、
许可证和哈希记录见 [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md)。

## 检查依据

1. arXiv 页面：<https://arxiv.org/abs/2608.28444v1>。
2. arXiv 源码包：<https://export.arxiv.org/e-print/2608.28444v1>。
   `paper_arxiv.tex` 中没有作者仓库、代码归档或实验脚本链接；源码中的
   GitHub URL 仅指向 `goodfeli/dlbook_notation`、文献仓库或被测模型资源。
3. GitHub Repository API 对题名、arXiv 编号 `2608.28444`、作者和相关
   关键词的公开检索没有返回与论文对应的仓库（题名搜索 `total_count=0`）。
4. 论文作者列表为 Alexia Jolicoeur-Martineau、Rhea Sanjay Sukthanker、
   Pashmina Cameron、Emy Gervais；公开作者仓库中也未发现本论文项目。

上述结论是“截至日期”的来源状态；若作者之后发布仓库，应记录新 URL、
commit/tag、下载日期和哈希，并重新做一次原作者实现复现。

## 为什么选择 StreamingLLM 作为基线

论文把方法写成训练无关的

```text
SWA(w, 4) = 4 个初始 attention sinks + 最近窗口中的 w-4 个 token
```

论文引用的 Attention Sinks 工作公开了 `StartRecentKVCache`，其物理操作
正是保留 `start_size` 个开头 KV，再保留 `recent_size` 个最新 KV，并提供
`evict_for_space` 等缓存接口。它是与论文机制最直接对应的权威实现。

需要注意：论文没有说明作者内部是否直接调用 StreamingLLM；下面的选择是
一个可审计的外部复现路径，而不是对作者内部代码的推断。

StreamingLLM 的代码面向 Hugging Face 风格的 `past_key_values`，而本项目的
TinyLM 使用显式的 `LocalKVCache` 状态。因此公平接入 TinyLM 时只允许增加
一个很薄的协议适配层（把相同的 cache policy 接到 TinyLM 的状态格式），
不改变 sink 数量、窗口规则或注意力计算。这类结果应标记为：

```text
StreamingLLM-source SWA baseline
paper-aligned external SWA reference
```

而不是“Sliding-window beats linear attention 官方代码结果”。

## 本项目的基线协议

所有模型必须使用同一个 TinyLM、tokenizer、数据划分、seed、训练步数、
batch/token budget、精度和 GPU。主比较建议为：

```text
TinyLM + StreamingLLM-source SWA(w=128, s=4) only
vs.
TinyLM + SWA(w=128, s=4) + gated fact memory（当前模型）
```

对应当前配置的 `attention_budget=128`、`sink_tokens=4`、
`recent_window=124`。如需与论文表格更贴近，可另做 `SWA(64,4)`，即
`recent_window=60`，但窗口 64 与窗口 128 的结果不能混在同一主表中。

应报告：TinyStories 验证 NLL/PPL、结构化事实 Exact Match、strict read
hit、按 delay 的召回、active slots、state bytes、训练吞吐、解码延迟和
峰值显存。论文中的 1.3B--70B MMLU/BABILong 数字只能作为外部背景，不能
与 TinyLM/TinyStories 数字直接拼成同一统计结论。

## 复现层级与命名规则

| 层级 | 可以声称什么 | 不能声称什么 |
|---|---|---|
| A. 作者代码复现 | 只有在找到作者仓库、固定 commit、原模型和原配置后 | 目前不能声称已完成 |
| B. 上游机制复现 | 使用 `StartRecentKVCache` 的 sink + recent policy | 不能称为本文作者代码 |
| C. 公平 TinyLM 基线 | 同一 TinyLM/数据/预算下的 source-policy adapter | 不能把适配层当作新 SWA 机制 |
| D. 当前创新模型 | SWA-sink 加 gated fact memory | 不能把它的结果当作论文 SWA 单独结果 |

这样既保留源代码的可审计性，也避免因 TinyLM 状态接口不同而强行运行
不适配的 Hugging Face 示例。

## 下一步实验顺序

### 阶段 0：代码级等价性检查（不训练）

在随机 KV 张量上比较 `StreamingSWASourcePolicy(window_size=128,
sink_tokens=4)` 与 TinyLM 当前 `StreamingSinkSlidingAttention` 的保留位置：
两者都应保留位置 `0--3` 和最后 `124` 个非 sink 位置。还要检查因果可见
键数不超过 128。该检查只证明 cache policy 一致，不证明两个实现的 CUDA
速度或所有位置编码行为完全一致。

### 阶段 1：seed 17 最小公平对照

沿用已有的 seed-17 TinyStories backbone 和结构化数据，固定 2,000 个结构化
训练 step、相同 batch/context/optimizer，在同一张 RTX 4090 上运行：

1. `SWA-sink only`：4 sinks + 124 recent，无事实记忆分支；
2. `SWA-sink + gated fact memory`：当前完整模型（已有 seed-17 结果作为
   初始参照：Exact Match 86.67%，strict read hit 100%）。

两者必须使用相同的输入顺序和评估样本。先完成这两个条件，才能回答“长期
记忆模块相对于 SWA 本身带来了多少收益”。可选的第三个条件是无 sink 的
普通 sliding window，用于单独验证 sink 作用；它不是主比较必需项。

如果论文的“训练无关 SWA”主张也要在 TinyLM 上单独验证，还需要另建一个
**全注意力预训练** backbone，然后在不做结构化再训练的情况下切换为
`SWA(64,4)` 或 `SWA(128,4)` 进行验证。现有 seed-17 backbone 本身就是用
sink-SWA 训练得到的，因此不能把它的结果命名为“full-attention checkpoint
的 training-free SWA 转换”。

### 阶段 2：三 seed 统计

当 seed 17 流程稳定后，使用统一协议的 seeds `17, 29, 43`。严格的三 seed
均值 ± 标准差需要为每个 seed 准备对应的 TinyStories backbone，再分别
训练上述两个条件。因此主结果至少是 6 个结构化训练 run；seed 17 已有
完整模型结果，仍需补齐相同协议的 SWA-only 以及 seeds 29/43 的对应 run。

### 阶段 3：效率和消融

对阶段 2 的最佳 checkpoint 统一测量：

- TinyStories 验证 NLL/PPL；
- 结构化 Exact Match、strict read hit 和不同 delay 的召回曲线；
- active slots、state bytes；
- 训练吞吐、单 token 解码延迟、峰值显存。

如需解释机制，再增加固定槽位、无 sink、`SWA(64,4)` 等消融，但不要把
这些不同窗口预算混入主表。

## 与论文的比较边界

当前 TinyLM 实验不能直接复现论文的数值结论。论文使用预训练的
1.3B--70B 模型、MMLU/BABILong/Needle-in-a-Haystack 等任务及特定的
FlashAttention/ThunderKittens 环境；本项目使用 31.9M 参数 TinyLM、
TinyStories 和结构化事实对话。可以比较的是机制趋势、缓存规则和在匹配
TinyLM 条件下的相对收益，不能把两套 Exact Match、PPL 或吞吐数字放在同一
主统计表中。

若以后必须做论文级直接对比，应另开一组外部实验：选择论文列出的同一个
公开预训练模型和任务，用论文定义的 `SWA(64,4)` 或 `SWA(128,4)`，并记录
模型、checkpoint、kernel、硬件和评测脚本。该组结果与 TinyLM 主实验分开
报告。
