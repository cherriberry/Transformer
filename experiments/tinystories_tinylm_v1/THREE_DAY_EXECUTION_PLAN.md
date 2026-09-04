# 三天六框架 TinyLM + TinyStories 验证集实验计划

版本：`three_day_validation_screening_v1`  
性质：时间受限的可比验证集筛选，不是最终论文级主实验

## 1. 实验目标

三天内让六个指定框架都留下可追溯、可比较的结果：

- Longformer、Performer、Linformer、Reformer、Memformer：在同一个 TinyLM 主干上从头训练 TinyStories，并完成 `train → validation`；
- Keyformer：不伪装成训练期 backbone，使用同一个 Full-Attention TinyLM checkpoint，在验证集顺序评估/生成阶段比较 KV-cache 压缩；
- Full Attention：作为质量和效率参考，不计入六个框架名额，但必须额外训练一个共享 checkpoint，供 Keyformer 和所有方法的参考表使用。

本轮主要回答：在相同 TinyLM 主干和训练 token 预算下，六种设计的验证集质量、运行效率、状态/缓存开销和失败边界如何。它不承诺完整收敛、三 seed 统计或长对话记忆结论。

**实现前提：**仓库目前已有六种方法的模块、历史微基准和 WikiText 脚本，但没有直接符合本协议的统一 TinyStories `train → validation` runner。第 1 天必须先完成统一数据流、TinyLM wrapper、方法 adapter、checkpoint/validation probe 和结果 schema；现有历史结果不能直接充当本轮 validation 结果。

## 2. 数据和预处理

| 项目 | 三天版设置 |
|---|---|
| 数据集 | `roneneldan/TinyStories`，revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| train split | 2,119,719 条故事 |
| validation split | 21,990 条故事 |
| tokenizer | `openai-community/gpt2`，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| 词表 | 50,257；EOS=50,256 |
| 文档处理 | 每篇末尾追加 EOS，按固定 parquet 顺序串接 |
| packing | 串接后切成无 padding 的 512-token block |
| 跨文档 attention | 允许通过 EOS 发生；所有模型完全一致 |
| loss | 对 packed block 中所有目标 token 计算 causal loss |
| train token 上限 | 10,000,000；不是把整个 train split 都训练一遍 |
| validation | 训练中用固定 262,144-token probe 监控；最终对最后 checkpoint 和 probe 最佳 checkpoint 各完整扫描一次 validation stream，当前约 4,765,918 tokens |
| test | TinyStories 没有官方 test split，本轮不声称 test 结果 |

跨故事 packing 只用于提高三天内的训练利用率。它不能证明故事内部存在长距离依赖；真正的跨话题记忆仍需后续独立合成任务。

## 3. 公共 TinyLM 设置

所有五个训练型模型共享以下主干：

| 参数 | 值 |
|---|---:|
| layers | 6 |
| hidden size | 384 |
| heads | 8 |
| head dim | 48 |
| FFN | 1,536 |
| normalization | Pre-LN |
| activation | GELU |
| position | RoPE，最大位置 32,768 |
| dropout | 0.1 |
| vocab | 50,257 |
| causal | 是 |
| input/output embedding | tied |
| Linear bias | 关闭 |
| LayerNorm | affine weight+bias 保留 |
| 公共参数量 | **29,925,504（约 29.93M）** |

公共 embedding、MLP、LayerNorm 和输出头使用匹配初始化；各方法特有的投影、随机特征、hash 或 memory 参数使用记录在案的确定性初始化。主比较采用“相同公共主干”，不强行调整 FFN 来掩盖方法特有参数；每个 run 必须报告总参数量和方法特有参数量。若关闭 LayerNorm bias，公共参数量为 29,920,512，必须在 resolved config 中明确记录，不能与标准设置混用。

## 4. 训练设置

| 项目 | 设置 |
|---|---|
| optimizer | AdamW |
| learning rate | `3e-4` |
| betas | `(0.9, 0.95)` |
| epsilon | `1e-8` |
| weight decay | `0.1` |
| warmup | 总步数的 3% |
| schedule | cosine，最低学习率 `3e-5` |
| gradient clipping | 1.0 |
| dtype | BF16 |
| train context | 512 |
| effective batch | 8,192 tokens |
| 推荐 micro-batch | 4 sequences × 512，gradient accumulation 4 |
| 低显存 fallback | 2 sequences × 512，gradient accumulation 8 |
| pilot | 每个候选配置 1,048,576 tokens（约 128 optimizer steps） |
| main | 每个冻结配置 10,000,000 tokens |
| 必跑 seed | 17 |
| 可选 seed | 29；43 留到正式实验 |
| early stopping | 不启用，确保 token budget 相同 |

10M tokens 对应约 1,221 个 optimizer steps（按 8,192 effective batch tokens 计算）。这足以判断 train/validation 曲线是否下降，但不能称为 TinyStories 完整收敛训练。总训练量是五个训练型框架各 10M tokens，加一个共享 Full-Attention reference 10M tokens；Keyformer 本身不新增训练量。

## 5. 六个框架的主配置

每个框架先用候选值做短 pilot，再按“先正确性、再 validation NLL、再吞吐、最后显存”的顺序冻结主配置。pilot 结束后不得根据 main validation 结果反向挑参数。

| 框架 | 候选设置 | 三天版主设置 | 备注 |
|---|---|---|---|
| Longformer | left window 128/256 | 128（总窗口 257），global=0 | causal sliding window |
| Performer | random features 64/128 | 128 | 固定 feature，不做 redraw |
| Linformer | rank 64/128 | 128 | projection length 固定为 512 |
| Reformer | bucket 32/64，hash=4 | bucket=64，hash=4 | causal，关闭 reversible 以降低接入风险 |
| Memformer | segment/slots=(128,32)/(128,64) | segment=128，slots=64 | segment 间不 detach；每个 packed example reset |
| Keyformer | cache ratio 50/75/100% | 全部报告；重点 50/75% | 原始推理期 cache policy，不训练 |

这些容量数字不是同一种资源单位，不能直接横比。结果必须同时给出实际 latency、显存、状态 bytes 和质量；若时间允许，再画每个方法自己的容量曲线。

可选的 context=1024 只用于原生支持或已经预先声明长度适配规则的方法。Linformer 的序列投影固定按 512 训练时，不得直接拿到 1024 使用；若实现插值或单独训练 1024 投影，必须单独标注，不能和原生长度外推混称。Keyformer 接入 RoPE TinyLM 时必须保留被选 K/V 的原始位置相位，并通过 100% cache 等价测试。

## 6. 验证与效率指标

### 必须结果

- train loss 曲线和 validation probe 曲线；
- validation token-weighted causal NLL；
- validation PPL（由完整 validation NLL 计算）；
- 最后 checkpoint 和 probe 最佳 checkpoint 的完整 validation 结果；
- training tokens/s、optimizer step time；
- batch=1 的 forward latency、peak allocated/reserved memory；
- 方法特有资源：Longformer 实际窗口、Performer features、Linformer rank、Reformer buckets、Memformer state bytes、Keyformer KV bytes；
- OOM、NaN、mask 错误和未完成运行必须保留原始记录。

### 正确性闸门

- causal future-invariance；
- shape/dtype 和有限梯度；
- padding/bucket mask；
- 适用时 full-window 与 dense reference 等价；
- Memformer state 跨 segment 生效且新 example reset；
- Keyformer cache ratio=100% 与 FullKV 的 token/PPL 等价。

长程 copy/passkey 或 `A → 长 B → 回到 A` 任务不是本轮六模型 validation 主表的必要条件；若有余力，只做小规模机制 smoke，并单独标记，不与 TinyStories PPL 混合。

## 7. 三人分工

| 人员 | 模型 | 共享主题 | 额外交付 |
|---|---|---|---|
| A | Linformer + Performer | 近似计算与 rank/features | 两模型完整 train/validation、fidelity/近似误差、质量和效率 |
| B | Longformer + Reformer | 稀疏连接与长度开销 | 两模型 causal mask、bucket/window 正确性、质量和效率 |
| C | Memformer + Keyformer | 有限历史状态压缩 | Memformer 跨 segment 验证；Keyformer FullKV 等价、cache sweep；合并 schema |

每个人对两个模型承担完整闭环，不能只负责某个指标。C 的“合并 schema”是协调职责，不改变其两个模型的实验责任。

## 8. 三天时间表

以下估计采用三张独立的 RTX 4090（每人一张、三个 worker 并行）；实际时间以第一个 100-step calibration 为准。若三张卡位于同一主机，给每个 worker 分配不同的 `CUDA_VISIBLE_DEVICES`；若位于三台主机，每个 worker 都使用本机的 `cuda:0`。数据/tokenizer cache 可只读共享，但 checkpoint、日志和结果目录必须按人员/模型隔离。

| 时间 | 全体任务 | 个人任务 |
|---|---|---|
| 第 1 天 0–2h | 安装依赖、检查 manifest、生成/缓存 train 10M 与 validation token stream | 各自复制 frozen baseline，确认设备和 dtype |
| 第 1 天 2–6h | 六个 adapter 完成 forward/backward、causal 和 reset smoke | A/B/C 各自处理两个框架的接口问题 |
| 第 1 天 6–10h | 运行 pilot 候选（每候选约 1M tokens） | 冻结各模型主配置，保存选择依据 |
| 第 2 天 0–8h | 启动 main 10M-token 训练；完成 Full-Attention reference 后启动 Keyformer shared-checkpoint eval | A/B 在各自 GPU 上跑两个模型；C 在同一 GPU 上先跑 reference，再跑 Memformer/Keyformer |
| 第 2 天 8–12h | 检查训练/validation 曲线，修复可重复性问题 | 必须保留失败 run，不静默替换配置 |
| 第 3 天 0–6h | 对最后/最佳 probe checkpoint 完成 full validation、latency、memory 和 OOM 矩阵 | 补跑 seed29 或关键长度（按剩余时间） |
| 第 3 天 6–10h | 自动生成表格、曲线、方法卡片和局限说明 | 每人提交两个模型的 raw JSON、README、结论 |

### 预计计算时间

在推荐硬件和实现没有 Python 瓶颈的情况下：

- 数据 tokenization/cache：20–90 分钟；首次处理完整 validation 可能接近 30 分钟；
- 每个训练型模型候选 pilot：约 15–60 分钟，未融合稀疏实现可能 1–2 小时；
- 每个训练型模型 10M-token main：约 30–120 分钟，Longformer/Reformer 的未融合实现可能达到 2–4 小时；共享 Full-Attention reference 也需约 30–120 分钟；
- 每个模型完整 validation：约 10–40 分钟，取决于 batch 和 backend；
- Keyformer cache ratio sweep：约 20–60 分钟，不含 backbone 训练；
- 每人两个模型的纯计算时间：通常 2–6 小时，复杂稀疏实现可能 6–10 小时；
- 三天总墙钟时间：三张独立 RTX 4090 并行时约 12–24 小时（含调试和整理）最稳妥；只有一张共享 GPU 时约 24–48 小时，必须启用 fallback。

若单个 run 在首轮 calibration 后预计超过 6 小时，按预先规定的 fallback 将 main token budget 降为 5M，并在结果中写明 `token_budget_deviation`；不得临时改变 context、模型宽度或删除失败记录。

## 9. 结果解释边界

本轮可以支持：

- 六种框架在相同 TinyLM/TinyStories 训练条件下的初步 validation 质量比较；
- 不同设计的实测速度、显存、状态/缓存开销和适用长度差异；
- 参数容量变化的初步趋势；
- 哪些实现已经达到可运行、可验证的程度。

本轮不能支持：

- 完整 TinyStories 收敛或论文级排行榜；
- 三 seed 统计显著性；
- Keyformer 作为独立训练 backbone 的结论；
- 仅凭 packed TinyStories 证明跨文档长期记忆；
- 你的新模型已经优于固定 memory 的最终结论。

新模型的多话题情景记忆、原子事实 Key–Value、soft gate、hard top-k 和 payload 压缩属于下一阶段独立实验。
