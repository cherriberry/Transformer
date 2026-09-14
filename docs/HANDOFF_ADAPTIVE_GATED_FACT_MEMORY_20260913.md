# 仓库交接文档：Adaptive Gated Fact Memory 实验与后续优化

更新时间：2026-09-13（UTC）
工作区：/root/BDMI/26summerBDMI_transformer
当前分支：role_B，HEAD：a2540bf（feat: refresh pdf）
大体积数据、checkpoint 和聚合结果：/root/autodl-tmp/26summerBDMI_transformer/

本文写给下一位接手本仓库的 AI/实验执行者。目标是让接手者在不重新猜测实验
含义、不覆盖历史结果的情况下，继续运行和优化 adaptive_gated_fact_memory。
本文是当前状态的执行入口，不替代各个专项报告；数值以数据盘中的
summary.json 和最新专项报告为准。

## 0. 先看结论

当前项目有两条相互关联、但不能混为一谈的研究线：

1. 用 TinyStories/TinyLM 比较 Longformer、Memformer、Linformer、Performer、
   Reformer、Full Attention 和 Keyformer 等高效 Transformer 方法；
2. 在有界 sink+sliding local KV 之上加入自适应事实记忆，研究内容级写入、保留、
   更新、读取和融合，并进一步测试 Value-token alignment 与 MoE。

adaptive_gated_fact_memory 当前最重要的结果是：

- 六层 decoder-only TinyLM，hidden size 384，8 heads，局部 KV 为 4 个 sink 加
  124 个 recent token；
- seed=17 的 selective-memory stress 中，完整 Value-token Gated + MoE 在
  288/288 验证样本上达到 100% EM、100% target survival、100% Recall@1、
  100% Recall@4 和 MRR=1；
- 最难条件为 8 条临时事实加 16 个 delay segments（约 2048 个 TinyStories
  干扰 token），仍只激活平均 1 个用户事实槽位；
- 旧 Gated 已能保存和检索事实，但 Dense EM 为 52.43%，MoE EM 为 32.99%。
  Value-token path 将两者都提升到 100%，说明主要瓶颈在“检索后如何生成具体
  value token”，而不是事实是否写入或 reader 是否命中；
- 这仍然只是单 seed、封闭颜色词表、受控提示模板的 screening，不能宣称已经
  证明开放域长期记忆能力；当前槽位张量仍预分配，logical active bytes 也不等于
  物理显存压缩。

当前建议的主优化方向是：先补齐多 seed、独立组件消融、无显式重要性提示的任务、
多事实/冲突/数字日期实体测试，再做 packed memory、fused attention/MoE 和真正的
端到端延迟测量。

## 1. 仓库地图

| 路径 | 内容 | 接手时的用途 |
|---|---|---|
| adaptive_gated_fact_memory/ | 本项目的自适应事实记忆原型、训练脚本、测试和报告 | 继续优化的主目录 |
| adaptive_gated_fact_memory/src/adaptive_fact_memory/ | config.py、model.py、memory.py、state.py、attention.py、baselines.py | 模型和状态实现 |
| adaptive_gated_fact_memory/scripts/ | TinyStories 预训练、结构化记忆、压力测试、诊断和数据生成 | 训练/评估入口 |
| adaptive_gated_fact_memory/tests/ | 原型、MoE、状态、因果性和 source-policy 测试 | 修改后第一道回归检查 |
| adaptive_gated_fact_memory/references/ | Performer、Gemma、StreamingLLM、TriAttention、Memformer 等只读源码快照 | 机制和来源审计，不直接作为本模型代码 |
| experiments/tinystories_tinylm_v1/ | 四种训练型 backbone 的统一 Dense TinyStories 实验 | A+B 三 seed 基线 |
| experiments/tinystories_tinylm_moe_v1/ | 六种 backbone 的全层 MoE 统一实验 | 与 adaptive stress 不同的 MoE 线 |
| longformer/、memformer/、linformer/、performer/、reformer/ | 各方法的旧版脚本、README、图和 JSON 结果 | 基线实现及历史结果 |
| keyformer/、models/keyformer.py | Keyformer KV cache 选择/压缩 | 推理期 cache 基线 |
| common/、benchmarks/、根目录脚本 | WikiText-2、微基准和 Keyformer 工具 | 非 adaptive 辅助实验 |
| data/ | TinyStories 原始说明、tokenizer 和仓库内小缓存 | 数据协议与可复核 manifest |
| docs/ | 实验协议、结果报告、研究分析、交接文档 | 推荐阅读入口 |
| AAAI-temp-LaTeX/LaTeX/ | 中文论文草稿、AAAI 模板、PDF 和图 | 论文排版，不参与训练 |

根目录的 README.md 和 docs/README.md 是总索引。仓库中有部分历史文档仍保留
旧状态；不要因为旧文档写着“MoE 待补训”就重复或覆盖已经完成的 run。

## 2. 推荐阅读顺序和权威性

如果只继续 adaptive 实验，按下面顺序阅读：

1. 本文；
2. docs/ADAPTIVE_GATED_MEMORY_MOE_COMPARISON_S17_REPORT.md（当前四 Dense + 四 MoE 的最新总表）；
3. docs/ADAPTIVE_GATED_MEMORY_STRESS_S17_RESULT_REPORT.md（Dense stress 的详细分析）；
4. ADAPTIVE_GATED_MEMORY_VALUE_TOKEN_ALIGNMENT_S17_EXPERIMENT_REPORT.md（V3 词级对齐）；
5. ADAPTIVE_GATED_MEMORY_VALUE_TO_TOKEN_DIAGNOSTIC_REPORT.md 和
   ADAPTIVE_GATED_MEMORY_BOTTLENECK_DIAGNOSTIC_PAPER_REPORT.md（旧 Gated 的根因诊断）；
6. docs/ADAPTIVE_FACT_MEMORY_ARCHITECTURE_EVOLUTION_AND_EXPERIMENT_SETUP_REPORT.md（架构演进、参数和结果解释）；
7. adaptive_gated_fact_memory/DESIGN.md、README.md、PROTOTYPE_STATUS.md（代码边界）。

需要注意的过时或历史材料：

- docs/HANDOFF_NEXT_MODEL_20260910.md 记录的是 MoE retry 前状态，其中“旧 Gated + MoE
  失败、Value-token + MoE 未完成”已经过时；最新 retry1 已完成；
- adaptive_gated_fact_memory/PROTOTYPE_STATUS.md 是 2026-09-09 的阶段盘点，测试数和
  “尚未完成”列表不是当前最终状态；
- ADAPTIVE_GATED_MEMORY_CURRENT_RESULTS_PAPER_REPORT.md 有摘要和早期主结果，仍有参考
  价值，但 MoE 和统一 state schema 应以最新 MoE 报告为准；
- 当前工作区已有未跟踪文件 docs/ADAPTIVE_GATED_MEMORY_FOUR_GATES_FORMULAS_ZH.md，不要删除或重置。

## 3. 研究目标和证据边界

### 3.1 总体目标

项目试图解决标准 Transformer 在长上下文中的两个问题：局部 KV cache 随上下文增长，
以及把全部历史压缩为固定状态后难以精确找回事实。目标架构采用两个时间尺度：

~~~text
短期：每层固定大小的 4 sink + 124 recent token-level KV
长期：固定上限、可寻址、可更新的事实 memory slots
~~~

长期记忆不是全文 token 的无损备份。只有被事实提取器提出、通过写入门并在 commit
时接受的内容，才会离开局部 KV 后继续存在。被门控拒绝或淘汰的信息无法恢复。

### 3.2 adaptive 专项问题

当前 selective stress 主要检验：

1. 写入门能否保留耐久事实、拒绝临时事实；
2. 固定容量超载时，重要事实是否仍能存活；
3. reader 是否能在长干扰后找到正确槽位；
4. fusion 和输出路径能否把语义记忆转成具体答案 token；
5. 在不改变 memory 路径的情况下，MoE FFN 是否带来额外条件容量；
6. 选择性槽位使用是否形成较好的质量—状态开销折中。

### 3.3 结论强度

当前可写：

> 在 seed-17 的受控 selective-memory stress 中，门控能够选择性拒绝临时事实，
> Value-token alignment 能够修复旧 Gated 的词级答案恢复瓶颈，并与 Top-2 MoE 稳定组合。

当前不可写：

- 已证明开放域长期对话中普遍有效；
- 已证明物理显存随 active slots 降低；
- 已证明门控能在没有显式重要性线索或 span 标签时自主理解任意事实重要性；
- 已证明 MoE 在严格等总参数或多 seed 下优于 Dense；
- 已证明对数字、日期、罕见实体和任意多 token 字符串均能精确复制。

## 4. Adaptive Gated Fact Memory 的实现

### 4.1 TinyLM 主干

默认主干配置：

| 参数 | 默认值 |
|---|---:|
| decoder layers | 6 |
| hidden size | 384 |
| attention heads | 8（head dim=48） |
| Dense FFN hidden | 1536 |
| vocab | GPT-2 BPE，50,257 |
| position | RoPE，最大位置 32,768 |
| normalization | Pre-LN |
| dropout | 0.1 |
| input/output embedding | tied |

普通层为：

~~~text
LayerNorm → causal local attention → residual
          → LayerNorm → Dense FFN 或 MoE → residual
~~~

第 3、6 层（代码下标 2、5）在 local attention residual 后增加：

~~~text
Memory LayerNorm → memory cross-attention → fusion gate
                 → memory residual → FFN/MoE
~~~

### 4.2 局部短期 KV

每一层都有独立的 LocalKVCache，字段包括 sink_k/sink_v、recent_k/recent_v、
位置和有效性掩码。attention budget=128：

~~~text
4 个 sink KV + 124 个 recent non-sink KV = 128 个唯一 key
~~~

sink 是对话最初位置的普通 K/V，永久保留以稳定滑动窗口；它不是长期事实槽位，
也不是拥有全局双向访问权的 Longformer global token。recent 部分随新 token 到来而
滑动淘汰。局部 cache 每层即时更新，不经过 memory gate，也不等待 commit。

### 4.3 长期事实状态

长期事实状态在六层主干之外，由 ConversationState.memory 持有，通过第 3、6 层的
SharedMemoryFusion 注入主干。一个用户槽位包含：

~~~text
keys                事实级检索 key，hidden-size=384
values              关系/事实语义 value，hidden-size=384
lexical_values      value 子跨度的词级连续表示（V3 才启用）
payload_ids/mask    完整事实的原始 token ID 及有效掩码
value_payload_ids/mask  value 子跨度 token ID 及掩码（V3）
active              连续活动权重
age/last_access/access_count
confidence/conflict/source_role
~~~

因此槽位不是局部 attention 的原始 K/Q/V，也不是完整 QKV；它是“事实级连续向量 +
可审计 token payload + 生命周期元数据”。assistant_keys/assistant_values 另有
4 个固定 assistant-state 槽位，但 stress 协议通过把 assistant write bias 设为 -10，
实际上关闭了 assistant-state 写入；不要将其误解为当前主结果的重要来源。

### 4.4 一轮的因果生命周期

一轮定义为 user prompt 加完整 assistant response。逻辑顺序：

~~~text
读取上一轮提交的 M_r
  → 对完整 user/system prompt pooling，计算 reader query
  → semantic Top-k，缓存本轮槽位索引
  → 生成回答；assistant continuation 复用同一组索引
  → 新 user candidates 暂存 pending buffer
  → 当前 logits 计算结束并收到 commit
  → retention + write + merge/update + eviction
  → 得到下一轮 M_{r+1}
~~~

memory_visibility 从 prompt 的最后一个 user/system token 开始，因此旧记忆在回答
边界前不会影响 prompt 内部的早期 logits。新写入只在 commit 后进入下一轮状态，
避免回答使用自己刚写入的未来信息。代码允许在同一个 forward_round 调用末尾完成
commit，但先计算 logits、再更新 state 的因果顺序不变。pending 和 round_open 是
为了支持用户预填充、assistant 分块流式生成与一次性整轮训练的一致语义。

### 4.5 Reader、Top-k 和融合

当前 prompt embedding pooling 为临时查询向量，不是长期记忆状态：

\[
u_r=\operatorname{Pool}(E(P_r)),\qquad q_r=\operatorname{Normalize}(W_q u_r).
\]

reader 对 active slots 计算：

\[
s_{r,i}=q_r^\top k_i,\qquad \mathcal I_r=\operatorname{TopK}_i(s_{r,i}).
\]

Top-k 不会把 k 个槽位永久合成一个槽位。第 3、6 层对这 k 个槽位重新做 token-to-memory
cross-attention：

\[
c_t=\sum_{i\in\mathcal I_r}\alpha_{t,i}\,\widetilde v_i,
\]

其中 c_t 是当前 token 的临时 384 维 memory context，槽位仍以独立形式存在。
随后由 fusion gate 决定注入强度：

\[
g_t=\sigma(W_g[h_t;c_t]+b_g),\qquad h'_t=h_t+g_tW_oc_t.
\]

Reader 决定候选集合，alpha 决定集合内部相对贡献，fusion gate 决定长期记忆
整体影响当前 token 的强度。固定策略的 fusion gate 可设为全开，诊断脚本也支持固定
扫描 0、0.25、0.5、0.75、1.0。

### 4.6 四类记忆门

MoE router 不属于记忆四门。当前四门是：

~~~text
write gate      新候选是否写入长期槽
retention gate  旧槽在下一次 commit 后是否继续保留
update gate     合并新旧事实时采用多大新信息
fusion gate     已读记忆对当前 decoder token 影响多少
~~~

代表公式：

\[
g_i^{write}=\sigma(W_{write}\bar h_i+b_{write}),
\]
\[
g_r^{ret}=\sigma(W_{ret}[k_r;v_r;m_r]+b_{ret}),
\]
\[
g_{i,r}^{update}=\sigma(W_{update}[v_i^{cand};v_r^{old};s_{i,r}]+b_{update}),
\]
\[
g_t^{fusion}=\sigma(W_{fusion}[h_t;c_t]+b_{fusion}).
\]

训练时使用 straight-through soft/hard 路径；评估默认阈值为 0.5。stress 训练时
write/retention threshold 为 0.1，以便保留梯度；评估时为 0.5。槽位目标选择会先
尝试高相似度 merge，否则使用空槽或按策略淘汰低价值槽。

### 4.7 Value-token alignment

旧 Gated 将完整事实 span 平均池化后形成 semantic value，并将完整 payload embedding
平均后用于 fusion；这保留“事实讲了什么”，却没有明确标出需要回答的 object/value。

V3 在不改变基本 write/retention/read 结构的前提下增加 value-span start/length、
lexical value、value payload 和 memory-to-token auxiliary loss，使用
--memory-value-mode semantic_lexical 与 --value-token-alignment。

辅助 head 为 lexical value 加 answer-position embedding，经 projection、LayerNorm 和
tied token embedding 投影输出 token logits。它只参加训练辅助损失；正式回答仍由普通
自回归 LM head 生成，不是 pointer/copy head。

### 4.8 Dense 与 adaptive stress MoE

adaptive stress 中的 MoE 只替换每个 block 的 FFN：4 experts、Top-2、dropless。
每个 token 由 router 选两个 expert，按归一化权重相加；padding 不参与路由统计。
普通 PyTorch 实现采用 token indexing、逐 expert forward 和 index_add_，没有
fused grouped-GEMM/MegaBlocks。

代码保留 ffn_in/ffn_out 是为了兼容 Dense checkpoint，并将 Dense 权重复制到每个
expert 做初始化；MoE 初始化后这两个 Dense 参数被冻结，不参与 active computation。
adaptive stress 的 MoE（Top-2）与 experiments/tinystories_tinylm_moe_v1 六模型线
的 MoE（Top-1）不同，绝对不能混写。

## 5. Adaptive 实验协议

### 5.1 数据

TinyStories 固定版本：

~~~text
dataset revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
tokenizer: openai-community/gpt2
tokenizer revision: 607a30d783dfa663caf39e06633721c8d4cfcd7e
~~~

stress filler 使用数据盘：

~~~text
/root/autodl-tmp/26summerBDMI_transformer/data/cache/
  tinystories_tinylm_v1/train_100000257.int32.bin
~~~

每个样本：

~~~text
durable user fact
→ 0/若干 temporary user facts
→ TinyStories system filler
→ user query
→ assistant answer
~~~

耐久提示训练用 Remember/Keep/Save，验证用未见的 Store；临时提示训练用
Temporary/Skip/Ignore，验证用 Discard。名字和颜色来自固定 16 个值的封闭词表，
关系主要是 NAME likes VALUE。重要性标签来自提示词和结构化监督，不是完全无提示
的开放域重要性推断。

### 5.2 两套 adaptive memory 配置

不要混淆默认配置和正式 stress 配置：

| 参数 | FactMemoryConfig 默认/Stage 1 | stress v2/v3 正式设置 |
|---|---:|---:|
| user memory slots | 64 | 8 |
| read Top-k | 8 | 4 |
| max write candidates | 4 | 1（load_stress_model 内硬编码） |
| merge threshold | 0.78 | 0.999 |
| payload token budget | 12 | 12 |
| local KV | 4 sink + 124 recent | 相同 |
| fusion layers | 2、5（0-based） | 相同 |
| write/ret threshold（train/eval） | 依调用设置 | 0.1 / 0.5 |

Stage 2 structured curriculum 使用 64 slots、Top-8；正式 selective stress 为了制造
明确容量压力改用 8 slots、Top-4。论文或新报告必须写清楚使用哪一套。

### 5.3 训练超参数

| 项目 | stress v2/v3 设置 |
|---|---:|
| seed | 17 |
| steps | 2,000 |
| physical batch | 2 |
| gradient accumulation | 2（每 step 4 examples） |
| examples | 8,000 |
| backbone LR | 3e-5 |
| memory/new-head LR | 3e-4 |
| optimizer | AdamW，betas=(0.9,0.95)，eps=1e-8，weight decay=0.1 |
| gradient clip | 1.0 |
| precision | RTX 4090 BF16 autocast，TF32 关闭 |
| train delays | 1、2、4 segments |
| eval delays | 1、4、16 segments |
| train noise | 0、1、2、4 |
| eval noise | 0、2、4、8 |
| eval examples/cell | 24，共 288 |
| checkpoint | 每 500 step，另存 final |

损失权重（非 V3 旧路径）：LM=1、fact start=0.75、fact length=0.25、write=0.75、
read=1、key alignment=0.50、retention=0.10、budget=0.01。V3 额外为 value start=0.50、
value length=0.25、candidate memory-token=0.50、retrieved memory-token=0.50。adaptive
stress MoE 另加 0.01 * moe_load_balance。

## 6. Adaptive 实验分组和结果

### 6.1 Stage 1：TinyStories backbone 预训练

入口：adaptive_gated_fact_memory/scripts/train_tinystories.py。本阶段每个 packed
block 使用新 state，关闭 episodic memory，只训练局部语言模型 backbone；不能当作
事实记忆结果。

run：

~~~text
/root/autodl-tmp/26summerBDMI_transformer/runs/
  adaptive_gated_fact_memory_tinystories/tinystories_backbone_s17_10m_20260908/
~~~

结果：10M training tokens，full validation NLL=2.889852、PPL=17.9906、训练 430.9 s、
峰值 allocated 4.66 GB、31,854,750 参数。该 checkpoint 是所有当前 adaptive stress
run 的父模型，SHA-256：

~~~text
d995248587a5e535af59d850f36a5bc01925c54bd9dff213f0f18bfb26f2b78f
~~~

### 6.2 Stage 2：早期结构化事实记忆课程

入口：adaptive_gated_fact_memory/scripts/train_structured_memory.py。任务为：

~~~text
early user fact → long TinyStories/system distractor → query
~~~

run：

~~~text
/root/autodl-tmp/26summerBDMI_transformer/runs/
  adaptive_gated_fact_memory_structured/structured_memory_s17_20260909/
~~~

配置为 2,000 steps、64 slots、Top-8、延迟训练 1/2/4/8、验证 1/2/4/8/16、120 条
验证样本。结果为 EM=86.67%（104/120），strict target-slot read hit=100%，平均 active
slots=1，state bytes=2,598,885。16-segment 延迟下 EM=87.50%、read hit=100%。

限制：事实 span、写入和读取都有结构化监督；只有单事实；assistant-state write 被
抑制；不能据此推断无标签开放域抽取或多事实冲突更新已经解决。

### 6.3 Stage 3：selective stress v2 Dense 三条件

三条件共用同一 seed-17 父模型和 8-slot 压力协议：

| 条件 | 行为 |
|---|---|
| SWA-only（memory_policy=none） | 只有局部 sink+recent，没有长期事实槽位 |
| Fixed-LRU（fixed_lru） | 所有有效候选都写入，满容量后按 LRU 淘汰 |
| old Gated（gated） | 学习式 write/retention/update/读写融合门 |

Dense run：

~~~text
selective_stress_v2_swa_only_s17_20260909
selective_stress_v2_fixed_lru_s17_20260909
selective_stress_v2_gated_s17_20260909
~~~

总体结果：

| 条件 | EM | answer NLL | target survival | Recall@1 | Recall@4 | MRR | active slots | noise accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWA-only | 5.21% | 1.38947 | N/A | N/A | N/A | N/A | 0.00 | N/A |
| Fixed-LRU | 74.65% | 0.50323 | 92.01% | 90.28% | 92.01% | 0.9103 | 4.25 | 75.00% |
| old Gated | 52.43% | 0.63089 | 100.00% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% |

按容量压力聚合：noise=8 时 Fixed-LRU 目标 survival 降至 68.06%、Recall@1 降至
61.11%，而 old Gated 仍为 100%；Gated 拒绝全部临时事实。按 delay 聚合时，Gated
在 delay=16 仍保持 100% survival/Recall，但 EM 仍只有约 51%。

关键解释：门控的“选择性保存”优于固定写入，但这项旧模型的最终回答准确率不如
Fixed-LRU，说明“存住/找对”与“生成正确 value”是两个独立问题。

### 6.4 Stage 4：Value-token Gated V3 Dense

run：

~~~text
selective_stress_v3_gated_value_token_s17_20260910
~~~

V3 在 old Gated 上增加 value-span start/length、lexical value、value payload 和
memory-to-token auxiliary loss，使用 --memory-value-mode semantic_lexical
与 --value-token-alignment。

结果：

| 指标 | V3 Dense |
|---|---:|
| EM | 100.00%（288/288） |
| target survival / Recall@1 / Recall@4 / MRR | 100% / 100% / 100% / 1.0000 |
| value start/length/span exact | 100% / 100% / 100% |
| memory-token token/sequence exact | 100% / 100% |
| teacher-forced answer NLL | 0.00001946 |
| first-token mean rank / Top-5 | 1.0000 / 100% |
| first-token logit margin | +13.5763 |
| active slots / noise accept | 1.00 / 0% |
| 训练时间 / 吞吐 | 1380.1 s / 5.797 ex/s |
| peak allocated / reserved | 2.682 GB / 6.702 GB |
| 参数量 | 32,160,043 |

12 个 noise×delay cell 全部 100% EM，包括 noise=8、delay=16。相对 old Gated，参数
增加 0.96%，完整 state 增加约 0.61%，训练时间增加 4.31%。这强烈支持“旧 Gated
主要缺少词级 value 对齐”的假设，但不等于开放域 copy 能力已经被证明。

### 6.5 Stage 5：adaptive stress MoE 四条件

adaptive stress 的 MoE 配置为 4 experts、Top-2、dropless、load-balance weight=0.01。
四个正式 run：

~~~text
selective_stress_moe_swa_only_s17_20260910
selective_stress_moe_fixed_lru_s17_20260910
selective_stress_moe_gated_s17_20260910_retry1
selective_stress_moe_value_token_gated_s17_20260910_retry1
~~~

统一结果：

| 条件 | EM | answer NLL | survival | Recall@1 | Recall@4 | MRR | active slots | noise accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWA-only + MoE | 5.21% | 1.38736 | N/A | N/A | N/A | N/A | 0.00 | N/A |
| Fixed-LRU + MoE | 65.28% | 0.57731 | 93.06% | 93.06% | 93.06% | 0.9306 | 4.25 | 75.00% |
| old Gated + MoE | 32.99% | 0.85217 | 100.00% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% |
| Value-token Gated + MoE | 100.00% | 0.00002212 | 100.00% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% |

Dense→MoE 资源变化：

| 条件 | 总参数 Dense→MoE | active/token Dense→MoE | 训练时间 Dense→MoE | 吞吐 Dense→MoE | peak allocated Dense→MoE |
|---|---:|---:|---:|---:|---:|
| SWA-only | 31.85M→60.18M | 31.85M→38.94M | 880.1→1479.8 s | 9.09→5.41 ex/s | 2.871→3.433 GB |
| Fixed-LRU | 31.85M→60.18M | 31.85M→38.94M | 1200.6→1804.0 s | 6.66→4.43 ex/s | 2.666→3.190 GB |
| old Gated | 31.85M→60.18M | 31.85M→38.94M | 1323.0→1905.0 s | 6.05→4.20 ex/s | 2.673→3.201 GB |
| Value-token Gated | 32.16M→60.48M | 32.16M→39.25M | 1380.1→1927.1 s | 5.80→4.15 ex/s | 2.682→3.205 GB |

MoE 让总容量增加约 89%，但当前非融合 dispatch 使训练明显变慢、峰值 allocated 增加。
旧 Gated + MoE 的 survival/Recall 仍满分但 EM 更低，证明 MoE 不是词级答案恢复的替代
方案；Value-token + MoE 保持 100% EM。

最终验证路由统计（token fraction / dispatch fraction）：

| 条件 | token fraction E0/E1/E2/E3 | dispatch fraction E0/E1/E2/E3 | entropy | load-balance |
|---|---|---|---:|---:|
| SWA-only + MoE | 0.536/0.434/0.574/0.456 | 0.251/0.242/0.273/0.233 | 1.230 | 2.240 |
| Fixed-LRU + MoE | 0.532/0.527/0.533/0.408 | 0.297/0.238/0.262/0.202 | 1.282 | 2.275 |
| old Gated + MoE | 0.545/0.457/0.569/0.429 | 0.275/0.232/0.270/0.223 | 1.312 | 2.210 |
| Value-token Gated + MoE | 0.471/0.508/0.511/0.511 | 0.239/0.259/0.244/0.258 | 1.355 | 2.111 |

Top-2 的均衡参考是 token fraction≈0.5、dispatch fraction≈0.25、load-balance≈2。
当前没有单 expert collapse，但全局均衡不代表每层或每个小 batch 都均匀，也不等于
experts 已形成可解释的语义分工。

## 7. 旧 Gated 瓶颈诊断

旧 Gated 的核心证据链：

~~~text
target survival = 100%
Recall@1 = 100%
MRR = 1.0
但 EM = 52.43%
~~~

进一步诊断：正确答案首 token 平均词表排名 2.04，进入 Top-5 的比例 97.57%；常见
错误是另一个颜色词。因此问题主要位于：

~~~text
正确槽位 → memory fusion → decoder hidden → tied LM head 的词表排序
~~~

已完成的 inference-only/诊断实验：

| 实验 | 结果 | 解释 |
|---|---|---|
| 固定 fusion gate 0/0.25/0.5/0.75/1 | EM=5.21/14.58/57.99/55.21/52.43% | gate 太低会失去记忆，0.5 略优，但不是根因 |
| checkpoint 500/1000/1500/2000 | EM=11.46/19.79/42.36/52.43% | EM 持续上升，无明显中途过拟合 |
| 训练模板 vs 验证改写模板 | 78.13% vs 52.43% | 模板迁移显著影响答案生成，Recall 仍满分 |
| 统一答案前导空格重训 | EM=20.14% | tokenization 是因素，但简单加空格不是修复 |
| fusion payload_only / value_only | 21.53% / 27.78% | 两条表示都含信息，单独删除任一条都会退化 |
| representation mapping | payload 与答案 cosine≈0.520；compressed value≈-0.036 | 平均池化保留语义但弱化词级身份 |

因此下一阶段不要只继续调 write/retention gate；优先检查 answer utilization、value
tokenization、fusion 路径和跨轮梯度。

## 8. 非 adaptive 基线实验概览

这些结果用于论文背景和横向比较，但不能与 adaptive fact-memory EM 直接混成同一
统计任务。它们使用 TinyStories language modeling 协议，指标主要是 validation NLL/PPL。

### 8.1 A+B unified Dense（三 seed）

统一 10M prediction tokens、context=512、seeds=17/29/43：

| 方法 | Val NLL 均值±std | PPL 均值±std | 训练 tok/s | peak allocated | 参数量 |
|---|---:|---:|---:|---:|---:|
| Memformer | 2.849148±0.004651 | 17.273176±0.080427 | 14,669.5±300.5 | 2.633 GiB | 33,466,752 |
| Longformer | 2.885517±0.004599 | 17.912946±0.082476 | 13,498.8±268.5 | 6.799 GiB | 29,925,504 |
| Linformer | 2.933816±0.006033 | 18.799463±0.113420 | 27,857.8±106.0 | 2.580 GiB | 29,925,504 |
| Performer | 3.450477±0.008266 | 31.516144±0.260635 | 27,079.0±541.2 | 2.704 GiB | 29,925,504 |

当前 100M 扩展显示训练预算会改变结论：Longformer PPL=6.0967，Memformer PPL=6.4622，
但 Memformer 训练峰值显存更低。两者的 adapter、参数量和训练预算不能简单合并排名。

### 8.2 六模型全层 MoE（另一条 Top-1 协议）

experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml 使用全六层 4-expert、Top-1、
dropless、10M predictions、seed=17：

| backbone | Val NLL | PPL | 训练 tok/s | peak GiB |
|---|---:|---:|---:|---:|
| Memformer | 2.758941 | 15.7831 | 12,201.3 | 2.960 |
| Longformer | 2.869545 | 17.6290 | 12,265.3 | 7.234 |
| Full Attention | 2.907851 | 18.3174 | 26,503.8 | 2.911 |
| Linformer | 2.925335 | 18.6405 | 20,340.2 | 2.913 |
| Reformer | 2.980978 | 19.7071 | 26,821.2 | 2.900 |
| Performer | 3.457819 | 31.7477 | 20,358.5 | 3.120 |

这条线的 MoE Top-1 和 adaptive stress 的 Top-2 不同；不要用它解释 adaptive memory
门控的因果效果。

### 8.3 Keyformer 和微基准

- Keyformer 是推理期 KV-cache policy，不是独立训练 backbone。Dense/MoE FullKV 对照
  均显示 cache ratio=0.5 可将物理 KV bytes 减半，但当前 Python selection/gather
  使 decode 吞吐下降约 16%–17%；ratio=1 与 FullKV 数值等价；
- 早期 Standard/Memformer/Performer/Reformer 随机输入微基准只测 attention 模块在
  L=128/256/512 的前向时间和 CUDA peak memory，短序列上 Standard 最快；不能当成
  训练质量排名；
- Sliding-window beats linear attention 没找到可确认的作者公开代码。adaptive 的
  SWA baseline 使用冻结的 StreamingLLM StartRecentKVCache 规则适配，不应称为论文
  作者官方实现。

## 9. 代码入口、环境和可复现命令

### 9.1 环境变量和数据盘

stress 脚本默认读取：

~~~text
TINYSTORIES_TOKENIZER_DIR=/root/autodl-tmp/26summerBDMI_transformer/data/tokenizer/gpt2
TINYSTORIES_CACHE_DIR=/root/autodl-tmp/26summerBDMI_transformer/data/cache/tinystories_tinylm_v1
ADAPTIVE_TINYSTORIES_CHECKPOINT=<seed-17 backbone checkpoint>
ADAPTIVE_MEMORY_STRESS_RUNS_DIR=/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress
~~~

训练脚本会拒绝覆盖已有 summary.json 或 checkpoint.final.pt。每个新实验必须使用
新 --run-id，不要复用旧目录。

### 9.2 回归测试和 smoke

~~~bash
cd /root/BDMI/26summerBDMI_transformer
PYTHONPATH=adaptive_gated_fact_memory/src \
  python3 -m unittest discover \
  -s adaptive_gated_fact_memory/tests -p 'test_*.py' -v

PYTHONPATH=adaptive_gated_fact_memory/src \
  python3 adaptive_gated_fact_memory/scripts/smoke_test.py
~~~

当前测试实际结果为 23/23 passed。

### 9.3 复跑一个新的 old Gated + MoE stress

~~~bash
PYTHONUNBUFFERED=1 \
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 adaptive_gated_fact_memory/scripts/train_memory_stress.py \
  --device cuda:0 \
  --run-id <new_unique_run_id> \
  --runs-dir /root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress \
  --memory-policy gated \
  --seed 17 \
  --steps 2000 --batch-size 2 --gradient-accumulation-steps 2 \
  --memory-slots 8 --memory-read-top-k 4 \
  --moe-experts 4 --moe-top-k 2 \
  --moe-load-balance-weight 0.01
~~~

### 9.4 复跑 Value-token Gated + MoE stress

~~~bash
PYTHONUNBUFFERED=1 \
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 adaptive_gated_fact_memory/scripts/train_memory_stress.py \
  --device cuda:0 \
  --run-id <new_unique_run_id> \
  --runs-dir /root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress \
  --memory-policy gated \
  --seed 17 \
  --steps 2000 --batch-size 2 --gradient-accumulation-steps 2 \
  --memory-slots 8 --memory-read-top-k 4 \
  --memory-value-mode semantic_lexical --value-token-alignment \
  --moe-experts 4 --moe-top-k 2 \
  --moe-load-balance-weight 0.01
~~~

单张 RTX 4090 上不要并发两个训练进程。正式 run 应包含：
config.resolved.json、metrics.jsonl、checkpoint.step_*.pt、checkpoint.final.pt、
summary.json。失败目录也保留，用新 retry ID 重跑。

### 9.5 训练阶段入口

~~~bash
# Stage 1 backbone
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 adaptive_gated_fact_memory/scripts/train_tinystories.py \
  --run-id <new_backbone_run> --train-tokens 10000000 --full-validation

# Stage 2 structured memory
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 adaptive_gated_fact_memory/scripts/train_structured_memory.py \
  --run-id <new_structured_run>
~~~

具体参数可从 config.resolved.json 和相应脚本的 --help 核对；Stage 2 与 stress
不是同一个协议，不能只改 run name 就当作同一实验。

## 10. 结果文件和可追溯资产

### 10.1 adaptive 报告

- adaptive_gated_fact_memory/README.md
- adaptive_gated_fact_memory/DESIGN.md
- adaptive_gated_fact_memory/TINYSTORIES_STAGE1_REPORT.md
- adaptive_gated_fact_memory/STRUCTURED_MEMORY_STAGE2_S17_REPORT.md
- adaptive_gated_fact_memory/SWA_BASELINE_SOURCE_AUDIT.md
- docs/ADAPTIVE_GATED_MEMORY_STRESS_S17_RESULT_REPORT.md
- docs/ADAPTIVE_GATED_MEMORY_MOE_COMPARISON_S17_REPORT.md
- ADAPTIVE_GATED_MEMORY_CURRENT_RESULTS_PAPER_REPORT.md
- ADAPTIVE_GATED_MEMORY_VALUE_TOKEN_ALIGNMENT_S17_EXPERIMENT_REPORT.md
- ADAPTIVE_GATED_MEMORY_VALUE_TO_TOKEN_DIAGNOSTIC_REPORT.md
- ADAPTIVE_GATED_MEMORY_BOTTLENECK_DIAGNOSTIC_PAPER_REPORT.md

### 10.2 正式 run 目录

父 backbone：

~~~text
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/
  tinystories_backbone_s17_10m_20260908/
~~~

结构化 Stage 2：

~~~text
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_structured/
  structured_memory_s17_20260909/
~~~

selective stress 的正式目录：

~~~text
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/
  selective_stress_v2_swa_only_s17_20260909/
  selective_stress_v2_fixed_lru_s17_20260909/
  selective_stress_v2_gated_s17_20260909/
  selective_stress_v3_gated_value_token_s17_20260910/
  selective_stress_moe_swa_only_s17_20260910/
  selective_stress_moe_fixed_lru_s17_20260910/
  selective_stress_moe_gated_s17_20260910_retry1/
  selective_stress_moe_value_token_gated_s17_20260910_retry1/
~~~

同目录下还存在 smoke、debug、failed retry 和诊断 JSON；它们用于审计，不要误当成
正式主结果，也不要删除。历史失败的
selective_stress_moe_gated_s17_20260910/ 因 step-0 diagnostics 错误失败，retry1
才是正式结果。

### 10.3 代码快照

~~~text
code_snapshots/pre_value_token_alignment_20260910_utc/
code_snapshots/pre_moe_20260910_utc/
~~~

它们记录 Value-token 和 MoE 修改前的代码及 SHA-256。需要做结构性改动时先创建
新的快照，不要覆盖旧快照。

## 11. 已知限制和常见误读

1. **只有 seed=17。** 100% EM 是点估计，不能报告跨 seed 均值、标准差或显著性。
2. **任务封闭。** 16 个颜色值、固定 NAME likes VALUE 关系和强提示词可能让模型
   学到 cue/分类，而不是一般重要性理解。
3. **人工 span 监督。** fact/value start/length 由生成器直接提供；开放域需要弱监督、
   关系抽取或联合学习。
4. **V3 是多项改动同时加入。** value span、lexical fusion、token auxiliary loss
   尚未在同一协议下逐项独立消融。
5. **辅助 head 不是 copy head。** 当前生成仍使用 tied LM head；OOV、长数字和罕见多
   token 字符串可能需要显式 pointer/copy。
6. **逻辑 bytes 不是物理压缩。** 槽位张量按 M_max 预分配；只有实现 packed/ragged
   active slots 后，active-slot bytes 才可能转化为实际显存/状态节省。
7. **MoE 总参数增加。** Top-2 active parameters/token 较低不意味着 checkpoint、
   optimizer state 或实际 wall-clock 成本较低。
8. **普通 PyTorch kernel。** local attention、memory gather 和 MoE dispatch 都没有
   生产级 fused kernel；吞吐结果是当前原型成本，不是理论上限。
9. **不同协议不能硬排名。** Stage 2、stress v2/v3、A+B Dense、六模型 MoE、Keyformer
   的数据、任务、Top-k、Top-1/2、计时和 state schema 不完全相同。
10. **SWA 来源边界。** Sliding-window beats linear attention 未找到作者公开代码；
    StreamingLLM source-policy 只能称为 paper-aligned external baseline。

## 12. 下一阶段优化路线（建议按顺序）

### P0：先建立可解释的 answer-utilization 诊断

对每个新 checkpoint 同时报告：

~~~text
normal read EM
oracle target-slot read EM
fusion disabled EM
条件于 target retrieval hit 的 EM
teacher-forced answer NLL
free-generation EM
首 token rank / margin
~~~

这样可以把 write、retention、reader、fusion、decoder 五个环节分开，避免将满分
Recall 误读成满分任务能力。

### P1：多 seed 和严格组件消融

先为 seed=29、43 训练各自的 10M TinyStories backbone，再按相同 stress 协议训练：

~~~text
SWA-only / Fixed-LRU / old Gated / V3 Value-token Gated
~~~

V3 至少拆成：

1. value span + lexical fusion，但 token auxiliary loss=0；
2. value span + token loss，但 fusion 仍为旧 combined；
3. 完整 V3；
4. （可选）只加 untied output head。

不要从 seed-17 backbone 复制给 seed-29/43 后再声称是独立 seed；每个 seed 需要自己的
Stage 1 父模型。

### P2：降低任务捷径并扩大事实类型

逐步去掉 Remember/Store/Discard 这类直接 cue，增加：

- 多种句法和 query paraphrase；
- 多人、多属性、多事实；
- 冲突更新（同一人先喜欢 blue 后改为 navy）；
- 数字、日期、实体名、罕见多 token 字符串；
- 相同长度但不同潜在后续依赖的文本；
- query 未提前给出的关联记忆任务。

目标是测量模型是否真的按内容和后续需求选择记忆，而不是按提示词分类。

### P3：容量、Top-k 和长程扫描

至少扫描 memory_slots=4/8/16/32、Top-k=1/2/4/8、noise/slot 比例以及更长 delay。
报告质量—active slots—state bytes—延迟的 Pareto 曲线。固定 8 slots 只是当前压力
构造，不是最优容量。

### P4：真实精确复制路径

若 V3 在开放值测试上退化，按顺序尝试：

1. value token sequence cross-attention，而不是仅平均 lexical value；
2. memory-to-token contrastive/CE 的 hard-negative 版本；
3. value payload pointer/copy head，与普通 LM logits 做可学习混合；
4. 只对短 value path 保留跨轮梯度，filler 仍 detach；
5. 统一 query/answer 的空格、换行和 token 边界；
6. 最后再评估 untied LM head。

### P5：物理状态和 kernel 优化

只有以下工作完成后，才可声称“动态记忆节省物理显存/延迟”：

- packed/ragged active slots，而不是完整 M_max 张量；
- 读取无效槽位时跳过计算；
- fused/SDPA sliding-window attention；
- grouped-GEMM/MegaBlocks 风格 MoE dispatch；
- 单 token prefill/decode latency、peak allocated/reserved、state bytes 的统一 benchmark。

## 13. 接手检查清单

- [ ] 先读本文件和最新 MoE 对比报告；
- [ ] 确认 git status，不要 git reset --hard 或 git checkout --；
- [ ] 不覆盖 /root/autodl-tmp/.../runs/ 下已有 checkpoint、summary 和失败目录；
- [ ] 修改前建立新的代码快照；
- [ ] PYTHONPATH=adaptive_gated_fact_memory/src 下 23/23 测试通过；
- [ ] 新 run 使用唯一 --run-id，单卡串行；
- [ ] 在 config.resolved.json 中核对 seed、父 checkpoint SHA、slots、Top-k、MoE Top-k；
- [ ] 将 Dense、adaptive stress MoE（Top-2）和六模型 MoE（Top-1）分开报告；
- [ ] 结果至少包含 EM、answer NLL、survival、Recall、MRR、active slots、noise accept、
  state bytes、训练时间/吞吐、显存和参数量；
- [ ] 将逻辑 active bytes 明确标为逻辑量，不写成物理显存压缩；
- [ ] 多 seed、无 cue、冲突/多事实和真实 packed-state 优化完成前，保持 screening 级结论。
