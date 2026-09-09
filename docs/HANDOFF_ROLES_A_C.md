# Handoff：角色 A 与角色 C 的 TinyStories/TinyLM 实验

版本：roles_a_c_handoff_v1
适用对象：接手本仓库的下一位 Codex 或实验执行者
最后更新：2026-09-05
实验状态：角色 B（Longformer + Memformer）已经完成；角色 A、C 尚未完成本协议下的新实验。

本文件是**执行说明**，不是结果报告。接手者应按照这里的边界实现代码、运行检查、保存原始结果，并在此基础上另写 A/C 的结果报告。旧的 HANDOFF.md、STATUS.md 以及各框架目录中的 WikiText/旧微基准脚本仍然有参考价值，但不能把它们的结果直接当成本轮 TinyStories 验证集结果。

---

## 0. 先读什么、从哪里开始

### 可直接转发给下一位 Codex 的启动指令

你接手的是 TinyStories/TinyLM 三天版筛选实验的角色 A 或角色 C。先完整阅读本文件、`protocol.yaml`、`three_day_validation_screening.yaml` 和角色 B 的 runner/报告；不要把旧 WikiText 或 RTX 4070 微基准结果当作本轮结果。先做 correctness，再做 pilot，再冻结配置，最后跑 main、完整 validation、效率 benchmark 和 aggregate。所有原始 JSON、配置 hash、数据 hash、环境信息和失败记录都必须保留；Keyformer 只能作为同一 Full-Attention checkpoint 上的推理期 KV-cache policy 报告，不能和训练型 backbone 直接总排名。

先阅读以下文件：

1. experiments/tinystories_tinylm_v1/protocol.yaml：数据、模型和审计要求；
2. experiments/tinystories_tinylm_v1/three_day_validation_screening.yaml：三天版冻结配置、候选值和选择规则；
3. experiments/tinystories_tinylm_v1/run_person_b.py：角色 B 已验证的数据流、TinyLM 结构、训练循环、validation 和结果 schema 参考；
4. PERSON_B_TINYLM_REPORT.md、PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md、PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md：角色 B 的报告格式和结果解释边界；
5. experiments/tinystories_tinylm_v1/README.md：数据 cache、数据盘和复现实验入口；
6. 本文件：A/C 的具体工作边界和验收标准。

当前仓库信息：

~~~bash
cd /root/BDMI/26summerBDMI_transformer
git fetch myorigin role_B
git switch role_B
git pull --ff-only myorigin role_B
git status --short --branch
~~~

如果协作环境使用另外的集成分支，不要为了切换分支执行破坏性 reset/checkout；只需确认已经包含 role_B 的 B 角色代码和聚合结果。接手者必须先保留已有 B 结果，再开始 A/C 工作。

本机已知环境（实际运行时仍要写入 environment.json/config.resolved.json）：

~~~text
Python 3.12.3
PyTorch 2.12.1+cu130
Transformers 5.16.1
datasets 4.8.5
NumPy 2.4.6
PyArrow 25.0.1
GPU: NVIDIA RTX 4090 24GB
~~~

---

## 1. 研究问题和角色边界

本轮不是把所有方法改造成新架构，而是在同一个 causal TinyLM/TinyStories 协议下，取得五个训练型 backbone 和一个推理期 cache policy 的可追溯基线。用户自己的动态记忆架构（动态 active slots、内容驱动写入/读取/保留、Key–Value payload 等）属于后续实验，**不得混入 A/C baseline**。

本轮六个框架明确是：Longformer、Performer、Linformer、Reformer、Memformer、Keyformer。仓库 README 旧表中的 xFormer 仍可作为历史资产，但不在这次 A/C 任务范围内，也不能擅自替换 Reformer 或 Keyformer。

| 角色 | 负责对象 | 机制主题 | 必须交付 |
|---|---|---|---|
| A | Linformer、Performer | 低秩序列投影 vs 随机特征近似 | 两个模型的 correctness、pilot、10M-token train→validation、近似误差、速度/显存和分析 |
| C | Reformer、Keyformer；另训练一个 Full-Attention reference | LSH 稀疏连接 vs 推理期 KV-cache 淘汰 | Reformer 完整训练实验；Full reference checkpoint；Keyformer cache sweep；统一 schema 和报告 |

角色 C 的 Full-Attention 只是一份共享参考，不是“第七种框架”，也不计入六框架排名。Keyformer 不独立训练，必须明确标记为：

> inference-time KV-cache compression policy（推理期 KV-cache 压缩策略），不能和五个训练型 backbone 的 validation PPL 直接做无条件总排名。

A/C 的代码应独立写入：

~~~text
experiments/tinystories_tinylm_v1/
├── run_person_a.py              # 新建；或调用清晰拆分的公共 runner
├── run_person_c.py              # 新建；含 Full reference、Reformer、Keyformer eval
├── tinylm_common.py             # 推荐：从 run_person_b.py 抽出的公共数据/模型/训练工具
├── aggregate/
│   ├── person_a_*.json
│   └── person_c_*.json
└── runs/
    ├── person_a/
    └── person_c/
~~~

可以复用 B 的公共逻辑，但不要让 A/C 的实现悄悄改变 B 的已完成结果。若为减少重复而重构 run_person_b.py，必须先运行 B 的已有 correctness/aggregate 命令，并证明旧结果 schema 没有被破坏。

---

## 2. 冻结的公共实验协议

除非记录为明确的 protocol_deviation，以下设置不能在 main run 中临时改变。

### 2.1 数据和 tokenizer

~~~yaml
dataset: roneneldan/TinyStories
dataset_revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
tokenizer: openai-community/gpt2
tokenizer_revision: 607a30d783dfa663caf39e06633721c8d4cfcd7e
text_field: text
add_eos_at_document_end: true
document_order: pinned_parquet_order
cross_document_packing: true
block_size: 512
padding: none
loss_mask: all_packed_next_tokens
train_token_budget: 10000000
validation: full_validation_stream
~~~

具体规则：

- 每个故事文本末尾追加 GPT-2 EOS（ID 50256）；
- 按固定 parquet 文件顺序串接，再切成 512-token block；
- 允许跨文档 packing，EOS 后的跨文档 attention 对所有模型一致；
- 这只是三天内统一利用数据的方式，不能据此宣称模型学会了真实跨文档长期记忆；
- TinyStories 没有官方 test split，本轮只能写 validation 结果，不能写 test accuracy/PPL；
- 必须保存 token cache 的元数据、文件列表和数据 manifest/hash。

首次准备数据：

~~~bash
python -m pip install -r requirements-tinylm.txt
HF_ENDPOINT=https://hf-mirror.com \
  python experiments/tinystories_tinylm_v1/prepare_data.py

python experiments/tinystories_tinylm_v1/run_person_b.py \
  prepare-cache --train-loss-tokens 10000000 --context 512
~~~

如果使用数据盘（推荐用于 checkpoint、日志和长实验）：

~~~bash
export TINYSTORIES_CACHE_DIR=/root/autodl-tmp/26summerBDMI_transformer/data/cache/tinystories_tinylm_v1
export TINYSTORIES_DATA_DIR=/root/autodl-tmp/26summerBDMI_transformer/data/raw/tinystories/data
export TINYSTORIES_TOKENIZER_DIR=/root/autodl-tmp/26summerBDMI_transformer/data/tokenizer/gpt2
export PERSON_A_RUNS_DIR=/root/autodl-tmp/26summerBDMI_transformer/runs/person_a
export PERSON_A_AGGREGATE_DIR=/root/autodl-tmp/26summerBDMI_transformer/aggregate/person_a
export PERSON_C_RUNS_DIR=/root/autodl-tmp/26summerBDMI_transformer/runs/person_c
export PERSON_C_AGGREGATE_DIR=/root/autodl-tmp/26summerBDMI_transformer/aggregate/person_c
mkdir -p "$TINYSTORIES_CACHE_DIR" "$PERSON_A_RUNS_DIR" \
  "$PERSON_A_AGGREGATE_DIR" "$PERSON_C_RUNS_DIR" "$PERSON_C_AGGREGATE_DIR"
~~~

环境变量名称可以按 runner 的实际实现调整，但必须把最终绝对路径写入 resolved config；不要把数据盘路径默认为 Git 中存在。

### 2.2 公共 TinyLM 主干

五个训练型模型和 Full reference 使用相同的 decoder-only causal TinyLM：

| 参数 | 冻结值 |
|---|---:|
| layers | 6 |
| hidden size | 384 |
| heads | 8 |
| head dimension | 48 |
| FFN size | 1536 |
| normalization | Pre-LN |
| activation | GELU |
| position | RoPE，最大位置 32768 |
| dropout | 0.1 |
| attention dropout | 0.0 |
| vocabulary | 50,257 |
| causal | true |
| input/output embedding | tied |
| Linear bias | false |
| LayerNorm affine/bias | true/true |

公共参数量的协议期望值：

~~~text
29,925,504  # 当前标准设置，含 LayerNorm bias
29,920,512  # 关闭 LayerNorm bias 时的另一个值；不得混用
~~~

每个 run 必须同时报告：

- total_parameters；
- trainable_parameters；
- common_backbone_parameters；
- method_specific_parameters；
- 若随机特征矩阵或 hash 矩阵是 buffer，还要单独报告其 bytes，不能把不可训练 buffer 错写成 trainable parameter。

所有训练型方法都应使用与 B 相同的 state 初始化协议：seed 固定、公共参数名字/形状一致、初始化后记录 digest。比较前可调用类似 common_parameter_delta() 的函数，确认公共部分确实匹配。

### 2.3 优化和 token budget

~~~yaml
optimizer: AdamW
learning_rate: 3e-4
betas: [0.9, 0.95]
epsilon: 1e-8
weight_decay: 0.1
warmup_ratio: 0.03
schedule: cosine
minimum_learning_rate: 3e-5
gradient_clip_norm: 1.0
compute_dtype: BF16
parameter_dtype: FP32
context: 512
micro_batch_sequences: 4
gradient_accumulation_steps: 4
effective_batch_tokens: 8192
pilot_tokens: 1048576
main_tokens: 10000000
required_seed: 17
optional_seed: 29
early_stopping: false
~~~

显存不足时允许使用 micro_batch=2、gradient_accumulation=8，但 effective batch 必须仍为 8192 tokens。若首轮校准显示单个 run 会超过约 6 小时，可按协议把 main budget 降至 5M，并写入：

~~~json
{
  "protocol_deviation": {
    "field": "main_tokens",
    "requested": 10000000,
    "actual": 5000000,
    "reason": "time_cap"
  }
}
~~~

不能为了让结果更好看而临时改变 context、宽度、seed、数据顺序或删除失败 run。

### 2.4 GPU 分配

每人一张 RTX 4090 24GB；同一台机器上运行时必须使用不同的 CUDA_VISIBLE_DEVICES。不要在同一张卡上并发运行两个候选模型。A、B、C 的日志/checkpoint 目录必须隔离。

---

## 3. 先实现公共 runner，再实现方法 adapter

推荐把 B runner 中与方法无关的部分拆到 tinylm_common.py，至少包括：

~~~python
build_token_cache(split, requested_tokens)
PackedTokenStream.batch(...)
RotaryEmbedding
masked_loss_sum
evaluate
learning_rate_for_step
seed_all
environment_record
parameter_record
model_parameter_digest
atomic_json
~~~

训练型 attention adapter 的统一约定建议为：

~~~python
class CausalAttentionAdapter(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,                 # [batch, seq, hidden]
        rotary_emb: RotaryEmbedding,
        position_ids: torch.Tensor,           # [seq]
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:                        # [batch, seq, hidden]
        ...
~~~

注意：不同第三方包的 API 可能接收 [B, H, N, D]、只接收 embedding，或要求 padding 到 bucket 倍数。adapter 负责把它们转换到统一接口，不能把第三方默认的 causal=False 路径直接当作本实验结果。

训练 runner 至少提供以下命令或等价入口：

~~~text
correctness       只跑正确性和有限梯度检查
pilot             跑每个候选约 1,048,576 tokens，并输出选择 manifest
train             用已冻结配置跑 10M tokens
validate          对最后 checkpoint 和最佳 probe checkpoint 做完整 validation
benchmark         测 forward/prefill/decode latency、显存和长度矩阵
aggregate         汇总 raw summaries，不覆盖失败记录
~~~

如果把多个命令合并，仍必须能从命令行明确区分 pilot、main、validation 和 benchmark；不能只保留一个无法复现的 notebook。

---

## 4. 角色 A：Linformer + Performer

角色 A 的核心问题是：在相同 TinyLM 主干下，低秩序列投影和随机特征近似各自损失多少质量，换来多少速度/显存收益。

### 4.1 Linformer：需要写什么

现有 linformer/quality_wikitext.py 和 linformer/benchmark_perf.py 是旧 WikiText/性能脚本，不能直接作为本轮 runner。它们可以用于了解命名，但必须新建 TinyStories/RoPE/causal adapter。

候选和主配置：

~~~yaml
rank_candidates: [64, 128]
main_rank: 128
projection_length: 512  # 固定为训练 context
causal: true
~~~

必须重点解决因果性。标准 Linformer 对完整序列一次性使用同一个 E 投影 K/V；如果 query 在位置 t 能读到包含 t+1... 的投影结果，就发生未来泄漏。以下做法均可接受，但必须在 resolved config 和报告中写清楚实际采用哪一种：

1. prefix-causal projection：第 t 个 query 的 K/V 投影只由 0..t 的 K/V 计算；
2. 分段因果 projection：先声明 segment/block 边界，每个 query 只能读当前及过去 segment 的投影；
3. 其他有形式化因果证明的低秩构造。

不要为了沿用旧代码而使用“整段 K/V 先投影，再对 query 加下三角 mask”的伪因果实现；投影本身已经可能混入未来信息。若在时间内无法实现无泄漏方案，应把 run 标为 blocked_unverified_causality，而不是提交一个看似更好的 PPL。

如果 prefix projection 采用逐位置重算，必须单独记录这种实现开销；优先使用累计/分段可复用的计算，或明确说明这是 correctness-first 的参考实现，不能把它的速度直接解释为 Linformer 的理论复杂度。

adapter 要求：

- Q/K/V 投影、RoPE、输出投影与 TinyLM 其他模型保持一致；
- 明确 projection 是 per-head、shared-head 还是跨层共享；
- 明确 projection 参数是否训练、初始化 seed 和 dtype；
- 支持 512 训练长度；
- 对 1024 只在预先实现并记录插值/独立 projection 时测试，不能把 512 projection 的无声明外推称为原生支持；
- 报告 projection 的参数量和实际中间 tensor bytes。

建议实现的接口（伪代码）：

~~~python
class CausalLinformerAttention(CausalAttentionAdapter):
    def __init__(self, dim, heads, context, rank, projection_mode, ...):
        # 创建 E_k/E_v 或等价的因果分段投影参数
        ...

    def forward(self, hidden, rotary_emb, position_ids, attention_mask=None):
        q, k, v = self.project_qkv(hidden)
        q, k = rotary_emb.apply_qk(q, k, position_ids)
        # 每个 query 位置只能使用 prefix-causal 的 projected K/V
        out = self.causal_low_rank_attention(q, k, v, attention_mask)
        return self.merge_output(out)
~~~

这段伪代码不是要求固定某个数学实现，而是要求接口和因果约束可审计。

### 4.2 Linformer：必须检测什么

用 FP32、小序列和固定权重先跑 correctness，再进入 BF16 训练：

1. shape_dtype：输入 [B,N,384]，输出同形状，N=1、2、32、128、512；
2. future_invariance：只修改位置 t 之后的输入，检查位置 0..t 的输出最大绝对差接近 0；
3. finite_gradient：随机 loss backward，所有 trainable 参数的 loss/gradient finite；
4. mask_padding：改变 padding token 或 padding mask，不能改变有效 prefix 的输出；
5. projection_length_guard：N 超出 512 时必须明确报错、padding 或使用已声明的长度策略，不得静默错误；
6. rank_shape：rank=64/128 的参数和输出 shape 正确；
7. deterministic_seed：相同 seed 得到相同 projection/输出，不同 seed 的差异被记录；
8. dense_reference_fidelity：同一组 Q/K/V 下与 dense causal attention 比较 MSE、relative L2、cosine similarity。低秩近似不要求零误差，但指标必须有限且随 rank 变化可解释；
9. 若某种 rank/长度理论上可退化为完整表示，做相应 equivalence test；否则不要伪造“full rank exact”。

必须保存每个长度和 rank 的 raw fidelity 记录，不能只保存最后一个均值。

### 4.3 Performer：需要写什么

现有 performer/quality_wikitext.py 中的旧 fidelity 路径使用了 causal=False，不能直接复用为 causal TinyLM 实验。可复用 performer-pytorch/performer_pytorch/performer_pytorch.py 的 FastAttention/causal_linear_attention_noncuda，但 adapter 必须显式设置 causal。

候选和主配置：

~~~yaml
feature_candidates: [64, 128]
main_features: 128
causal: true
redraw: false
projection_seed: 17  # 必须独立记录；可与模型 seed 相同但不能隐含
~~~

要求：

- 使用 causal FAVOR+，不是 global non-causal FAVOR+；
- random feature/projection matrix 在一个 run 内固定，不进行隐式 redraw；
- 记录 projection matrix 的 seed、shape、dtype、hash；
- 若使用 fast_transformers CUDA kernel，记录 backend；若回退到 Python/PyTorch causal scan，也记录 backend；
- Q/K/V、RoPE、输出投影、dropout 和 TinyLM 其余层与公共协议一致；
- 对 BF16 训练和 FP32 correctness 分别记录 dtype。

建议 adapter 结构：

~~~python
class CausalPerformerAttention(CausalAttentionAdapter):
    def __init__(self, dim, heads, num_features, projection_seed, ...):
        self.fast_attention = FastAttention(
            dim_heads=dim // heads,
            nb_features=num_features,
            causal=True,
        )
        self.fix_projection_matrix_()  # 或等价的 redraw=False 机制

    def forward(self, hidden, rotary_emb, position_ids, attention_mask=None):
        q, k, v = self.project_qkv(hidden)
        q, k = rotary_emb.apply_qk(q, k, position_ids)
        out = self.fast_attention(q, k, v)
        return self.merge_output(out)
~~~

如果第三方模块无法接受本 TinyLM 的 RoPE 或 bias 约定，就写一个薄 wrapper；不要为了调用方便切换为绝对位置 embedding 或非因果路径。

### 4.4 Performer：必须检测什么

至少包括：

1. shape_dtype；
2. future_invariance；
3. finite_gradient；
4. fixed_projection_reproducibility：相同 model seed + projection seed 的矩阵和输出一致；
5. different_projection_seed_recorded：不同 projection seed 的 fidelity/训练结果分开记录；
6. dense_causal_fidelity：MSE、relative L2、cosine similarity；
7. feature_count_sweep：64 vs 128 的误差、validation NLL/PPL、tokens/s、latency、显存；
8. long_sequence_guard：长度增加时不发生 silent shape 错误或未来泄漏；
9. 若依赖 CUDA causal product，测试无该依赖时的 fallback，并把 backend 差异写入结果。

### 4.5 A 的 pilot、main 和分析

pilot 候选固定为：

~~~text
Linformer rank=64, 128
Performer features=64, 128
~~~

每个候选先通过 correctness，再跑约 1,048,576 train tokens。选择顺序不能改变：

1. 正确性全部通过；
2. loss/gradient finite；
3. pilot validation NLL 更低者优先；
4. 若 NLL 在 5% 内，再比较训练吞吐；
5. 若吞吐在 10% 内，再比较峰值显存。

把选择过程写到 aggregate/person_a_pilot_selection.json，包括通过和淘汰候选，不能只留下主配置。冻结后跑：

~~~text
Linformer rank=128, seed=17, 10M tokens
Performer features=128, seed=17, 10M tokens
~~~

若时间允许再补 seed=29，但不得用 seed=29 事后替换 seed=17 主结果。

A 的报告至少回答：

- rank/features 增大是否降低近似误差？
- 近似误差是否转化为 validation NLL/PPL 差异？
- 哪个方法的训练 tokens/s、forward latency、显存更有优势？
- 速度是理论复杂度带来的，还是当前 Python/PyTorch backend 的实现差异？
- 哪些长度或 mask 情况失败？
- Linformer 的因果 projection 变体与标准非因果论文实现有何差别？

不要把“rank=128 比 features=128 好”写成严格公平的容量结论；两者不是同一种资源单位，必须同时展示参数、投影/特征 bytes 和实际运行指标。

---

## 5. 角色 C：Reformer + Full reference + Keyformer

角色 C 的核心问题是：LSH 稀疏连接能否在训练阶段保持质量并降低计算；Keyformer 在同一已训练模型上能否用选择性 KV cache 降低推理状态开销。

### 5.1 Reformer：需要写什么

现有 reformer/quality_wikitext.py 有 causal=False 的旧 benchmark/fidelity 路径和 structural placeholder，不能直接作为本轮结果。应包装或实现真正的 causal LSH attention。

候选和主配置：

~~~yaml
bucket_size_candidates: [32, 64]
main_bucket_size: 64
n_hashes: 4
causal: true
reversible: false
~~~

必须处理：

- LSH bucket padding；第三方 Reformer 通常要求长度满足 2 * bucket_size 的倍数；
- padding mask 必须防止 padded token 被读写；
- causal mask 必须在 bucket 内和跨 hash round 都有效；
- hash projection/rotation 的 seed 和 rehash_each_round 配置必须记录；
- 关闭 reversible 以降低接入风险，并明确记录；
- 不要把 non-causal module 运行结果标成 causal；
- 对不满足长度约束的输入，要显式 padding 并裁剪，或清晰报错。

建议 adapter 结构：

~~~python
class CausalReformerAttention(CausalAttentionAdapter):
    def __init__(self, dim, heads, bucket_size, n_hashes, hash_seed, ...):
        self.inner = LSHSelfAttention(
            dim=dim,
            heads=heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=True,
            allow_duplicate_attention=True,  # 若第三方版本需要，写明原因
        )

    def forward(self, hidden, rotary_emb, position_ids, attention_mask=None):
        # 处理 RoPE、padding 到合法长度、attention mask、调用 inner、裁剪回原长度
        ...
~~~

如果第三方 LSH 模块只能在 embedding 内部生成 Q/K/V，需确认 RoPE 施加位置和 TinyLM 的 QKV 权重一致；无法做到时，写自定义 wrapper 并在报告中标出与标准 Reformer 的差异。

### 5.2 Reformer：必须检测什么

正确性闸门至少包括：

1. shape_dtype；
2. future_invariance；
3. finite_loss_and_gradient；
4. padding_mask：同一有效 prefix 在不同 padding 长度下输出一致；
5. bucket_length_constraint：N=32、64、128、256、512 以及非法长度的行为明确；
6. hash_determinism：相同 hash seed 的 buckets 相同；不同 seed 的差异可记录；
7. bucket_statistics：每个长度/hash round 的 bucket occupancy、空桶数、碰撞/重复 attention 比例；
8. dense_fidelity：在相同或可比 Q/K/V 下记录 MSE、relative L2、cosine similarity。LSH 近似不要求精确相等，但必须说明参考路径和比较条件；
9. 若设置 full bucket/full attention 能形成等价条件，做等价检查；普通 bucket=64 的 LSH 不得声称与 dense exact；
10. OOM/unsupported length：保留失败记录，不在 aggregate 中静默删除。

pilot：

~~~text
bucket_size=32, n_hashes=4
bucket_size=64, n_hashes=4
~~~

冻结原则仍是 correctness → finite → validation NLL → throughput → memory。main 至少运行 bucket=64、hashes=4、seed=17，10M tokens。

Reformer 报告要把三种量分开：

- 理论 LSH/稀疏复杂度；
- attention-only 实测 latency/memory；
- 完整 TinyLM train/validation 的 wall-clock 和 tokens/s。

当前 Python/PyTorch adapter 可能比 dense SDPA 慢，这不是 Reformer 理论复杂度失效的证明，只能说明当前实现 backend 的工程开销。

### 5.3 Full-Attention reference：C 必须先完成

Keyformer 必须使用一个共享 Full-Attention TinyLM checkpoint。角色 C 负责训练并保存它：

~~~text
method: full_attention
seed: 17
train_tokens: 10000000
context: 512
optimizer/data/token order: 与所有训练型模型完全一致
~~~

需要保存：

~~~text
full_attention_tinylm_tinystories_v1/
├── checkpoint.tokens_10000000.pt
├── config.resolved.json
├── summary.json
├── metrics.jsonl
├── environment.json
└── checkpoint_sha256.txt
~~~

Full reference 的完整 validation 必须先完成，再启动 Keyformer。把 checkpoint 的绝对路径、相对路径、sha256、seed、token budget、model config hash 写入 person_c_keyformer_manifest.json。如果 reference 未完成，Keyformer 只能做代码 correctness smoke，不能写正式质量结果。

Full reference 不计入六个框架名额，但在所有质量/推理表格中作为 FullAttentionReference 列出现。

### 5.4 Keyformer：需要写什么

仓库已有：

~~~text
models/keyformer.py
models/gpt2_keyformer.py
keyformer/test_keyformer.py
keyformer/test_gpt2_keyformer.py
keyformer/benchmark_perf.py
~~~

这些代码主要面向 GPT-2 Medium、绝对位置 embedding 和旧质量协议。不能直接加载 GPT-2 Medium 来代替本轮 TinyLM；需要写 TinyLM/RoPE 适配层，或把 KVCacheState/policy 逻辑安全地复用到新 engine。

Keyformer 的对照策略必须是：

~~~text
FullKV
Keyformer
RecentWindow
Random+Recent
~~~

cache ratio：

~~~text
0.50 / 0.75 / 1.00
recent_ratio = 0.5
tau_init = 1.0
tau_delta = 0.01
~~~

Keyformer 是推理期 cache policy，因此不调用 TinyLM optimizer，不额外训练参数。评估流程建议采用同一 Full checkpoint 的顺序 prefill + token-by-token decode：

1. 取固定 validation prompt/suffix 窗口；
2. FullKV、Keyformer、RecentWindow、Random+Recent 使用完全相同 token、RoPE、dtype 和 seed；
3. prefill 建立 cache；
4. 先用 prefill 最后一个位置的 logits 评分第一个目标 token；此后每次把上一个真实 token 作为 decode_step 输入，用返回 logits 评分下一个真实目标 token，再按 policy 更新 cache。这样所有策略评分的是同一组 teacher-forced targets；
5. 对每个 ratio 重复相同窗口，汇总 token-weighted NLL、PPL/生成质量和效率；
6. 若按 512 block 重置 cache，必须在方法和 JSON 中明确 reset_policy=per_block；不要把 block reset 的结果描述为无限长会话记忆。

最重要的实现约束：

- 保留 K/V 的**原始绝对位置**；物理删除 cache token 不能令 RoPE 位置重新编号；
- 新 token 的 Q/K 必须使用其真实 position_id=seen_tokens；
- cache token 数不能超过预算；
- 每层 cache 独立维护，batch item/head 不能互相泄漏；
- reset() 必须清空所有 policy state、cache 和 seen position；
- decode 时 batch reorder 后 state 必须跟着样本走；
- ratio=1.0 的 Keyformer 必须与 FullKV 等价（允许明确的浮点容差）；
- ratio=1.0 的 Recent/Random 也应作为对照检查，不要只测 Keyformer；
- cache selection 的开销必须计入 decode latency，不能只测压缩后的 matmul。

建议新增的 engine 接口：

~~~python
class TinyLMKVCacheEngine:
    def reset(self) -> None: ...
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def decode_step(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def cache_summary(self) -> dict[str, int | float]: ...
    def generate_greedy(self, input_ids, max_new_tokens): ...
~~~

建议让每层 state 至少含有：

~~~text
keys, values, original_positions, accumulated_scores,
decode_step, seen_tokens, batch identity (如需要)
~~~

### 5.5 Keyformer：必须检测什么

必须逐项写入 person_c_keyformer_correctness.json：

1. fullkv_prefill_equivalence：TinyLM FullKV engine 与同一实现的 dense causal prefill 一致；
2. ratio_1_keyformer_equivalence：Keyformer@1.0 与 FullKV 的 logits/NLL/greedy tokens 一致；
3. ratio_1_recent_random_equivalence：适用时检查；
4. cache_budget：任何 decode step 的 cached tokens ≤ budget；
5. original_position_preservation：物理删除 token 后保留位置序列，RoPE 不错位；
6. cache_reset：新 prompt/新 conversation 不继承旧 cache；
7. batch_reorder：交换 batch 样本后输出和 cache state 对应关系正确；
8. deterministic_policy_seed：相同 seed 的 Keyformer/Random selection 可复现；
9. finite_logits_and_grad_free_eval：eval logits finite；
10. shape_dtype：prefill/decode 的 batch、head、position、dtype 检查；
11. no_future_read：decode step 只能读取当前及过去 K/V；
12. state_storage_accounting：每层/总 KV bytes 与实际 tensor numel×element_size 一致。

若 ratio=1.0 不等价，立即停在 correctness 阶段修复；不得继续做质量 sweep。

Keyformer 正式指标：

- generated-suffix token-weighted NLL/PPL；
- greedy token exact-match 或与 FullKV 的 token agreement；
- logit max-abs error、relative L2、cosine similarity；
- prefill latency；
- decode p50/p95 latency；
- decode tokens/s；
- peak CUDA allocated/reserved memory；
- 每层和总 KV bytes；
- compression_ratio = 1 - compressed_kv_bytes / full_kv_bytes；
- cache budget、实际 cached tokens、selection/gather 时间（若可分解）。

Keyformer 结果的正确写法是“同一 Full checkpoint 上的推理期 cache 压缩质量—资源曲线”。不能写成“Keyformer 训练 PPL 胜过 Reformer/Performer”。

---

## 6. 统一 correctness 测试矩阵

所有 A/C 方法都要先在 FP32 下通过下表，再开始 BF16 pilot。每个测试保存 value、limit/minimum、passed 和环境信息；不能只打印一句 ok。

| 测试 ID | 适用对象 | 判定 |
|---|---|---|
| shape_dtype | A/C 全部 | 输出 shape 与输入一致；dtype 合法 |
| future_invariance | 所有 causal attention | 改变未来 token 不改变过去位置输出，误差在 FP32 limit 内 |
| finite_gradient | Linformer/Performer/Reformer/Full | loss、grad norm、参数梯度均 finite |
| padding_mask | Linformer/Reformer | padding 不影响有效 prefix |
| projection_or_hash_determinism | Linformer/Performer/Reformer | 相同 seed 可复现，seed 写入结果 |
| dense_fidelity | Linformer/Performer/Reformer | MSE、relative L2、cosine 全部 finite |
| length_guard | A/C attention | 合法/非法长度行为明确 |
| full_window_equivalence | 若某配置满足等价条件 | 只在数学条件成立时声称 exact |
| fullkv_ratio1_equivalence | Keyformer | ratio=1 与 FullKV logits/tokens/NLL 等价 |
| position_preservation | Keyformer | 原始绝对位置不被物理压缩改变 |
| cache_budget | Keyformer | cache 不超过预算 |
| state_reset | Keyformer；如有 recurrent adapter | 新样本不会继承旧 state |
| batch_reorder | Keyformer；有 batch state 的方法 | reorder 后 state 跟样本移动 |
| no_nan_no_oom_record | 全部 | 失败也生成 raw failure record |

future-invariance 的推荐实现：在相同 prefix 后拼接两个不同 future，分别 forward；只比较 prefix 输出。不要用训练中随机 dropout 直接判定，测试时必须 eval() 或固定 RNG。

近似误差统一定义：

~~~text
MSE = mean((approx - reference)^2)
relative_L2 = ||approx-reference||_2 / max(||reference||_2, 1e-12)
cosine = cosine_similarity(vec(reference), vec(approx))
~~~

dense reference 必须和近似方法共享可比的 Q/K/V、输入、RoPE 和 mask；如果第三方 Reformer 无法做到同一 QKV，报告中必须写明这是 module-level comparison，而不是伪装成严格同权重 fidelity。

---

## 7. 运行顺序和建议命令

不要一上来跑 10M。顺序必须是：

~~~text
环境/数据检查
→ adapter import/shape smoke
→ correctness
→ pilot candidates
→ 冻结主配置并写 selection manifest
→ main train
→ last/best full validation
→ efficiency benchmark
→ aggregate/report
~~~

命令名可按实际 runner 调整，以下是建议形式：

### 7.1 角色 A

~~~bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/tinystories_tinylm_v1/run_person_a.py correctness

CUDA_VISIBLE_DEVICES=0 \
python experiments/tinystories_tinylm_v1/run_person_a.py pilot \
  --method linformer --values 64 128 --tokens 1048576 --seed 17

CUDA_VISIBLE_DEVICES=0 \
python experiments/tinystories_tinylm_v1/run_person_a.py pilot \
  --method performer --values 64 128 --tokens 1048576 --seed 17

CUDA_VISIBLE_DEVICES=0 \
python experiments/tinystories_tinylm_v1/run_person_a.py train \
  --method linformer --rank 128 --run-id main_a_linformer_r128_s17 \
  --train-tokens 10000000 --seed 17 --full-validation

CUDA_VISIBLE_DEVICES=0 \
python experiments/tinystories_tinylm_v1/run_person_a.py train \
  --method performer --features 128 --run-id main_a_performer_f128_s17 \
  --train-tokens 10000000 --seed 17 --full-validation

CUDA_VISIBLE_DEVICES=0 \
python experiments/tinystories_tinylm_v1/run_person_a.py aggregate
~~~

### 7.2 角色 C

~~~bash
CUDA_VISIBLE_DEVICES=2 \
python experiments/tinystories_tinylm_v1/run_person_c.py correctness

CUDA_VISIBLE_DEVICES=2 \
python experiments/tinystories_tinylm_v1/run_person_c.py train-reference \
  --run-id full_attention_tinylm_tinystories_v1_s17 \
  --train-tokens 10000000 --seed 17 --full-validation

CUDA_VISIBLE_DEVICES=2 \
python experiments/tinystories_tinylm_v1/run_person_c.py pilot \
  --method reformer --bucket-sizes 32 64 --n-hashes 4 \
  --tokens 1048576 --seed 17

CUDA_VISIBLE_DEVICES=2 \
python experiments/tinystories_tinylm_v1/run_person_c.py train \
  --method reformer --bucket-size 64 --n-hashes 4 \
  --run-id main_c_reformer_b64_h4_s17 --train-tokens 10000000 \
  --seed 17 --full-validation

CUDA_VISIBLE_DEVICES=2 \
python experiments/tinystories_tinylm_v1/run_person_c.py keyformer \
  --reference-checkpoint /path/to/full_attention_tinylm_tinystories_v1_s17/checkpoint.tokens_10000000.pt \
  --cache-ratios 0.50 0.75 1.00 --recent-ratio 0.5 --seed 17

CUDA_VISIBLE_DEVICES=2 \
python experiments/tinystories_tinylm_v1/run_person_c.py aggregate
~~~

正式命令运行前，先用 --tokens 8192 或等价 smoke 验证参数名和路径。命令中的 /path/to/... 必须替换为真实 checkpoint，并把 sha256 写入 manifest。

---

## 8. 每个 run 必须保存什么

训练型 run 至少有：

~~~text
run_dir/
├── config.resolved.json       # 所有默认值展开后的配置 + config_hash
├── metrics.jsonl              # step/train/probe/checkpoint/failure 原始事件
├── summary.json               # 最终汇总
├── checkpoint.tokens_*.pt     # main/最佳 probe（按磁盘空间保留）
├── environment.json           # Python/PyTorch/CUDA/GPU/backend
├── data_manifest.json         # split、parquet、tokenizer、cache hash
└── plots/                     # train loss、validation probe、效率曲线
~~~

Keyformer eval 至少有：

~~~text
keyformer_run_dir/
├── config.resolved.json
├── metrics.jsonl
├── summary.json
├── correctness.json
├── reference_checkpoint_manifest.json
├── environment.json
└── plots/
~~~

summary.json 建议统一采用以下结构（可增加字段，但不要删除这些语义）：

~~~json
{
  "protocol_version": "tinystories_tinylm_v1",
  "role": "person_a_or_person_c",
  "method": "linformer_or_performer_or_reformer_or_full_attention_or_keyformer",
  "mode": "pilot_or_main_or_reference_or_inference",
  "run_id": "...",
  "seed": 17,
  "status": "ok_or_failed_or_oom_or_blocked_unverified_causality",
  "config_hash": "sha256...",
  "data_manifest_hash": "sha256...",
  "tokens_completed": 10000000,
  "optimizer_steps": 1221,
  "train": {
    "final_loss": null,
    "final_ppl": null,
    "tokens_per_second": null,
    "step_time_seconds": null,
    "peak_allocated_bytes": null,
    "peak_reserved_bytes": null
  },
  "validation": {
    "probe_curve": [],
    "last_checkpoint": {"tokens": null, "nll": null, "ppl": null},
    "best_probe_checkpoint": {"tokens": null, "nll": null, "ppl": null},
    "prediction_tokens": null
  },
  "efficiency": {
    "context_lengths": [],
    "latency_ms": [],
    "tokens_per_second": [],
    "peak_allocated_bytes": [],
    "peak_reserved_bytes": []
  },
  "parameters": {
    "total": null,
    "trainable": null,
    "common": null,
    "method_specific": null
  },
  "method_specific": {},
  "correctness": {},
  "failure": null,
  "protocol_deviation": null
}
~~~

方法特有字段建议：

~~~json
{
  "linformer": {"rank": 128, "projection_length": 512, "projection_mode": "..."},
  "performer": {"num_features": 128, "projection_seed": 17, "backend": "..."},
  "reformer": {"bucket_size": 64, "n_hashes": 4, "hash_seed": 17, "bucket_stats": {}},
  "keyformer": {
    "policy": "keyformer",
    "cache_ratio": 0.5,
    "recent_ratio": 0.5,
    "cached_tokens": null,
    "kv_bytes_total": null,
    "compression_ratio": null,
    "prefill_p50_ms": null,
    "decode_p50_ms": null,
    "decode_p95_ms": null,
    "logit_error": {}
  }
}
~~~

每行 metrics.jsonl 至少标明 event、timestamp、step 或 tokens_seen。OOM、NaN、unsupported length、future leakage 都要写入一条 failure event，并在 aggregate 中保留。

---

## 9. 指标怎么计算

### 9.1 质量

训练型模型使用完整 validation stream 的 token-weighted causal NLL：

~~~text
NLL = sum(token_loss_i) / number_of_valid_prediction_tokens
PPL = exp(NLL)
~~~

必须分别报告：

- 训练中固定 262,144-token validation probe 曲线；
- 最后 checkpoint 的完整 validation NLL/PPL；
- probe 最佳 checkpoint 的完整 validation NLL/PPL；
- prediction_tokens，防止不同 run 因 mask 错误比较了不同数量 token。

Keyformer 不独立训练，质量指标应写成同一 Full checkpoint 的推理窗口结果：

- suffix token-weighted NLL/PPL；
- 与 FullKV 的 greedy token agreement；
- logit max-abs、relative L2、cosine；
- 如使用 block reset、prompt/decode 长度或只评估 suffix，必须写清楚。

### 9.2 近似误差

Linformer、Performer、Reformer 的 attention-level fidelity 用共同输入和 dense causal reference 计算：

~~~text
MSE
relative_L2
cosine_similarity
~~~

每个误差记录都要带：method、seed、length、rank/features/bucket/hash、dtype、backend、mask 模式。不要把随机初始化的旧 fidelity 数字和训练后 TinyLM 的 validation PPL 混成一个指标。

### 9.3 速度和显存

训练：

~~~text
tokens_per_second = completed_train_tokens / measured_wall_seconds
step_time = optimizer_step_elapsed_seconds
~~~

请同时记录“纯训练 step 时间”和“包含 validation/checkpoint 的总 wall time”，否则不同 checkpoint interval 会造成误读。

推理/attention benchmark：

- CUDA warmup 至少 10 次，timed runs 至少 30 次；
- 报告 mean、median/p50、p95、std；
- GPU 使用 CUDA events 或同步计时；
- 显存同时报 max_memory_allocated 和 max_memory_reserved；
- Keyformer 另报物理 KV bytes，因为 allocator peak 可能包含碎片和临时 tensor；
- 长度矩阵至少包含 context=512；若方法原生支持且规则已声明，可加 1024/2048/4096；
- Reformer 长度必须满足 bucket/padding 约束；Linformer 不能把 512 projection 无声明外推到 1024。

常用派生量：

~~~text
speedup = baseline_latency / method_latency
memory_saving = 1 - method_bytes / baseline_bytes
compression_ratio = 1 - compressed_kv_bytes / full_kv_bytes
quality_delta_ppl = method_ppl - reference_ppl
relative_quality_delta = method_nll / reference_nll - 1
~~~

### 9.4 失败边界

如果出现 OOM、NaN、unsupported length、mask mismatch 或 future leakage：

- 保留该 run 的配置和已经产生的 metrics；
- status 写为 oom、nan、unsupported_length 或 failed_correctness；
- 在 aggregate 中保留失败行；
- 报告“最大成功长度”和“首个失败长度”；
- 不用删掉失败点来制造平滑曲线。

---

## 10. 如何分析和比较

不要只按一个 PPL 排名。最终 A/C 报告至少分成以下五层。

### 10.1 质量层

训练型 backbone 只比较在相同 data order、token budget、seed 和 TinyLM 主干下的 validation NLL/PPL。Full reference 是质量上界参考。Keyformer 单独报告 inference-window NLL/生成保持率，不放入五个训练型 backbone 的同一排名列。

### 10.2 训练资源层

比较：

- 总参数和方法特有参数；
- train tokens/s；
- optimizer step time；
- 峰值 allocated/reserved memory；
- 达到相同 probe NLL 所需 token/时间（若曲线有交点）。

如果不同方法参数量不完全相等，必须写“公共主干匹配、方法特有容量不同”，不能写成严格等参数比较。

### 10.3 推理资源层

比较：

- full forward/prefill latency；
- decode p50/p95 latency（Keyformer）；
- tokens/s；
- attention 临时 tensor bytes；
- Keyformer 每层/总 KV bytes 和压缩比例；
- 长度增长趋势和失败边界。

训练吞吐、attention-only latency、完整 forward latency 和 decode latency 必须分栏；不要用其中一个代替另外三个。

### 10.4 正确性层

先回答“能否安全比较”，再回答“谁更快/更好”：

- 是否通过 causal future-invariance？
- mask/padding 是否正确？
- projection/hash 是否可复现？
- gradient 是否 finite？
- FullKV@1.0 是否等价？
- cache reset/reorder 是否正确？

任何 correctness 未通过的模型只能列为“实现状态/失败边界”，不能列入质量冠军结论。

### 10.5 机制解释层

角色 A 应重点解释：

- Linformer 的 rank 如何控制 sequence projection 容量和长度适配；
- Performer 的 feature 数量如何控制随机近似方差；
- 两者误差—速度—显存曲线为什么可能不同。

角色 C 应重点解释：

- Reformer 的 bucket/hash 如何影响碰撞、可见范围和训练质量；
- Keyformer 的选择性 cache 如何在固定预算下保留历史 token；
- Keyformer 的选择/gather overhead 何时抵消 attention 节省；
- Recent/Random 的资源相似但质量可能不同的原因。

跨角色汇总时可以画 Pareto 图：横轴 latency 或显存，纵轴 validation PPL/NLL；但必须把 Keyformer 标为另一种 marker/面板。建议至少输出：

1. trainable backbones 的 validation PPL vs train tokens/s；
2. trainable backbones 的 PPL vs peak memory；
3. attention-level approximation error vs latency；
4. Keyformer cache ratio vs KV bytes/quality/decode latency。

### 10.6 结论边界

报告中必须明确以下限制：

- seed=17 的 screening 不能证明统计显著；
- 10M tokens 不是 TinyStories 完整收敛；
- packed TinyStories 不能证明真实对话长期记忆；
- 当前 Python/PyTorch backend 的实测速度不等于 fused kernel 的理论速度；
- 不同方法的 rank/features/bucket/slots 不是同一容量单位；
- Keyformer 不是独立训练 backbone；
- 任何未通过 causal/mask/position test 的结果都不能作为模型优劣证据。

---

## 11. 交付物检查清单

### 角色 A 必交

- [ ] run_person_a.py 或等价可复现 runner；
- [ ] Linformer causal adapter，且有 future-invariance 证据；
- [ ] Performer causal FAVOR+ adapter，固定 projection 和 seed；
- [ ] A 全部 correctness raw JSON；
- [ ] rank/features pilot 选择 manifest；
- [ ] 两个主配置的 10M run（或有记录的 5M fallback）；
- [ ] last/best full validation；
- [ ] fidelity、训练效率、推理效率、显存和失败边界；
- [ ] aggregate/person_a_*.json；
- [ ] PERSON_A_TINYLM_REPORT.md（详细）和简短结果摘要；
- [ ] 配置 hash、数据 manifest hash、环境信息。

### 角色 C 必交

- [ ] run_person_c.py 或等价可复现 runner；
- [ ] Reformer causal LSH adapter，含 bucket/padding/mask/hash 检查；
- [ ] Full-Attention shared reference 的 checkpoint、完整 validation 和 sha256 manifest；
- [ ] Reformer pilot/main/validation/efficiency；
- [ ] Keyformer TinyLM/RoPE inference engine 或安全 adapter；
- [ ] FullKV/Keyformer/Recent/Random cache sweep；
- [ ] ratio=1.0 equivalence、位置保留、reset/reorder/budget 证据；
- [ ] Keyformer prefill/decode latency、KV bytes、压缩率、质量指标；
- [ ] aggregate/person_c_*.json；
- [ ] C 详细报告和简短结果摘要；
- [ ] 配置 hash、数据 manifest hash、环境信息。

### 粗略时间预算（RTX 4090 24GB）

这些只是排程参考，不是可替代实测的承诺：数据/token cache 约 20–90 分钟；每个候选的 1M-token pilot 约 15–60 分钟，未融合的 Python 稀疏实现可能达到 1–2 小时；每个 10M-token 训练 run 约 30–120 分钟，Reformer 或其他未融合实现可能达到 2–4 小时；完整 validation 约 10–40 分钟；Keyformer ratio sweep 约 20–60 分钟。首轮先做 100-step calibration；若单个 run 预计超过 6 小时，按协议降到 5M tokens 并记录 deviation，不要静默改变设置。

### 代码和文档验收

完成前运行：

~~~bash
git diff --check
python -m py_compile experiments/tinystories_tinylm_v1/*.py
~~~

再运行 A/C correctness。若引入新测试目录，使用仓库实际测试框架执行并保存命令/版本。最终检查：

~~~bash
git status --short
find experiments/tinystories_tinylm_v1/aggregate -maxdepth 1 -type f | sort
~~~

Git 通常只追踪代码、协议、manifest、聚合结果和报告；原始数据、tokenizer cache、训练日志和 .pt checkpoint 可能被 .gitignore 忽略。交接时必须在报告中注明：哪些文件在 Git，哪些只在数据盘，并给出数据盘的绝对路径和 sha256。

---

## 12. 给接手 Codex 的最终执行原则

1. 先确认数据、依赖、GPU 和 B 结果可读，再写 adapter；
2. 先 correctness，后 pilot，最后 main；
3. 不把旧 WikiText/微基准结果冒充 TinyStories validation；
4. 不使用 non-causal 代码冒充 causal attention；
5. 不把 Keyformer 当作独立训练模型；
6. 不删除失败 run，不用事后最优 seed/参数替换冻结配置；
7. 原始 JSON、配置 hash、环境和数据 hash 比漂亮的图更重要；
8. 每个结论都区分“已测事实”“理论解释”和“尚未验证的推断”；
9. 用户的动态混合记忆创新机制留到后续独立实验；
10. 如果某个实现无法在时间内满足因果性或位置正确性，诚实报告 blocked/unverified，而不是提交不可解释的数字。
