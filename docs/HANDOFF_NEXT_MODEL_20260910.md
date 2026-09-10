# 交接文档：自适应门控事实记忆与 MoE 实验

更新时间：2026-09-10（UTC）  
工作区：`/root/BDMI/26summerBDMI_transformer`  
实验数据盘：`/root/autodl-tmp/26summerBDMI_transformer`  
适用对象：接手本项目的下一位 Codex/实验执行者

本文件是当前状态的执行交接，不替代旧报告，也不要求回退或覆盖已有结果。接手
后先阅读本文件，再根据需要阅读下面列出的结果报告。所有新增实验必须使用新的
`run-id` 和新的报告文件。

## 1. 当前结论

项目当前研究的是一个 decoder-only TinyLM：

```text
4 sink tokens + 124 recent local KV
        +
bounded semantic fact memory
        +
content-dependent write/retention/merge/read gates
        +
lexical value span + memory-to-token alignment
```

在 seed=17 的受控选择性记忆压力任务中，value-token 版本已经达到：

- 288/288 validation EM；
- 100% target survival；
- 100% Recall@1、Recall@4、MRR；
- 平均只使用 1 个 active user-memory slot；
- 8 条临时事实和 16 个 128-token filler segments 的最长验证条件仍成功。

这支持“门控能过滤临时事实”和“value-token 对齐能修复检索后词级恢复”的判断。
但该结论目前只有一个 seed、16 个颜色值和受控模板，不能外推为开放域记忆能力。

MoE 已实现为 Transformer FFN 的独立替换路径：每层 4 experts、top-2、dropless
token dispatch，局部 KV、事实记忆和 value-token 路径不变。当前 MoE 正式结果中
SWA-only 和 Fixed-LRU 已完成；旧 Gated+MoE 仍失败在 step 0，Value-token
Gated 的正式 run 已完成，但它是早先的 value-token（非 MoE）版本，不能把它误当
作已完成的 `Value-token Gated + MoE` 对照。

## 2. 必须保护的资产

不要执行 `git reset --hard` 或 `git checkout --`。工作区已有用户和实验改动。

不要覆盖以下内容：

- 原有 Dense 结果和报告；
- `/root/autodl-tmp/26summerBDMI_transformer/runs/` 下已有 checkpoint；
- 已失败 run 的 `summary.json`；
- 旧报告，例如 `docs/TINYLM_MOE_UNIFIED_EXPERIMENT_REPORT.md`；
- 代码快照目录。

MoE 修改前快照：

```text
code_snapshots/pre_moe_20260910_utc/
```

快照只包含预期会修改的 `config.py`、`model.py`、`train_memory_stress.py` 和
测试文件，另有 `README.md` 与 `SHA256SUMS`。value-token 修改前快照在：

```text
code_snapshots/pre_value_token_alignment_20260910_utc/
```

## 3. 代码现状

主要实现文件：

```text
adaptive_gated_fact_memory/src/adaptive_fact_memory/config.py
adaptive_gated_fact_memory/src/adaptive_fact_memory/model.py
adaptive_gated_fact_memory/src/adaptive_fact_memory/memory.py
adaptive_gated_fact_memory/src/adaptive_fact_memory/state.py
adaptive_gated_fact_memory/scripts/synthetic_memory.py
adaptive_gated_fact_memory/scripts/synthetic_memory_stress.py
adaptive_gated_fact_memory/scripts/train_memory_stress.py
adaptive_gated_fact_memory/tests/test_prototype.py
```

### 3.1 记忆路径

- `StreamingSinkSlidingAttention`：物理有界的 sink+recent local KV；
- `AtomicFactExtractor`：事实 span、write probability、semantic key/value；
- value-token 模式额外预测 value 起点/长度，保存 `lexical_values`、
  `value_payload_ids` 和 `value_payload_mask`；
- `FactMemoryController`：写入、合并、update、retention、eviction；
- `FactMemoryReader`：每个 user prompt 开始时选择 semantic top-k，assistant
  continuation 复用同一读取结果；
- `SharedMemoryFusion`：把 semantic value 与 `combined`、`semantic_lexical`、
  `lexical_only`、`payload_only` 或 `value_only` 路径融合到 decoder；
- `memory_to_token_logits`：位置条件的 lexical value 辅助 token head，使用
  tied token embedding；它只提供辅助监督，正式生成仍使用普通 LM head。

### 3.2 MoE 路径

`model.py` 中的 `SparseMoE`：

- 每个 Transformer block 独立拥有 4 个 experts；
- 每个 token 通过 router 选择 top-k experts；
- 正式实验使用 `moe_top_k=2`；
- padding token 不参与 dispatch 或 load 统计；
- dropless：不因 capacity 截断而丢 token；
- `moe_load_balance` 为 `E * sum(importance * load)`，top-2 完全均衡时理论值约
  为 2，而不是 1；
- `initialize_moe_from_dense()` 把父模型 Dense FFN 权重复制到所有 expert；
- 为保持旧 checkpoint 的 state-dict 兼容性，Dense `ffn_in/ffn_out` 仍保留，但
  在 MoE 模式初始化后被冻结且不参与 active computation；
- 训练 objective 以 `0.01 * moe_aux_loss` 加入所有 memory policy，避免把 MoE
  load-balance 差异混入 memory 条件比较。

当前实现是普通 PyTorch token indexing、逐 expert forward 和 `index_add_`，没有
fused grouped-GEMM/MegaBlocks。因此吞吐结果反映当前原型实现成本，不能外推到优化
后的 MoE kernel。

## 4. 测试与正确性

正确测试命令必须设置源码路径：

```bash
cd /root/BDMI/26summerBDMI_transformer
PYTHONPATH=adaptive_gated_fact_memory/src \
  python3 -m unittest discover \
  -s adaptive_gated_fact_memory/tests -p 'test_*.py' -v
```

当前结果：21/21 tests passed。

也可运行：

```bash
PYTHONPATH=adaptive_gated_fact_memory/src \
  python3 adaptive_gated_fact_memory/scripts/smoke_test.py
```

如果不设置 `PYTHONPATH`，测试会出现 `ModuleNotFoundError: adaptive_fact_memory`；
这不是代码回归。

MoE correctness 还应检查：

- logits/loss/gradients finite；
- future-invariance；
- 每层 expert 是否有 token；
- padding 是否被排除；
- top-k dispatch fractions；
- routing entropy；
- load-balance 数值接近 2（top-2 时）；
- 不要把“小 batch 中某 expert 未收到 token”误写成 token 丢失；正式 dropless
  训练的全局路由需单独检查。

## 5. 父模型、数据和固定协议

所有 selective stress run 从同一个 seed-17 TinyStories backbone 开始：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_tinystories/
tinystories_backbone_s17_10m_20260908/checkpoint.final.pt
```

父 checkpoint SHA-256：

```text
d995248587a5e535af59d850f36a5bc01925c54bd9dff213f0f18bfb26f2b78f
```

TinyStories filler：

```text
/root/autodl-tmp/26summerBDMI_transformer/data/cache/
tinystories_tinylm_v1/train_100000257.int32.bin
```

固定 selective stress 协议：

| 项目 | 设置 |
|---|---:|
| seed | 17 |
| TinyLM | 6 layers, hidden 384, heads 8, FFN 1536 |
| local KV | 4 sinks + 124 recent |
| user memory slots | 8 |
| reader | semantic top-k=4 |
| max write candidates | 1 |
| merge threshold | 0.999 |
| steps | 2000 |
| batch / accumulation | 2 / 2 |
| train delay | 1, 2, 4 segments |
| train noise | 0, 1, 2, 4 temporary facts |
| eval delay | 1, 4, 16 segments |
| eval noise | 0, 2, 4, 8 temporary facts |
| eval examples | 24 per cell，288 total |
| backbone LR | 3e-5 |
| memory/new-head LR | 3e-4 |
| optimizer | AdamW，betas=(0.9, 0.95)，weight decay=0.1 |
| precision | RTX 4090 BF16 autocast |
| checkpoint interval | 500 |

MoE 正式配置：

```text
--moe-experts 4
--moe-top-k 2
--moe-load-balance-weight 0.01
```

旧 Gated 和 Value-token Gated 的本次 MoE 对比应使用同一设置；Value-token Gated
还需：

```text
--memory-value-mode semantic_lexical
--value-token-alignment
```

## 6. 已完成 run 和结果位置

所有目录位于：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
```

### 6.1 MoE 已完成

| 条件 | run ID | 状态 | 训练时间 | EM | survival | Recall@1 | MRR | 说明 |
|---|---|---|---:|---:|---:|---:|---:|---|
| SWA-only + MoE | `selective_stress_moe_swa_only_s17_20260910` | ok | 1479.8 s | 5.21% | N/A | N/A | N/A | 无长期记忆 |
| Fixed-LRU + MoE | `selective_stress_moe_fixed_lru_s17_20260910` | ok | 1804.0 s | 65.28% | 93.06% | 93.06% | 0.9306 | 固定写入+LRU |
| 旧 Gated + MoE | `selective_stress_moe_gated_s17_20260910` | failed | 0.83 s | N/A | N/A | N/A | N/A | step 0 失败，不得覆盖 |

MoE 平均验证指标来自 `summary.json` 的 `evaluation.by_condition` 等权平均。
SWA-only 的 EM 是偶然词表命中水平；其 survival/recall 应写 N/A，不应写成记忆
机制的“0 分”。

MoE 训练资源：

| 条件 | 平均 examples/s | peak allocated |
|---|---:|---:|
| SWA-only + MoE | 5.406 | 3.197 GiB |
| Fixed-LRU + MoE | 4.435 | 2.971 GiB |

路由统计可从 `metrics.jsonl` 读取。两者末步 `moe_load_balance` 分别约为
2.013 和 2.031，均未显示明显的单 expert collapse；但目前 selective-stress
summary 没有自动保存完整的按 expert 聚合表，需要从每步 metrics 或重新评估脚本
聚合。

### 6.2 非 MoE value-token 主模型

```text
selective_stress_v3_gated_value_token_s17_20260910/
```

这是 `value-token Gated` 的非 MoE 结果：

- EM 100%；
- survival/Recall@1/Recall@4/MRR 均 100%；
- 平均 active slots=1；
- 训练时间 1380.1 s；
- 训练吞吐 5.797 examples/s；
- peak allocated 约 2.498 GiB；
- 已有 `diagnostics_checkpoint_sweep.json` 与 `inference_benchmark.json`。

它只能用于说明 value-token 机制本身，不能作为 `Value-token Gated + MoE` 行。

### 6.3 Dense selective stress 对照

```text
selective_stress_v2_swa_only_s17_20260909/
selective_stress_v2_fixed_lru_s17_20260909/
selective_stress_v2_gated_s17_20260909/
```

总体 dense 结果：

- SWA-only：EM 5.21%；
- Fixed-LRU：EM 74.65%，survival 92.01%，Recall@1 90.28%，MRR 0.9103；
- 旧 Gated：EM 52.43%，survival/Recall@1/Recall@4/MRR 100%。

可参考：

```text
docs/ADAPTIVE_GATED_MEMORY_STRESS_S17_RESULT_REPORT.md
ADAPTIVE_GATED_MEMORY_CURRENT_RESULTS_PAPER_REPORT.md
ADAPTIVE_GATED_MEMORY_VALUE_TOKEN_ALIGNMENT_S17_EXPERIMENT_REPORT.md
ADAPTIVE_GATED_MEMORY_VALUE_TO_TOKEN_DIAGNOSTIC_REPORT.md
```

## 7. 下一步：完成四条件 MoE selective-stress 比较

目标是得到四个严格同协议条件：

1. SWA-only + MoE，已完成；
2. Fixed-LRU + MoE，已完成；
3. 旧 Gated + MoE，需重跑；
4. Value-token Gated + MoE，需重跑。

### 7.1 旧 Gated + MoE 命令

先确认新目录不存在，然后执行：

```bash
cd /root/BDMI/26summerBDMI_transformer
PYTHONUNBUFFERED=1 \
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 adaptive_gated_fact_memory/scripts/train_memory_stress.py \
  --device cuda:0 \
  --run-id selective_stress_moe_gated_s17_20260910_retry1 \
  --runs-dir /root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress \
  --memory-policy gated \
  --seed 17 \
  --steps 2000 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --memory-slots 8 \
  --memory-read-top-k 4 \
  --moe-experts 4 \
  --moe-top-k 2 \
  --moe-load-balance-weight 0.01
```

### 7.2 Value-token Gated + MoE 命令

```bash
cd /root/BDMI/26summerBDMI_transformer
PYTHONUNBUFFERED=1 \
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 adaptive_gated_fact_memory/scripts/train_memory_stress.py \
  --device cuda:0 \
  --run-id selective_stress_moe_value_token_gated_s17_20260910_retry1 \
  --runs-dir /root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress \
  --memory-policy gated \
  --seed 17 \
  --steps 2000 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --memory-slots 8 \
  --memory-read-top-k 4 \
  --memory-value-mode semantic_lexical \
  --value-token-alignment \
  --moe-experts 4 \
  --moe-top-k 2 \
  --moe-load-balance-weight 0.01
```

预期单个 run 约 23–31 分钟，但应以实际 GPU 状态为准。当前只有一张可见 RTX
4090 时串行运行，不要在同一张卡上并发两个训练进程。训练状态按用户要求低频查询。

### 7.3 失败处理

如果旧 Gated 重跑仍出现错误，保留失败目录，使用 `retry2`，不要修改旧
`summary.json`。已定位到两个历史错误：

1. 旧代码在 autocast 下让 `FactMemoryReader` 的 query 为 FP32、keys 为 BF16，
   在 `einsum` 处触发 dtype mismatch；
2. 修复 dtype 后，MoE diagnostics 聚合把形状 `[layers, experts]` 当成
   `[experts]` 转成 Python scalar，在 `train_memory_stress.py:738` 附近触发：

```text
ValueError: only one element tensors can be converted to Python scalars
```

接手者需先修复这两个问题并跑 1–2 step smoke test，再跑正式训练。建议修复原则：

- 在 `FactMemoryReader.forward` 中让 query 和 normalized keys 使用一致 dtype；
- 在 stress loss 中先对 layer 维度求均值，例如：

```python
mean_token_fracs = torch.stack(token_fracs).float().mean(dim=0)
```

其中 `token_fracs` 里的每个 tensor 若为 `[layers, experts]`，应再对 layer 维度
求均值，得到 `[experts]` 后再逐 expert `float(...)`；
- 同样处理 `dispatch_fracs`；
- 保持 `moe_load_balance`、router loss 和总 loss 的定义不变；
- 修复后运行单元测试和 2-step smoke，再开始正式 run。

不要直接把错误改成忽略异常，也不要把 MoE diagnostics 从 loss 中删除。

## 8. 完成四条件后要写的新报告

请新建，不要覆盖旧报告：

```text
ADAPTIVE_GATED_MEMORY_MOE_COMPARISON_S17_REPORT.md
```

报告至少包含以下三种比较，不能混为一个结论：

1. 四个 Dense 条件：SWA-only、Fixed-LRU、旧 Gated、Value-token Gated；
2. 四个 MoE 条件：同样四类，全部 `4 experts/top-2`；
3. 同一条件 Dense → MoE 的变化。

每行至少记录：

- EM；
- teacher-forced answer NLL；
- Recall@1、Recall@4、MRR；
- target survival；
- mean active slots；
- mean noise accept rate；
- state bytes 和 logical active-slot bytes；
- 训练时间、训练 examples/s；
- peak allocated/reserved memory；
- 总参数量和 active parameters/token；
- expert token fraction、dispatch fraction；
- routing entropy；
- load-balance loss；
- 是否出现 expert collapse。

报告中必须明确：

- `SWA-only` 没有 target survival/recall，写 `N/A`；
- MoE 增加总参数容量，不等于降低 checkpoint/optimizer 存储；
- top-2 的 load-balance 理论平衡参考约为 2；
- 单 seed 只能写 screening，不写统计显著性；
- 当前 PyTorch dispatch 未优化，吞吐不是 MoE 理论上限；
- Value-token auxiliary head 不等于 pointer/copy head；
- memory slot 仍是固定预分配张量，logical active bytes 不等于物理显存已经下降。

## 9. 推荐阅读顺序

```text
docs/HANDOFF_NEXT_MODEL_20260910.md       # 本文
ADAPTIVE_GATED_MEMORY_CURRENT_RESULTS_PAPER_REPORT.md
ADAPTIVE_GATED_MEMORY_VALUE_TOKEN_ALIGNMENT_S17_EXPERIMENT_REPORT.md
docs/ADAPTIVE_GATED_MEMORY_STRESS_S17_RESULT_REPORT.md
docs/TINYLM_MOE_UNIFIED_EXPERIMENT_REPORT.md
adaptive_gated_fact_memory/DESIGN.md
adaptive_gated_fact_memory/README.md
```

如果只需要继续完成 MoE，不必重新阅读六模型 A/B/C 历史资料；它们属于另一条
TinyStories backbone benchmark 线。

## 10. 最终交接检查清单

- [ ] 工作区已有改动未被回退；
- [ ] 新修复前先保存或确认 `pre_moe_20260910_utc` 快照；
- [ ] `PYTHONPATH=adaptive_gated_fact_memory/src` 下 21/21 测试通过；
- [ ] 旧 Gated+MoE 用新 run ID，历史失败目录不覆盖；
- [ ] Value-token Gated+MoE 与旧 Gated+MoE 使用同一 MoE 配置；
- [ ] 单卡串行，训练日志和 checkpoint 放数据盘；
- [ ] 四个条件全部有 `config.resolved.json`、`metrics.jsonl`、`summary.json`；
- [ ] 失败 run 也保留并在报告中说明；
- [ ] 新报告另建文件，不修改旧结果文件；
- [ ] 完成后报告 Dense、MoE 和 paired Dense→MoE 三类比较。

