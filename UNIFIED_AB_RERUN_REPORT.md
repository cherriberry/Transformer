# A+B 四模型统一重训报告

更新时间：2026-09-06  
实验状态：完成（12 个正式 run：4 个模型 × `seed={17,29,43}`）

## 1. 结论摘要

已从角色 A 分享链接下载 Linformer/Performer 的源码，并将其 attention backend 接入角色 B 的 TinyStories/TinyLM 训练协议；随后在同一张 RTX 4090 上顺序完成四个模型、三个随机种子的 10M-token 重新训练。下表为三 seed 的均值 ± 标准差；显存和参数量是三次相同配置下的固定值。

| 方法 | 统一配置 | 完整 validation NLL（均值 ± std） | PPL（均值 ± std） | 训练吞吐 tok/s（均值 ± std） | 峰值已分配显存 | 参数量 |
|---|---|---:|---:|---:|---:|---:|
| Linformer | `lin_pool=128`, `lin_chunk=128` | 2.933816 ± 0.006033 | 18.799463 ± 0.113420 | 27,857.8 ± 106.0 | 2.580 GiB | 29,925,504 |
| Performer | `n_features=384`, redraw disabled | 3.450477 ± 0.008266 | 31.516144 ± 0.260635 | 27,079.0 ± 541.2 | 2.704 GiB | 29,925,504 |
| Longformer | left window=128 | 2.885517 ± 0.004599 | 17.912946 ± 0.082476 | 13,498.8 ± 268.5 | 6.799 GiB | 29,925,504 |
| Memformer | segment=128, slots=64 | 2.849148 ± 0.004651 | 17.273176 ± 0.080427 | 14,669.5 ± 300.5 | 2.633 GiB | 33,466,752 |

12 个 run 都完成了请求的 10,000,000 个训练 prediction 和 4,765,917 个 validation prediction，状态均为 `ok`，没有 OOM 或 NaN。标准差为三个 seed 间的 run-level 标准差，不代表置信区间。

在这次统一条件下，按 validation NLL/PPL 的均值排序为 Memformer、Longformer、Linformer、Performer；但 Memformer 比公共主干多 3,541,248 个参数（约 +11.83%）。Linformer/Performer 的计算路径来自 A 下载源码，Longformer/Memformer 来自本地 B 实现。

## 2. 统一协议

本次重训没有沿用 A 原实验的 `context=4096`、`effective_batch=32768`，而是采用 B/统一筛选协议，以便四种方法可以横向比较：

| 项目 | 值 |
|---|---|
| 数据集 | `roneneldan/TinyStories`，固定 parquet 顺序 |
| tokenizer | `openai-community/gpt2`，vocab=50,257，EOS=50,256 |
| packing | 故事末尾追加 EOS，跨文档连续 packing，block/context=512 |
| 训练预算 | 10,000,000 next-token predictions |
| validation | 全量 validation stream，4,765,917 predictions，token-weighted NLL |
| TinyLM | 6 layers / hidden 384 / 8 heads / FFN 1536 / RoPE 32768 |
| 正则化 | Pre-LN，GELU，dropout=0.1，tied embeddings，Linear 无 bias |
| 优化器 | AdamW，lr=3e-4，betas=(0.9,0.95)，weight decay=0.1 |
| 学习率 | 3% warmup，cosine，最低 3e-5 |
| 梯度 | global norm clip=1.0 |
| 计算精度 | BF16 autocast；参数保存为 FP32；TF32 关闭 |
| batch | micro batch=4 sequences，gradient accumulation=4，effective=8192 tokens |
| seed | `17, 29, 43`；三次 run 使用相同固定数据顺序，seed 改变初始化，并改变 Performer 的按 seed 固定随机特征投影 |

每个模型都从零初始化；公共参数采用参数名规范化后的确定性初始化。四种模型之间的公共参数最大差异为 0，方法专属参数没有被错误地从公共参数中抹去。

## 3. A 源码下载与适配

A 分享链接：

`https://cloud.tsinghua.edu.cn/d/ace43d623eca4a56a3f1/`

下载目录（数据盘）：

`/root/autodl-tmp/26summerBDMI_transformer/source_partA_linformer_performer/`

其中包含 A 的 `code/models.py`、训练/评估 notebook、原始结果表和下载清单。A 模型源码 SHA-256 为：

`d8aef33738476c8422cd14ae119d5303d67896dc34abdfd05ec5f476b8161f4b`

下载文件逐项大小和 SHA-256 见 [DOWNLOAD_MANIFEST.json](/root/autodl-tmp/26summerBDMI_transformer/source_partA_linformer_performer/DOWNLOAD_MANIFEST.json)。

适配代码见 [run_unified_ab.py](experiments/tinystories_tinylm_v1/run_unified_ab.py)。适配层保留了 A 的：

- causal chunked Linformer（`lin_pool=128`）；
- causal FAVOR+ Performer（`n_features=384`）；
- Q/K/V、RoPE 和 output projection 的计算路径。

适配层只负责把 A backend 接到 B 的 512-token TinyLM 主干和数据流；没有把 A 的旧 `context=4096` 结果伪装成此次统一结果。A 的旧结果仍保留在下载目录中，作为历史/附录证据。

## 4. 正确性检查

正式训练前运行了四方法 correctness 闸门：

| 检查 | 结果 |
|---|---:|
| 输出 shape 与 finite logits | 4/4 PASS |
| 有限梯度 | 4/4 PASS |
| future invariance | 4/4 PASS，最大差异 0 |
| Linformer 满窗等价 | PASS，最大差异 0 |
| Longformer 满窗与 SDPA 对照 | PASS；FP32 reduction-order 差异 `1.778e-4`，低于记录的 `5e-4` 数值容差 |
| Memformer 跨 segment history effect | PASS，非零 |

检查原始记录：

`/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/correctness.json`

## 5. 正式 run 产物

统一结果目录：

`/root/autodl-tmp/26summerBDMI_transformer/runs/unified_ab_rerun/`

12 个正式 run（每个方法各三个 seed）：

```text
rerun_unified_linformer_s17
rerun_unified_linformer_s29
rerun_unified_linformer_s43
rerun_unified_performer_s17
rerun_unified_performer_s29
rerun_unified_performer_s43
rerun_unified_longformer_s17
rerun_unified_longformer_s29
rerun_unified_longformer_s43
rerun_unified_memformer_s17
rerun_unified_memformer_s29
rerun_unified_memformer_s43
```

每个目录包含：

- `config.resolved.json`：方法配置、协议、A 源码 hash、环境和 config hash；
- `metrics.jsonl`：逐 optimizer step 的训练 NLL、probe validation、梯度和吞吐；
- `checkpoint.tokens_*.pt`：恢复点；`seed=17` 按约 1,048,576-token 间隔保存，`seed=29/43` 为节省数据盘空间仅保存 10M-token 最终恢复点；这不影响已完成的指标计算；
- `final.pt`：最终训练状态；
- `summary.json`：完整 validation、资源和状态汇总。

聚合产物：

- `/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/per_run.csv`
- `/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/summary.csv`
- `/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/summary.json`
- `/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/SUMMARY.md`

`summary.json` 同时保存了仓库数据 manifest 和本次 train/validation binary cache 的 SHA-256。

另有一个 `smoke_unified_linformer` 目录，仅用于训练代码冒烟检查（8,192 tokens），已从正式聚合表排除。

逐 run 的 NLL、PPL、吞吐、显存和状态见 `per_run.csv`；三 seed 聚合见 `summary.csv` 和 `SUMMARY.md`。

## 5.1 逐 run validation 结果

| 方法 | seed=17 NLL / PPL | seed=29 NLL / PPL | seed=43 NLL / PPL |
|---|---:|---:|---:|
| Linformer | 2.933785 / 18.7987 | 2.939865 / 18.9133 | 2.927799 / 18.6865 |
| Performer | 3.459063 / 31.7872 | 3.442574 / 31.2673 | 3.449795 / 31.4939 |
| Longformer | 2.883914 / 17.8841 | 2.890703 / 18.0060 | 2.881933 / 17.8487 |
| Memformer | 2.845559 / 17.2112 | 2.854402 / 17.3641 | 2.847481 / 17.2443 |

## 6. 结果解释边界

1. 这是统一协议下的三个 seed（17、29、43）结果，但仍没有进行超参数再搜索；因此它比单 seed screening 更稳健，却不是完整的超参数最优或论文级置信区间结论。
2. TinyStories 没有官方 test split；这里的 PPL 是 validation PPL，不能称为 test PPL。
3. A 原实验报告中的 `context=4096 / batch=32768 / vocab=50258` 与本次统一重训不同；A 旧结果不能直接和本表做严格主榜比较。
4. 这里的长序列能力没有被重新训练验证；本次质量结果只对应训练/评估 context=512。
5. Performer 的 384 个随机特征矩阵是 non-trainable buffer（各层合计约 442,464 bytes），不应计入 trainable parameter count。
6. Memformer 的固定 memory slots 是运行时状态；本次实现不是动态 active-slot 创新架构，不能把结果直接解释为用户提出的动态记忆方案已被验证。
7. 除 seed 外，所有 run 的数据、tokenizer、context、训练预算、effective batch、TinyLM 主干、优化器、学习率计划、精度、TF32 设置、GPU、validation stream 和 A 源码 hash 均一致。方法专属 attention 配置、参数量、显存和运行时状态是有意保留的架构差异，不应被“控制掉”。
8. 三个 seed 使用相同的固定训练 token 顺序（未做每 seed 的数据 shuffle）；因此 seed 方差主要反映初始化和 Performer 随机特征的变化，而不是数据顺序方差。
9. `seed=17` 与 `seed=29/43` 的 checkpoint 保存间隔不同，只影响中途恢复点数量，不影响最终 10M-token 训练和 validation 指标。若后续要比较断点恢复或训练曲线存档，应统一 checkpoint 间隔后重新运行。
10. `seed=17` 的 config 文件是在补充 cache 审计字段前生成的，因此没有内嵌 `train_cache`/`validation_cache` 元数据；逐字段核对显示它与后两个 seed 的训练配置仅在 seed、run id、checkpoint 间隔和这些记录字段上不同，三次 run 实际读取同一份固定 cache。若需要论文级逐 run 代码哈希审计，下一轮应先冻结 runner 并为每个 run 写入 runner hash。

## 7. 复现命令

```bash
cd /root/BDMI/26summerBDMI_transformer

# correctness
python experiments/tinystories_tinylm_v1/run_unified_ab.py correctness --device cuda:0

# 例如重新运行 Linformer
python experiments/tinystories_tinylm_v1/run_unified_ab.py train \
  --method linformer --method-value 128 \
  --run-id rerun_unified_linformer_s29 \
  --seed 29 --train-tokens 10000000 \
  --full-validation --save-checkpoint --device cuda:0

# 聚合已有正式 run
python experiments/tinystories_tinylm_v1/run_unified_ab.py aggregate
```

当前工作区的原始统一记录 [SIX_MODEL_UNIFIED_COMPARISON_RECORD.md](SIX_MODEL_UNIFIED_COMPARISON_RECORD.md) 未被改写；本报告和新 runner 是独立新增的重训记录。
