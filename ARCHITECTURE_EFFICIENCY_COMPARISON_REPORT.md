# 不同高效 Transformer 架构的效率、优缺点与设计来源

**实验报告（基于当前目录已有结果）**  
审阅日期：2026-09-03  
项目目录：`26summerBDMI_transformer/`

## 摘要

本报告检查了当前目录中 Standard Attention、Linformer、Longformer、Performer、Reformer、Memformer、Keyformer 和 xFormers/SDPA 的代码、配置、原始 JSON、图表及已有实验报告，目标是回答三个问题：

1. 不同架构的效率优势来自哪一层设计；
2. 这些设计带来了什么优点和代价；
3. 在现有证据范围内，哪些结论可以横向比较，哪些不能。

结论可以概括为：没有脱离场景的“最快架构”，因为各方法优化的对象不同。Standard 是短序列和精确语义的可靠基线；xFormers/SDPA 保持注意力数学定义不变，主要通过 kernel 融合和分块降低显存，因此最适合作为通用工程默认值；Longformer 通过局部稀疏连接在足够长的序列上降低二次存储；Memformer 把历史压缩成固定容量状态，显存最稳定但存在信息瓶颈；Linformer 和 Performer 通过低秩/随机特征近似换取近似线性复杂度；Reformer 通过 LSH 稀疏化候选连接，理论上适合超长序列，但哈希和重排常数开销较大；Keyformer 专门压缩自回归解码的 KV cache，在真实 GPT-2 Medium 上显示出最清晰的质量—显存权衡。

最可信的定量结果是：在 RTX 4070 Laptop GPU、BF16、hidden=384、8 heads 的结构化长序列测试中，序列长度 4096 时 Longformer 的 median latency 为 10.76 ms、peak allocated 为 78.6 MiB，相比 Standard 的 21.95 ms 和 1333.5 MiB，约 **2.04× 加速、94.1% 显存节省**；Memformer（segment=256、slots=64）为 13.88 ms、37.0 MiB，约 **1.58× 加速、97.2% 显存节省**。在 RTX 5060 Laptop GPU 上的 GPT-2 Medium 实验中，Keyformer 保留 75%/50% KV 时，PPL 仅从 27.903 增至 28.111/28.697（+0.75%/+2.84%），物理 KV 容量按预算近似线性下降；prompt=768、保留 50% 时解码约 **1.44×** 于 FullKV。

与此同时，Linformer/Longformer/Performer/Reformer/xFormers 的 WikiText TinyLM 结果只训练约 120 步，且实现、mask、初始化和硬件并不完全统一；这些数字适合做机制和可运行性证据，不足以组成严格排行榜。报告因此把结果分为 `matched_tinylm`、`paper_aligned`、`innovation_validation` 和历史 artifact，而不把异构实验直接混合排名。

## 1. 实验对象与“效率来源”

标准自注意力对长度为 `N`、隐藏维度为 `d` 的序列形成 `N x N` 分数矩阵，主要代价约为 `O(N^2 d)` 计算和 `O(N^2)` 中间存储。不同方法减少的对象不同：

| 方法 | 主要优化对象 | 核心设计 | 理论资源形态 | 是否改变注意力语义 |
|---|---|---|---|---|
| Standard | 无，作为基线 | 全 token 两两注意力 | 时间/中间存储约 `O(N^2)` | 否 |
| Linformer | K/V 的序列维 | 学习低秩投影到 `k` 个位置 | 约 `O(Nkd)` | 是，低秩近似 |
| Longformer | 注意力连接图 | 滑动窗口 + 少量 global token | 约 `O(NW+NG)` | 是，结构化稀疏 |
| Performer | softmax kernel 的计算形式 | FAVOR+ 随机特征映射，特征数 `r` | 约 `O(Nrd)` | 是，核近似 |
| Reformer | 候选 token 集合与训练激活 | LSH 分桶；可配 reversible/chunking | 常表述为约 `O(N log N)` | 是，哈希近似 |
| Memformer | 跨 segment 的历史表示 | 固定数量 memory slots 递归传递 | 状态约 `O(Md)`，当前段另有工作集 | 是，历史摘要 |
| Keyformer | 自回归历史 KV cache | 重要性分数 + recent window，物理裁剪 K/V | cache 约按保留比例线性下降 | 是，丢弃部分历史 |
| xFormers/SDPA | GPU 计算和显存访问 | fused/分块/Flash 风格精确 kernel | 数学复杂度仍为二次，但不物化完整矩阵 | 通常否 |

这里的 `W`、`G`、`k`、`r`、`M` 不是同一种“预算”：分别表示窗口、全局 token 数、低秩维度、随机特征数和 memory slots，不能用参数数值直接横比。跨架构应比较实测 latency、吞吐、peak memory、OOM 边界以及相同质量约束下的成本。

## 2. 证据与实验口径

### 2.1 结果分层

| 证据层 | 主要文件 | 可支持的结论 |
|---|---|---|
| 结构化长序列矩阵 | `results/b_role/efficiency.json`、`results/b_role/ablations.json` | Longformer/Memformer 在同机、同 dtype、同主干下的长度趋势和消融 |
| Keyformer 深度实验 | `keyformer/results/perf_final.json`、`keyformer/results/quality_final.json` | 真实 GPT-2 Medium 的 KV 压缩、PPL、解码成本和基线差异 |
| 论文配置对齐检查 | `results/paper_comparison_b/*.json` | 保留论文核心机制时的 attention-level 趋势，不是完整论文复现 |
| 注意力保真度 | 各方法 `*/results/quality_wikitext.json` | 单层/固定投影下的输出差异；仅语义一致时可称 fidelity |
| TinyLM 质量 | 各方法 `quality_wikitext.json` | 短步数可训练性和初步质量 proxy，不能作为严格架构排名 |
| 历史脚本/图表 | 各方法 `benchmark_perf.py`、`figures/*.png` | 趋势探索；部分使用 RSS 或缺少完整环境元数据，不与结构化矩阵合并 |

### 2.2 统一程度与限制

结构化 B 组主要配置为 hidden=384、8 heads、BF16、batch=1、causal、warmup=2、每点 3 次，长度 128–4096；设备为 RTX 4070 Laptop GPU，PyTorch 2.11.0、CUDA runtime 13.0。Keyformer 使用 GPT-2 Medium（355M）、FP16、RTX 5060 Laptop GPU，prompt 256/512/768、生成 128 token、seeds=0/1/2。两块实验不能用绝对毫秒直接互相排名，只能分别与同机 Standard 比较。

现有 TinyLM 结果通常是 hidden=256、2 层、8 heads、长度 128、约 120 steps、15 个评估窗口；部分标准路径和高效路径的 causal mask 不一致，且多为单次训练。因此 TinyLM PPL 只作为补充，不覆盖长序列效率结论。

### 2.3 2026-09-03 新增长上下文参数筛选

`experiments/long_context_10m_v1/` 是当前目录最新的一批产物。6 个 run 完成约 203 万训练 token；Longformer window=1024 的 run 在 512 token 后因时间门限中断并标记 `needs_rerun`，window=2048 未运行。由于环境缺少 PG-19 依赖，这批实验实际使用 `synthetic_copy_stream_fallback`、seq_len=512、vocab=64 和 learned absolute positions，而不是计划中的 PG-19、seq_len=4096、GPT-2 vocab 和 RoPE。

它仍然提供了实现成本证据：Longformer left window 从 128 增至 512 时，训练吞吐从 2788 降至 775 token/s，peak allocated 从 1533 增至 5738 MiB，说明窗口变大会显著放大局部候选集合与中间张量；Memformer segment=512 时 slots=32/64/128 的吞吐分别约 10211/9766/19202 token/s、peak allocated 为 347/268/347 MiB。后者的非单调速度来自单 seed、系统噪声或 kernel 路径，不能据此宣称 slots=128 必然更快。该批结果只作为 parameter-screening/implementation smoke，不用于自然语言质量排名。

## 3. 横向效率结果

### 3.1 Standard、Longformer、Memformer：同机长序列矩阵

下表为 `results/b_role/efficiency.json` 的 median latency 和 CUDA peak allocated memory：

| N | Standard (ms / MiB) | Longformer (ms / MiB) | Memformer (ms / MiB) |
|---:|---:|---:|---:|
| 128 | 0.372 / 23.4 | 1.401 / 35.7 | 0.855 / 25.6 |
| 256 | 0.445 / 29.9 | 1.364 / 51.9 | 0.905 / 31.4 |
| 512 | 0.382 / 46.0 | 2.291 / 54.3 | 1.569 / 31.7 |
| 1024 | 1.103 / 108.4 | 3.273 / 57.8 | 3.547 / 32.5 |
| 2048 | 5.522 / 355.5 | 8.514 / 65.5 | 6.528 / 34.7 |
| 4096 | 21.951 / 1333.5 | 10.758 / 78.6 | 13.877 / 37.0 |

观察：

- **短序列不是高效架构的优势区间。** N≤512 时 Standard 最快，因为 dense GPU kernel 的常数项小，而滑窗索引、segment 拼接和额外投影的开销占主导。
- **Longformer 的收益在 N=4096 才明显。** 相对 Standard，延迟约降低 51%，显存约降低 94%。其显存随窗口 `W` 增长，而不是完全与 N 无关。
- **Memformer 的状态容量使显存最稳定。** N=4096 时仅 37.0 MiB；但 segment 更新、memory fusion 和状态投影使它不一定比 Longformer 快。
- **同一方法的实现决定实际收益。** 结构化矩阵使用了 query chunking；旧的 Python 循环版 Longformer 图中曾出现比 Standard 更慢的情况，说明理论稀疏并不会自动变成 kernel 级加速。

### 3.2 论文参数对齐的长度趋势

Longformer 的 d=512、8 heads、causal 对齐检查（`results/paper_comparison_b/longformer_paper_aligned.json`）显示：

| N | Full reference (ms / MiB) | Local-only (ms / MiB) | Local+global (ms / MiB) |
|---:|---:|---:|---:|
| 2048 | 6.329 / 347.1 | 25.656 / 169.8 | 26.780 / 177.5 |
| 4096 | 22.044 / 1327.1 | 48.748 / 185.9 | 51.386 / 202.3 |
| 8192 | 87.007 / 5231.1 | 97.201 / 218.0 | 103.398 / 249.5 |

到 N=8192 时，local-only 的延迟已接近 full reference，但显存节省约 95.8%；global token 增加了约 6–10% 的延迟和额外显存。该结果验证的是 `O(NW)` 存储趋势，不是 Longformer 论文的 text8/enwik8 五阶段训练复现。

### 3.3 统一微基准中的 Performer、Reformer、Memformer

根目录 `benchmark_results.json` 在 RTX 4070 Laptop GPU、hidden=256、8 heads、N≤512 下给出：

| 方法 | N=128 ms / MiB | N=256 ms / MiB | N=512 ms / MiB |
|---|---:|---:|---:|
| Standard | 0.378 / 11.9 | 0.368 / 15.6 | 0.664 / 29.1 |
| Memformer adapter | 0.789 / 14.3 | 0.623 / 18.8 | 1.060 / 33.8 |
| Performer | 1.254 / 16.8 | 1.116 / 21.4 | 2.328 / 30.7 |
| Reformer | 1.943 / 17.4 | 2.013 / 22.9 | 2.317 / 34.0 |

在这个短序列范围内 Standard 始终最快或显存最低。这不是对线性/对数渐近复杂度的否定，而是说明随机特征、LSH、memory 拼接等方法的常数项尚未被长序列收益抵消。该微基准只测注意力模块，不代表完整模型的训练吞吐或质量。

## 4. 质量与保真度结果

### 4.1 单层注意力输出保真度

WikiText-2 embedding、seq_len=128 的结果如下：

| 方法/模式 | Relative L2 | Cosine similarity | 正确解释 |
|---|---:|---:|---|
| xFormers/efficient | 3.78e-7 | 0.99999997 | 精确 kernel，与 full attention 数值等价 |
| Performer (FAVOR+) | 0.00223 | 0.9999975 | 当前随机特征设置下近似很接近 |
| Longformer local, W=64 | 0.725 | 0.809 | 远距连接被结构性删除，输出显著变化 |
| Linformer random projection, k=64 | 1.152 | 0.201 | 未训练的随机低秩投影不能代表 learned Linformer |
| Memformer-like | 1.374 | 0.013 | 这是 memory augmentation 的效果，不是 full-attention 近似误差 |
| Reformer LSH | 1.519 | -0.098 | 当前模块投影/输出权重未共享，不能归因于 LSH 本身 |

因此，xFormers 的优势来自实现层，不牺牲注意力语义；Longformer/Linformer/Performer/Reformer 的输出变化来自算法近似或稀疏化，必须通过训练适应；Memformer 的目标是跨段状态建模，不能用单层 MSE 排名。

### 4.2 TinyLM WikiText-2 结果（补充证据）

各目录保存的单次、短步数结果：

| 方法 | Standard PPL | Efficient PPL | 变化 |
|---|---:|---:|---:|
| Linformer | 29.682 | 27.974 | -5.8% |
| xFormers/SDPA | 29.405 | 28.170 | -4.2% |
| Memformer-like | 31.231 | 27.805 | -11.0% |
| Performer | 30.224 | 32.127 | +6.3% |
| Reformer | 29.076 | 30.720 | +5.7% |

这些 PPL 不能解释为架构优劣：训练仅约 120 steps、评估窗口仅 15 个，参数和初始化不完全匹配；更重要的是部分脚本存在 causal mask 不一致（例如 xFormers efficient 路径与 Standard 路径设置不同，Performer/Reformer 的 full 路径也未总是使用 causal mask）。Longformer 的 `allenai/longformer-base-4096` MLM 结果 PPL=8.514 是预训练 bidirectional 模型的 masked-token 指标，也不能与上述 causal TinyLM PPL 横比。

作为较严格的机制补充，`results/paper_comparison_b/causal_pair.json` 在相同合成 copy-stream、4 层/hidden=256、3 seeds 下得到 mean PPL：Standard causal 23.498、Longformer causal 22.878、Memformer causal 23.382；但训练仅 8 steps，结论仍限定于该合成任务。

## 5. Keyformer：自回归 KV cache 的独立效率结果

Keyformer 不改变训练阶段的 backbone attention，而是在解码时为每层历史 K/V 打分，保留 recent window 和高分旧 token，并**物理 gather** 被保留的 K/V。它优化的是 inference cache，不应和 Longformer 等训练/编码架构混成同一个“注意力复杂度”排名。

### 5.1 质量—缓存预算

GPT-2 Medium、WikiText-2 test、context=512、2044 个 token、3 seeds：

| 策略 | Cache ratio | PPL | 相对 FullKV | Top-1 |
|---|---:|---:|---:|---:|
| FullKV | 100% | 27.903 | 基线 | 41.14% |
| Keyformer | 75% | 28.111 | +0.75% | 40.93% |
| Keyformer | 50% | 28.697 | +2.84% | 40.64% |
| Keyformer | 25% | 32.822 | +17.6% | 38.96% |
| Random+Recent | 75% | 30.709 | +10.1% | 约39.8% |
| Random+Recent | 50% | 129.946 | +365.7% | 28.60% |
| RecentWindow | 75% | 76.660 | +174.7% | 33.37% |
| RecentWindow | 50% | 530.219 | +1800% | 21.62% |

在相同 cache budget 下，Keyformer 明显优于“只保留最近 token”或“随机保留旧 token”。这说明其优势来源不是单纯减少 token 数，而是**基于注意力重要性的内容选择**，同时以 recent window 防止局部连续性丢失。50%–75% 是当前模型和任务下最实用的折中区间。

### 5.2 物理容量与解码速度

总长度为 prompt+128 decode 时，KV tensor 大小几乎严格随预算变化：

| Prompt | FullKV | 75% | 50% | 25% |
|---:|---:|---:|---:|---:|
| 256 | 36 MiB | 27 MiB | 18 MiB | 9 MiB |
| 512 | 60 MiB | 45 MiB | 30 MiB | 15 MiB |
| 768 | 84 MiB | 63 MiB | 42 MiB | 21 MiB |

完整 decode（包含打分、选择、gather、attention 和 cache update）显示：prompt=256 时选择开销占主导，Keyformer 50% 约 43.7 ms/token，慢于 FullKV 22.1 ms/token；prompt=512、50% 时约 35.8 对 36.8 ms/token，接近持平；prompt=768、50% 时约 44.4 对 63.7 ms/token，约 1.44×。因此 Keyformer 的速度优势来源于“历史 attention 成本足够大后，减少 K/V 长度超过选择开销”，而不是任何长度下都更快。

## 6. 各架构优缺点及其设计来源

### 6.1 Linformer：低秩投影

**优点：**将 K/V 沿序列维投影到固定 `k`，长序列下计算和存储增长较慢；矩阵运算规则、易于批量化。  
**缺点：**低秩假设不适用于所有输入和层；`k` 过小会丢失细粒度或多峰依赖；投影通常需要学习，未经训练的随机投影会产生很大误差（当前 fidelity cosine=0.201）。固定长度/固定 rank 的部署灵活性有限。  
**优缺点来源：**把完整 token 维压缩成 `k` 个潜在基，收益来自降秩，代价来自不可逆信息压缩。  
**适用：**长度分布较稳定、可以充分训练 learned projection、且需要规则矩阵 kernel 的任务。

### 6.2 Longformer：局部窗口 + global token

**优点：**局部依赖用 `W` 个邻居处理，attention 中间存储从 `N^2` 降为约 `NW`；global token 提供跨文档汇聚通道；窗口和 global 数量可解释、可消融。  
**缺点：**窗口之外的 token 不能直接交互，远程信息需要多层传播或 global 路径；global token 数过多会增加成本；Python 索引/窗口展开若未融合，短序列可能比 dense attention 慢。  
**优缺点来源：**显式稀疏化连接图。窗口越小越省资源但感受野越窄；global 越多越能汇聚远程信息但越接近全局成本。  
**适用：**文档中局部连续性强、只需少量全局锚点的长序列；本实验在 N=4096 后效果最明显。

### 6.3 Performer：FAVOR+ 随机特征

**优点：**用特征映射重排 softmax kernel，可采用 prefix-sum 处理 causal attention，理论上对长度近似线性；当前单层 64 features 的相对 L2 仅 0.00223。  
**缺点：**存在随机估计方差，feature 数 `r` 增大才更稳定但也更慢；数值稳定、正值特征和归一化实现很重要；在短序列 GPU 上随机特征变换的常数项可能超过收益，TinyLM 单次结果 PPL 反而升高。  
**优缺点来源：**用有限维随机特征近似指数核，收益来自避免显式 `N x N` 矩阵，代价来自近似噪声和额外特征投影。  
**适用：**长度很长、允许可控近似误差，并能通过 feature 数和多 seed 调节稳定性的场景。

### 6.4 Reformer：LSH 注意力与可逆层

**优点：**相似 query/key 通过哈希进入同一 bucket，只在候选桶内注意；可结合 reversible residual 和 chunking 降低训练激活显存。  
**缺点：**哈希碰撞会漏掉相关 token，多轮 hashing 才能提高相遇概率；排序、bucket、padding 和不规则内存访问带来较大常数开销；bucket size、n_hashes、位置编码和 causal 细节耦合。当前 N≤512 微基准最慢，LSH fidelity 也不能直接解释为模型质量。  
**优缺点来源：**用数据依赖的哈希近似稀疏连接，并用可逆计算换激活显存；收益依赖“相关 token 能否被分到一起”。  
**适用：**超长序列、依赖具有可哈希相似性、且训练显存比单步 latency 更重要的场景。

### 6.5 Memformer：固定容量递归记忆

**优点：**将历史 segment 压缩为 `M x d` 的状态，状态大小与总历史长度无关；适合流式/分段处理；正确性测试显示跨 segment state delta 非零，确实存在持久状态路径。  
**缺点：**memory slots 是信息瓶颈，旧信息可能被覆盖或遗忘；segment 越短更新越频繁但开销更高，segment 越长效率更好但刷新不及时；必须正确 reset，detach 还会切断跨段梯度。  
**优缺点来源：**把“保存全部历史 token”改为“保存可学习摘要状态”；收益来自固定状态容量，代价来自摘要不可逆和 recurrent 优化难度。  
**适用：**在线文档、对话流和可以接受压缩历史的超长任务；严格显存预算下尤其有吸引力。

### 6.6 Keyformer：重要 token 的 KV cache 选择

**优点：**不需要重训 GPT-2 backbone；在相同预算下质量远好于 RecentWindow/Random+Recent；物理 KV bytes 按比例下降；中长 prompt 上可降低 decode latency。  
**缺点：**只解决自回归 decode，不降低训练或 prefill 的标准 attention；每步计算分数、采样/排序和 gather，短 prompt 可能更慢；25% cache 已出现明显 PPL 损失；需要保留原始绝对位置，不能简单重编号。  
**优缺点来源：**利用历史 token 重要性分布具有非均匀性，只保留高价值旧 token；收益来自 cache 长度减少，代价来自选择误差和动态控制开销。  
**适用：**decoder-only 在线生成、上下文较长、显存或 decode 成本是瓶颈的场景；建议从 50%–75% budget 开始。

### 6.7 xFormers/SDPA：精确 kernel 优化

**优点：**保持 full attention 语义，当前 fidelity cosine=0.99999997；融合 QK、softmax、加权 V 或采用 Flash 风格分块，减少 HBM 读写和中间矩阵，通常无需修改模型结构。  
**缺点：**数学上的 `O(N^2)` 计算并未消失；收益依赖 GPU、head dimension、dtype 和实际 backend；xFormers 不可用或 SM 不兼容时可能回退到 SDPA/普通实现，`available` 不等于实际调用了同一个 kernel。  
**优缺点来源：**优化的是 kernel 和内存层级，而不是连接图或模型表示；因此质量稳定但长序列渐近上限仍受二次计算约束。  
**适用：**短中序列、要求精确语义和最小改动的通用训练/推理；应记录 requested backend 与 actual backend。

## 7. 综合判断：如何选择架构

| 使用条件 | 首选 | 原因 | 需要警惕 |
|---|---|---|---|
| N≤1K、要求精确、硬件支持 Flash/SDPA | xFormers/SDPA | 语义不变、常数小、工程成熟 | backend 回退和 dtype 支持 |
| N≈2K–16K、局部依赖明显 | Longformer | `NW` 稀疏存储，global token 可补远程汇聚 | 窗口/全局 token 与 kernel 实现 |
| 流式 segment、显存必须固定 | Memformer | `M x d` 状态与总历史长度无关 | memory 容量、reset、detach、遗忘 |
| decoder 生成、KV cache 占主导 | Keyformer | 50%–75% cache 时质量损失小 | 选择开销、短 prompt、绝对位置 |
| 超长且可接受近似 | Performer | 近似线性，causal prefix-sum | feature 方差、数值稳定和调参 |
| 依赖可哈希聚类、训练激活紧张 | Reformer | LSH + reversible/chunking | hash collision、重排和实现复杂度 |
| 长度固定、可充分训练投影 | Linformer | 规则低秩矩阵，潜在高吞吐 | rank 选择和低秩信息损失 |

因此，推荐使用“分阶段/组合式”策略，而不是寻找单一总冠军：用 SDPA/Flash 处理短序列和局部 dense 子问题；需要超长编码时采用 Longformer 或 Memformer；decoder 推理再叠加 Keyformer 的 cache 控制。若质量约束严格，优先选择精确 kernel 或较温和的 cache ratio；若显存约束最严格，接受可控的信息摘要或近似误差。

## 8. 当前结果的主要局限

1. **硬件和软件不统一。** B 组使用 RTX 4070 Laptop GPU，Keyformer 使用 RTX 5060 Laptop GPU；绝对毫秒不可跨机排名。
2. **任务语义不同。** Longformer 有 causal attention-level 结果和 bidirectional MLM smoke test；Memformer 既有 causal synthetic pilot，也有 bidirectional BART attention 对齐；不能直接比较 PPL。
3. **TinyLM 训练过短。** 120 steps/15 windows 或 8–12 steps 的合成 pilot 只能证明可运行，不能替代充分训练。
4. **部分结果是不同信息通路。** Memformer 的“effect”、Reformer 未共享权重的 LSH 对照、随机 Linformer 投影均不应当标成 full-attention fidelity。
5. **历史 benchmark 测量不稳定。** 一些脚本使用进程 RSS 差分，曾出现负值或非单调跳变；应以 CUDA peak allocated/reserved 和 OOM 边界为准。
6. **xFormers 后端可能回退。** 需要在正式实验中记录实际执行的 kernel，而不能仅依据安装成功判断。
7. **Keyformer 上下文受 GPT-2 绝对位置限制。** GPT-2 Medium 的 `n_positions=1024` 使 prompt=1024 加生成 128 不可行；当前最终矩阵改用 prompt 256/512/768。
8. **最新长上下文筛选不是正式 PG-19 实验。** 数据、长度、词表和位置编码均发生协议降级，且有 1 个 interrupted、1 个 not-run 配置；其接近 1.0 的训练 PPL 反映合成 copy-stream 很容易拟合，不代表自然语言建模质量。

## 9. 建议的后续统一实验

如果要把当前集合升级为论文级横向比较，建议冻结一套 `matched_tinylm` 协议：相同 tokenizer、6 层/hidden=384/8 heads/FFN=1536、RoPE、causal mask、BF16、AdamW、相同实际训练 token 数和 seeds=17/29/43；长度统一为 128、512、1K、2K、4K、8K，并记录 median/p90 latency、tokens/s、peak allocated/reserved、OOM 和 token-weighted PPL。各方法只改变自身架构旋钮：Longformer window/global、Performer features、Reformer bucket/hashes、Linformer rank、Memformer segment/slots、Keyformer cache ratio。报告时分开 `paper_aligned`、`matched_tinylm` 和 `innovation_validation`，再用相同质量或显存预算绘制 Pareto 前沿。

## 10. 可复现材料

- 总览与协议：[`README.md`](README.md)、[`UNIFIED_EXPERIMENT_PROTOCOL.md`](UNIFIED_EXPERIMENT_PROTOCOL.md)
- 结构化 Longformer/Memformer 报告：[`B_LONGFORMER_MEMFORMER_EXPERIMENT_REPORT.md`](B_LONGFORMER_MEMFORMER_EXPERIMENT_REPORT.md)
- 论文条件对齐：[`B_PAPER_COMPARISON_REPORT.md`](B_PAPER_COMPARISON_REPORT.md)
- Keyformer 深度报告：[`KEYFORMER_REPORT.md`](KEYFORMER_REPORT.md)
- 统一微基准原始数据：[`benchmark_results.json`](benchmark_results.json)
- Longformer/Memformer 长序列数据：[`results/b_role/efficiency.json`](results/b_role/efficiency.json)、[`results/b_role/ablations.json`](results/b_role/ablations.json)
- Keyformer 性能与质量：[`keyformer/results/perf_final.json`](keyformer/results/perf_final.json)、[`keyformer/results/quality_final.json`](keyformer/results/quality_final.json)
- 最新长上下文筛选：[`experiments/long_context_10m_v1/aggregate/pilot_summary.json`](experiments/long_context_10m_v1/aggregate/pilot_summary.json)、[`experiments/long_context_10m_v1/jobs/jobs.csv`](experiments/long_context_10m_v1/jobs/jobs.csv)
- 各方法质量 JSON：`linformer/results/quality_wikitext.json`、`longformer/results/quality_wikitext.json`、`performer/results/quality_wikitext.json`、`reformer/results/quality_wikitext.json`、`memformer/results/quality_wikitext.json`、`xformer/results/quality_wikitext.json`

**最终结论：**本目录的实验支持“不同架构通过不同设计维度换取效率”的判断，而不支持无条件的总排名。若以当前证据选择：精确和通用工程优先 xFormers/SDPA；长序列局部稀疏优先 Longformer；固定状态和流式历史优先 Memformer；自回归 KV 显存优先 Keyformer；可接受近似且追求线性长度扩展时再考虑 Performer、Linformer 或 Reformer，并必须用充分训练和统一协议验证其质量代价。
