## 3. 当前 TinyLM 实验为什么需要重做

### 3.1 问题一：有些模型可能看到了未来答案

语言模型预测第 \(t\) 个位置时，只能看到第 \(t\) 个位置以前的内容。这叫 causal mask，可以想象成考试时用纸遮住后面的答案。

当前代码中：

- xFormers 的 Standard 路径设置了 `is_causal=True`；
- xFormers efficient 路径却设置了 `is_causal=False`；
- 若干 Standard reference 调用了没有 causal mask 的 full attention；
- 某些 efficient 路径又是 causal。

这相当于有些学生考试时能看到后面的答案，有些不能。此时比较 PPL 没有公平性。

### 3.2 问题二：比较模型没有从同一个起点出发

当前程序分别创建 Standard TinyLM 和某个 efficient TinyLM。创建第一个模型会消耗一批随机数，创建第二个又会消耗另一批随机数，因此：

- embedding 初始值不同；
- MLP 初始值不同；
- LM head 初始值不同；
- 不只是 attention 不同。

这像比较两辆发动机时，连轮胎、车身重量和驾驶员都随机换了。短训练下，初始化差异足以造成几 PPL 的波动。

正确做法是先制造同一个基础模型，再复制公共参数，只替换必须不同的 attention 部分。

### 3.3 问题三：训练集和测试集没有分开

当前 `load_wikitext_tokens` 直接加载 WikiText-2 的 `test` split。随后 `train_and_eval_ppl` 从这批 token 中随机抽窗口训练，又从同一批 token 的前部窗口评估。

这相当于用考试卷本身复习，再拿同一份考试卷评分。模型可能只是记住了这些内容，结果不能代表它对未见文本的泛化能力。

正确做法是：

- `train` split 只用于训练；
- `validation` split 用于选择配置和观察是否过拟合；
- `test` split 只在配置确定后做最终评分。

### 3.4 问题四：训练太短、数据太少

保存的 TinyLM 结果通常只有 120 steps、约 2048 tokens、15 个评估窗口。这个规模足以验证程序能运行，但不足以判断模型是否真正收敛。

如果 loss 还在快速下降，那么最后一步的 PPL 主要反映“谁在这 120 步学得快”，未必反映最终能力。

### 3.5 问题五：只有一次实验

神经网络训练具有随机性。只跑一个 seed，可能碰巧得到一个较好或较差的起点。正式比较至少需要 3 个 seed，然后报告平均值和波动范围。

### 3.6 问题六：模型参数量可能不同

不同 attention 内部包含的投影矩阵、随机特征或其他参数不一样。若一个模型参数更多，它的质量或速度变化可能来自模型大小，而不是架构设计。

需要同时报告参数量，并明确采用：

- 主干固定，只替换 attention；或
- 调整配置，让各模型总参数量接近。

## 4. 整个后续工作的通俗路线图

后续实验可以理解为盖房子：

1. **P0：打地基**——修复不公平和不可复现的问题；
2. **P1：盖主体**——得到一套可信的速度、显存和质量主结果；
3. **P2：装修并研究结构**——逐个改变架构参数，弄明白为什么好或不好；
4. **P3：压力测试**——换现代模型、更长文本和不同 GPU；
5. **整理交付**——自动画图、更新报告，让别人能复现。

最重要的原则是：**P0 没完成，不要急着跑大量 GPU 实验。** 否则花了很多时间后，可能发现结果因为偷看未来、数据泄漏或初始化不同而必须全部作废。

## 5. P0：先把实验变公平

## 第 0.1 步：写下一份统一比赛规则

**为什么做**

如果每个文件各自决定模型大小、长度、精度和指标，最后无法汇总。

**具体做什么**

新建 `EXPERIMENT_PROTOCOL.md`，明确写下：

1. 哪些方法属于精确计算内核，哪些属于近似架构，哪些属于 KV cache 方法；
2. 语言模型实验必须 causal；编码任务可以 bidirectional，但要放在另一张表；
3. 主 baseline 使用 PyTorch SDPA/Flash；
4. 快速模型使用几层、hidden size 多大、几个 heads；
5. 正式实验使用哪三个 seeds；
6. 训练、验证、测试分别用哪个 split；
7. 需要记录哪些指标；
8. 哪些配置属于正式结果，哪些只属于试运行。

**建议先定的默认值**

```text
seeds: 17, 29, 43
quick model: 2 layers, dim 256, 8 heads
main dtype: FP16 或 BF16
main baseline: PyTorch SDPA
quality dataset: WikiText-2 train/validation/test
```

**怎样算完成**

任何人拿到一个结果，都能从规则文档回答：模型看不看未来、和谁比较、数据从哪里来、运行在哪块 GPU 上。

## 第 0.2 步：修复 causal mask

**为什么做**

这是当前最会让 PPL 失效的问题。

**具体做什么**

1. 给所有注意力统一增加 `causal=True/False` 配置；
2. 语言模型运行时一律传 `causal=True`；
3. Standard full attention 在 softmax 前把未来位置分数设为负无穷；
4. SDPA/xFormers 的对照路径都使用同一个 `is_causal`；
5. Performer、Reformer 等 efficient 路径与它们的 Standard reference 使用相同 mask；
6. 检查 Linformer 是否真的支持合法 causal projection；如果不能，暂时不要把它放进 causal LM 主表；
7. 为 Longformer 分开实现“左右都能看”的编码窗口和“只能向左看”的生成窗口。

**怎么测试**

准备两条输入，它们前半段完全相同，只把后半段未来 token 换掉：

```text
A: 我 喜欢 吃 苹果 今天 晴天
B: 我 喜欢 吃 苹果 飞机 海洋
```

在预测“苹果”以前的位置时，A 和 B 的 logits 必须相同。若不同，就说明模型偷看了未来。

**怎样算完成**

- 所有 causal 方法通过 future-invariance 测试；
- Standard 与 SDPA 在同权重下输出接近；
- 结果文件明确写 `causal: true`。

## 第 0.3 步：让所有模型从同一起点出发

**为什么做**

排除随机初始化差异。

**具体做什么**

1. 创建一个 Standard 基础 TinyLM；
2. 保存它的 embedding、位置编码、MLP、LayerNorm 和输出头；
3. 创建其他架构 TinyLM；
4. 把所有形状相同的公共权重从基础模型复制过去；
5. 只有结构确实不同的 attention 参数重新初始化；
6. 记录无法复制哪些参数以及原因；
7. 固定数据 batch 顺序，让不同模型第 1 步、第 2 步看到完全相同的数据。

**还要记录什么**

- 总参数量；
- 可训练参数量；
- attention 参数量；
- 公共参数是否逐项相等。

**怎样算完成**

- 复制后公共权重完全一致；
- 同一个精确 attention 的两个实现初始 logits 一致；
- 参数量差异能被解释，而不是隐藏起来。

## 第 0.4 步：把训练集、验证集和测试集分开

**为什么做**

防止模型在考试题上训练。

**具体做什么**

修改数据加载逻辑，分别加载：

```text
WikiText-2 train       → 更新模型参数
WikiText-2 validation  → 观察训练效果、选择配置
WikiText-2 test        → 最终一次正式评分
```

将 tokenized 数据缓存到磁盘，所有模型复用同一份 token 序列。训练窗口只能从 train 中采样。

**怎样算完成**

- 三个 split 的来源写进结果；
- 训练代码不能收到 test tokens；
- 测试验证三个 split 没有混用；
- 正式报告的 test PPL 只在配置确定后生成。

## 第 0.5 步：统一配置和结果文件

**为什么做**

现在结果散落在不同目录，字段也不完全相同，人工复制数字容易出错。

**具体做什么**

每次运行都保存一条统一记录，至少包括：

```text
运行编号 run_id
代码版本 git commit
方法和具体参数
模型层数、维度、heads、参数量
数据集和 split
是否 causal
sequence length、batch size、dtype
seed、训练步数、训练 token 数
GPU、PyTorch、CUDA、实际 attention backend
每次延迟原始样本
峰值 allocated/reserved 显存
loss、NLL、PPL 或 accuracy
成功、失败、OOM 或跳过状态
```

正式结果可以存为 JSONL：一行一条记录。再用汇总程序生成 CSV/Parquet 和图表。

**怎样算完成**

- 一条结果足够还原运行环境和命令；
- 缺少关键字段时校验器会报错；
- OOM 不会被静默删掉；
- 老结果标记为 legacy，不自动混入新主表。

## 第 0.6 步：把六七份重复脚本合成统一 runner

**为什么做**

如果每种架构使用自己的计时代码，计时差异可能比模型差异还大。

**具体做什么**

建立两个入口：

1. `benchmark` runner：测速度和显存；
2. `quality` runner：训练并测 PPL/accuracy。

性能 runner 需要支持：

- forward inference；
- 完整 training step；
- 自回归 prefill；
- 自回归 decode；
- warmup；
- 多次计时；
- OOM 后继续；
- 断点续跑；
- 随机化方法运行顺序。

质量 runner 需要支持：

- 统一数据和模型配置；
- 统一 optimizer 和学习率；
- 多 seeds；
- checkpoint 保存/恢复；
- validation/test PPL；
- loss curve 和训练 tokens/s。

**怎样算完成**

同一条命令只需更换 `method` 参数就能运行不同架构，输出字段完全一致。

## 第 0.7 步：给重要假设加自动测试

**为什么做**

靠人眼检查很容易漏掉问题。测试能在正式运行前几分钟内发现错误。

**需要添加的测试**

1. 未来 token 改变不能影响过去 logits；
2. Standard 与 SDPA 同权重输出一致；
3. padding 位置不会参与 attention；
4. 所有方法输出形状正确且没有 NaN/Inf；
5. backward 后梯度有限；
6. 公共初始化相同；
7. 同 seed 可复现；
8. PPL 计算方式正确；
9. 结果 schema 能发现错误；
10. OOM 后 runner 能继续；
11. Keyformer 100% cache 与 FullKV 一致；
12. Memformer memory 确实跨块更新。

**怎样算完成**

CPU 快速测试全部通过，GPU smoke test 也通过。测试失败时，不启动正式实验。

## 第 0.8 步：先跑一个很小的 pilot

**为什么做**

一次正式矩阵可能包含数百个配置。先用很小的试运行验证整条流水线。

**具体做什么**

1. 只选 Standard、SDPA、Longformer、Performer；
2. 只测试长度 128 和 512；
3. 只用 seed=17；
4. 只训练 100–300 steps；
5. 从配置开始，完整执行训练、评估、保存、汇总和画图；
6. 记录每个配置耗时和显存；
7. 估算完整 3-seed 实验需要多少 GPU 小时。

**怎样算完成**

- 没有 NaN；
- 没有数据 split 混用；
- 图能从原始结果自动生成；
- 重复运行结果接近；
- 确认 8GB GPU 能承担选定的主模型和长度。

## 6. P1：得到第一套真正可信的主结果

## 第 1.1 步：统一测前向速度和显存

**想回答的问题**

不同方法从多长的序列开始才比 SDPA 更快或更省显存？

**具体做什么**

1. 在同一块 GPU、同一 dtype 下运行所有方法；
2. 长度从 128、256、512、1024、2048、4096、8192 逐步增加；
3. 能继续就测 16k/32k，直到 OOM；
4. batch 至少测 1 和 4；
5. causal 和 bidirectional 分开；
6. 每个配置先 warmup 20 次，再计时至少 50 次；
7. 保存每次计时，不只保存平均数；
8. 记录峰值 allocated 和 reserved 显存；
9. 实际 backend 回退也要记录。

**最后看什么**

- median latency；
- tokens/s；
- 波动区间；
- 最大不 OOM 长度；
- 相对 SDPA 的速度比和显存比；
- 速度曲线在哪里交叉。

**不要怎么解释**

如果某方法在 128–8192 都更慢，就写“本测试范围内没有速度优势”，不能因为理论复杂度更低就宣布它更快。

## 第 1.2 步：统一测完整训练速度

**想回答的问题**

只看 forward 不够。加入 backward 和 optimizer 后，哪种方法真正节省训练时间和显存？

**具体做什么**

1. 使用同一模型主干和相同 token batch；
2. 测一次完整训练 step；
3. 每个配置运行至少 50 个稳定 step；
4. 测固定 batch 下的 tokens/s；
5. 再测每种方法能放入显存的最大 batch；
6. 记录 forward、backward 和 optimizer 大致耗时；
7. 长度至少测试 512、2k、4k；
8. gradient checkpointing 开和关分开。

**最后看什么**

- 每秒处理多少训练 token；
- 峰值训练显存；
- 最大 batch；
- 达到相同训练 token 数需要多长时间。

## 第 1.3 步：重做 TinyLM/WikiText-2 质量主实验

**想回答的问题**

在公平条件下，各 attention 是否会影响语言建模质量？

**具体做什么**

1. 用 WikiText-2 train 训练；
2. 用 validation 观察曲线；
3. 配置确定后用 test 评分；
4. 所有方法使用相同 tokenizer、模型主干、数据顺序和 optimizer；
5. 运行 seeds 17、29、43；
6. context 先测 128 和 512，资源允许再测 1024；
7. 不再只训练 120 steps，至少 2k steps，理想情况训练到验证 loss 进入平台期；
8. 定期保存 checkpoint 和 validation PPL；
9. 最终 PPL 用“总 NLL 除以总有效 token 数”计算。

**最后看什么**

- 三个 seed 的平均 test PPL；
- 标准差或 95% 置信区间；
- loss 曲线是否收敛；
- 同样训练 token 数下的质量；
- 达到某个目标 PPL 所需时间。

**怎样写结论**

如果 A 的平均 PPL 比 B 低，但误差范围大量重叠，应写“没有观察到稳定差异”，而不是宣布 A 更好。

## 第 1.4 步：公平重做 attention fidelity

**它测的是什么**

Fidelity 不是模型最终准确率，而是给相同 Q/K/V 时，高效 attention 的输出与 full attention 有多像。

**具体做什么**

1. 不再只用随机 embedding，先从训练好的模型收集真实 hidden states；
2. full 和 efficient 路径使用同一套 Q/K/V/O 权重；
3. Linformer 使用训练后的 projection；
4. Reformer 使用真正共享投影的 full reference；
5. Performer 对多个随机 feature seeds 重复；
6. Longformer 分析距离窗口内外的误差；
7. 每种方法至少评估 100 个文本块；
8. 测长度 128、512、2k、4k。

**最后看什么**

- relative L2；
- cosine similarity；
- logits KL；
- top-k token overlap；
- 误差是否随长度和层数增加。

**特别注意**

Memformer 的目标不是模仿无记忆 full attention，所以不要把它放进“谁最像 full attention”的排名。它应测跨块记忆有没有保留信息。

## 第 1.5 步：把 Keyformer 的统计做扎实

**想回答的问题**

当前 768 prompt 的计时波动很大。Keyformer 的加速是否稳定，而不是某一次 FullKV 特别慢造成的？

**具体做什么**

1. 先保持 GPT-2 Medium 和原配置不变；
2. 每个 prompt、ratio、policy 至少计时 20 次，理想为 50 次；
3. 随机改变策略运行顺序；
4. 保存所有原始 latency；
5. 同时报告：
   - 两边平均延迟之比；
   - 每一对运行 speedup 的平均；
   - median speedup；
6. 用 bootstrap 计算 95% CI；
7. PPL 扩大到完整 WikiText-2 test 或明显更多 token；
8. 生成长度增加到 32、128、256；
9. profile score update、top-k、gather 和 attention 各占多少时间。

**最后看什么**

- 50% cache 在不同 prompt 下是否稳定加速；
- 速度区间是否跨过 1×；
- PPL 增幅和物理 KV 节省；
- 选择开销从多长上下文开始被节省的 attention 成本抵消。

## 第 1.6 步：画 Pareto 图而不是做总排名

**Pareto 是什么意思**

如果一个配置同时更快、更省显存、质量还更好，那么它显然支配另一个配置。剩下那些各有取舍、无法被全面击败的点组成 Pareto 前沿。

**具体做什么**

画以下图：

- PPL vs latency；
- PPL vs peak memory；
- throughput vs memory；
- sequence length vs speedup；
- sequence length vs maximum batch；
- Keyformer PPL 增幅 vs KV 节省。

把不同 GPU、dtype、模型规模和 causal/bidirectional 结果分开画。

**最后回答什么**

- 如果允许 PPL 增加不超过 1%，哪个配置最省？
- 如果显存固定为 8GB，哪个配置吞吐最高？
- 各方法相对 SDPA 的交叉长度是多少？
- 哪些方法在整个测试范围都没有工程优势？

## 第 1.7 步：做一次阶段审计

**具体检查**

1. 所有测试是否通过；
2. 三个 seeds 是否齐全；
3. 是否有 backend 回退；
4. 是否有不同配置被误合并；
5. 原始 latency 能否重新算出表格；
6. 图能否从空输出目录重新生成；
7. 报告中的每个数字能否找到 run_id；
8. OOM 和失败结果是否保留；
9. 结论有没有超过证据允许的范围。

完成这一步后，即使暂时不做 P2/P3，项目也已经有一套可信的课程级主要结果。

## 7. P2：逐个弄明白架构为什么好或不好

## 第 2.1 步：Linformer 消融

**通俗问题**

Linformer 把很多 token 的 K/V 压缩成较少的槽。压得越狠越快，但可能丢信息。需要找到压缩多少最合适。

**具体做什么**

1. 依次测试 rank k=32、64、128、256、512；
2. 比较 K 和 V 共用 projection 与各用一个 projection；
3. 比较每层独立 projection 与多层共享；
4. 在长度 512 上训练，再到 1k/2k 测试能否外推；
5. 对比随机 projection 与训练得到的 projection；
6. 每个主要候选至少跑 3 seeds。

**最后要回答**

- k 小到什么程度会明显损害质量；
- k 增大到哪里以后质量不再明显提升；
- 最佳 k 是否真的比 SDPA 更快或更省；
- 固定长度 projection 是否妨碍更长输入。

## 第 2.2 步：Longformer 消融

**通俗问题**

Longformer 让每个 token 主要看附近内容，再让少量“全局联络员 token”连接远处。要弄清窗口多大、联络员放哪里才有用。

**具体做什么**

1. 测窗口 64、128、256、512、1024；
2. 测全局 token 数量 0、1、4、16；
3. 把全局 token 放在文首、问题位置、任务标记位置或学习选择的位置；
4. 测不同层使用相同窗口与交错/dilated 窗口；
5. 运行局部语言建模；
6. 再运行需要远距离信息的长文分类或检索任务；
7. 长度从 512 扩展到 16k。

**最后要回答**

- 窗口太小会漏掉多少远程信息；
- 全局 token 是否真的恢复远距离能力；
- 当前实现从多长开始比 SDPA 更省；
- 全局 token 数增加带来多少额外成本。

## 第 2.3 步：Performer 消融

**通俗问题**

Performer 用一组随机特征近似 softmax attention。随机特征越多通常越准，但计算也越贵。

**具体做什么**

1. 测 features=32、64、128、256、512；
2. 测普通随机特征与正交随机特征；
3. 测固定随机特征与每隔一定训练步重新生成；
4. 每个 feature 数用至少 5 个随机种子；
5. causal 与 bidirectional 分开；
6. 长度从 512 测到 16k；
7. 同时记录 fidelity、PPL、速度和显存。

**最后要回答**

- 序列越长时是否需要更多 features；
- 低 features 的随机波动有多大；
- 哪个 feature 数在质量和效率之间最好；
- 重绘随机特征是帮助训练还是造成不稳定。

## 第 2.4 步：Reformer 消融

**通俗问题**

Reformer 像把 token 按内容相似性分到不同小组，只在组内认真交流。需要检查分组大小和重复分组次数。

**具体做什么**

1. 测 bucket size=32、64、128、256；
2. 测 n_hashes=1、2、4、8；
3. 每个重要配置测试至少 5 个 hash seeds；
4. 使用共享投影的 full attention 做公平 reference；
5. 长度从 512 扩展到 16k/32k；
6. 保存 hash/sort/bucket 各阶段 profile。

**最后要回答**

- 多做几次 hash 能恢复多少遗漏的信息；
- bucket 太小或太大分别有什么问题；
- 结果是否非常依赖随机 seed；
- 排序分桶的额外成本从多长开始值得。

## 第 2.5 步：真正测试 Memformer 的跨块记忆

**通俗问题**

当前 Memformer-like 实验只是给注意力拼接了一些固定 memory slots，尚未充分证明模型会把前一块信息写入记忆并带到下一块。

**具体做什么**

1. 使用作者实现中的 memory update/recurrent training 路径；
2. 把长文本切成多个 chunk；
3. 第一个 chunk 读完后更新 memory；
4. 第二个 chunk 只能看到自己和传来的 memory，不能直接看第一块原文；
5. 在第一块放一个 key，在几块之后要求输出对应 value；
6. 测 memory slots=16、32、64、128、256；
7. 测 chunk size=64、128、256、512；
8. 测梯度跨 1、2、4、8 个 chunk；
9. 对比没有 memory、只保留 recent hidden states 和完整历史。

**最后要回答**

- 记忆能保存多远；
- 多少 slots 足够；
- memory 是否只记住最近内容或发生塌缩；
- 固定 memory 成本是否真的成立。

## 第 2.6 步：Keyformer 机制消融

**通俗问题**

Keyformer 同时保留“重要旧 token”和“最近 token”。需要知道效果主要来自哪一部分，以及多久淘汰一次最划算。

**具体做什么**

1. 测 cache ratio=12.5%、25%、50%、75%、100%；
2. 测 recent 部分占预算的 0%、25%、50%、75%、100%；
3. 改变 Gumbel 温度初值和增长速度；
4. 比较累计 attention score、当前 score、频率和新近程度；
5. 比较所有层相同预算和每层不同预算；
6. 比较每个 head 独立预算和全局预算；
7. 比较每个 token 都淘汰与每 4/8/16 tokens 批量淘汰；
8. 测生成 32、128、256 tokens。

**最后要回答**

- important-old 和 recent 各自贡献多少；
- 哪些层或 heads 最不能压缩；
- 批量淘汰能否降低 top-k/gather 开销；
- PPL 增加不超过 1% 或 3% 时，最多能省多少 KV、提高多少速度。

## 第 2.7 步：xFormers/SDPA 后端实验

**通俗问题**

程序请求 xFormers 或 Flash，不代表 GPU 一定真的使用了那个 kernel。有时它会悄悄回退。

**具体做什么**

1. 分别请求 math、memory-efficient、Flash 和 xFormers backend；
2. 记录实际执行的 backend；
3. 测 FP32、FP16、BF16；
4. 测 head dimension 32、64、128；
5. 测 causal/bidirectional；
6. 测不同 batch 和长度；
7. 检查输出与精确 reference 的误差。

**最后要回答**

- 这块 GPU 在什么 shape/dtype 下使用哪个 kernel；
- 哪些配置会回退或不支持；
- 不损失质量时能获得多少速度和显存收益。

## 第 2.8 步：加入真正需要长记忆的任务

**为什么做**

短上下文 PPL 主要测试局部语言规律，不一定能暴露各架构的远程能力。

**先做低成本任务**

1. **Copy**：记住前面的序列并在后面复制；
2. **Selective Copy**：只复制被标记的部分；
3. **Associative Recall**：看到 key-value 对，后面给 key 时输出 value；
4. **Passkey**：在很长的无关文本中藏一个密码，最后询问密码；
5. **Needle-in-a-Haystack**：把目标放在不同长度、不同位置，画正确率热图。

**再做真实任务**

- Long Range Arena 子任务；
- 长文分类；
- 长文问答；
- PG-19 长文本语言建模。

**怎样公平**

- 先确认 SDPA baseline 能学会任务；
- 模型参数、训练 token 和数据相同；
- 目标长度和位置系统变化；
- Memformer 必须跨 chunk；
- Longformer 明确 global token 放置；
- Keyformer 必须让目标进入可能被淘汰的历史区。

## 第 2.9 步：检查误差会不会逐层放大

**通俗问题**

一层 attention 的输出只差一点，不代表堆 12 层后仍只差一点。误差可能逐层累积，也可能被残差连接抑制。

**具体做什么**

1. 只替换第 1 层；
2. 只替换中间层；
3. 只替换最后一层；
4. 替换前半层、后半层、隔层或全部层；
5. 每层记录 hidden state 相似度；
6. 最终记录 logits 差异、top-k overlap 和 PPL；
7. 按 attention head 和 token 距离分析；
8. Keyformer 另外检查每层/每头对 cache 压缩的敏感度。

**最后要回答**

- 哪些层最怕近似；
- 是否可以只在部分层使用高效 attention；
- 是否应该给不同层或 heads 不同 cache 预算。

## 8. P3：在现代模型和更大规模上验证

## 第 3.1 步：把 Keyformer 接到 RoPE 模型

**为什么做**

GPT-2 Medium 最长只有 1024 个位置，无法验证真正的长上下文。现代模型通常使用 RoPE，位置处理方式也不同。

**具体做什么**

1. 根据 8GB 显存先选择一个小模型，不要一次接多个；
2. 候选可以是 Pythia、TinyLlama 或 Qwen 小模型；
3. 保存 token 的原始绝对位置，不能因 cache 压缩重排 RoPE 位置；
4. 先做 100% budget 等价测试；
5. 再测 2k、4k、8k context；
6. 测完整 PPL、Passkey/Needle 和生成质量；
7. 先用 batch=1、FP16/BF16。

**停止条件**

只要 100% budget 不能与原模型对齐，就停止后续计时和质量实验，先修位置或 cache API。

## 第 3.2 步：加入更强 KV cache 对手

**为什么做**

RecentWindow 和 Random+Recent 是必要但较弱的基线。要判断 Keyformer 是否有现代竞争力，需要加入其他 KV eviction 方法。

**具体做什么**

从 H2O、StreamingLLM、SnapKV、PyramidKV 中按兼容性选择至少两个：

- 使用同一模型；
- 使用同一物理 KV budget；
- 使用同一 prompt/generation；
- 把选择开销算进 latency；
- 同时报质量、速度和显存。

## 第 3.3 步：扩大模型规模

**为什么做**

小模型上选择开销占比大，而大模型中 attention 和 KV cache 成本更高，交叉点可能改变。

**具体做什么**

1. 从约 100M 开始，再扩展到可承受的 300M/1B；
2. 优先使用预训练模型替换 attention 或微调；
3. 不建议一开始从头训练多个 1B 模型；
4. 比较模型规模增加后速度交叉长度和质量损失；
5. 保持统一 protocol 和结果 schema。

## 第 3.4 步：换不同 GPU 复测

**为什么做**

高效 attention 非常依赖 GPU 代际和 kernel 支持。在一块 GPU 上更快，不代表另一块也更快。

**具体做什么**

1. 先在当前 RTX 4070 Laptop 和 RTX 5060 Laptop 上运行同一小矩阵；
2. 使用同一依赖版本和 dtype；
3. 选择代表长度，不必复制所有训练；
4. 重点复测 SDPA/xFormers backend 和速度交叉点；
5. 条件允许再增加数据中心 GPU。

## 第 3.5 步：用 profiler 找出为什么慢

**为什么做**

理论计算量更少但实际更慢，通常不是理论错了，而是排序、索引、kernel 启动或内存搬运的成本太高。

**具体做什么**

- Keyformer：分别测 score、top-k、gather、attention、位置处理；
- Longformer：测窗口展开和局部 matmul；
- Reformer：测 hash、sort、bucket；
- Performer：测随机特征投影和线性 kernel；
- 使用 PyTorch Profiler 或 Nsight；
- 找到最大瓶颈后，先尝试批量化和减少 Python 循环；
- 确有价值时再考虑自定义 fused kernel。

**怎样算优化成功**

在同一协议下延迟下降，同时正确性测试和质量不退化。

## 第 3.6 步：加入标准长文数据集

在合成任务证明机制有效后，再选择 PG-19、LRA、LongBench 或长文分类/问答做真实验证。不要只追求数据集数量，要选择能够测试架构核心能力的任务。

## 9. 最后如何整理结果

## 第 4.1 步：所有图从原始数据自动生成

不要手动把数字复制到 Excel 或报告。汇总脚本应：

1. 读取统一 JSONL/Parquet；
2. 检查缺失配置和重复 run；
3. 计算均值、median、标准差和 CI；
4. 自动生成性能、PPL、显存和 Pareto 图；
5. 每张图同时输出对应 CSV；
6. 图标题标出 GPU、dtype、backend 和 protocol version。

## 第 4.2 步：更新最终报告

报告不要给一个简单的“总冠军”，而应按场景回答：

- 精确 attention 且重视工程部署：SDPA/xFormers 在什么范围最好；
- 长文局部任务：Longformer 需要多大窗口和多少 global tokens；
- 线性近似：Linformer/Performer 的质量成本是多少；
- 内容寻址：Reformer 从多长开始值得；
- 流式历史：Memformer 能记住多远；
- 自回归 KV 受限：Keyformer 在什么质量约束下最划算。

每个结论标记证据强度：

- 强：真实模型、完整数据、多 seed、公平对照；
- 中：统一小模型、多 seed；
- 弱：单层机制观察或 pilot。

## 第 4.3 步：让别人可以复现

README 需要说明：

- 如何安装依赖；
- quick、main、ablation 怎么运行；
- 需要多少显存和大致时长；
- 数据与模型怎么下载；
- 如何断点续跑；
- 输出在哪里；
- 第三方实现来自哪个版本；
- Windows/PowerShell 有什么注意事项。

## 第 4.4 步：最终审计和归档

1. 生成正式 run 清单；
2. 给原始结果和图表计算 checksum；
3. 保留失败和 OOM；
4. 从空目录重新生成所有图；
5. 抽查报告数字能否追到 run_id；
6. 把旧结果标为 legacy；
7. 确认报告没有混合不同 GPU、dtype 或任务语义。

## 10. 如果时间有限，最少应该做什么

### 只有 1 周

1. 修复 causal mask；
2. 分开 train/validation/test；
3. 实现配对初始化；
4. 添加未来不变性和精确等价测试；
5. 跑一个小 pilot。

这一周不要追求新结果，目标是保证以后跑出来的结果有效。

### 有 3–4 周

在上面基础上完成：

1. 统一性能 runner；
2. 128–8192 的同 GPU 性能曲线；
3. 3-seed 公平 TinyLM 实验；
4. Keyformer 20–50 次稳健计时；
5. Pareto 图和可信最小报告。

做到这里已经足以形成一套可靠的课程项目结论。

### 有 2–3 个月

继续完成各架构消融、长程任务、层级误差传播，并选择一个现代 RoPE 模型扩展 Keyformer。

## 11. 可以直接照着执行的第一批待办事项

下面是最适合立刻开始的顺序：

### 待办 1：创建协议文件

- 新建 `EXPERIMENT_PROTOCOL.md`；
- 写清 causal/bidirectional 分组；
- 决定主 baseline、模型大小、seeds 和数据 split；
- 暂时不运行 GPU 正式实验。

### 待办 2：写三个最关键测试

先写：

1. future-invariance test；
2. Standard vs SDPA equality test；
3. train/validation/test split test。

这些测试现在应该失败，因为它们会暴露现有问题。

### 待办 3：修复 attention mask

- 逐个修复 Standard、xFormers、Performer、Reformer；
- 每修一个就重新运行测试；
- Linformer 如果无法保证 causal，先标记不进入 causal 主表；
- 测试全部通过后再继续。

### 待办 4：修复数据加载

- 数据函数接受 `split` 参数；
- 训练只加载 train；
- 验证只加载 validation；
- 测试只加载 test；
- 将 split 写入 JSON。

### 待办 5：实现公共权重复制

- 创建一个基础 TinyLM；
- 复制 embedding/MLP/LayerNorm/head；
- 添加公共参数 equality test；
- 输出参数统计表。

### 待办 6：运行最小 pilot

- Standard 和 SDPA；
- 长度 128；
- 训练 100 steps；
- seed=17；
- 确认两者初始 logits、loss 曲线和结果 schema 正常。

### 待办 7：扩大 pilot

- 加入 Longformer 和 Performer；
- 加长度 512；
- 生成第一张自动图；
- 估算完整实验 GPU 时间。

完成这七项后，再启动正式的 3-seed 训练和长序列性能矩阵。