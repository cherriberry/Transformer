"""
Universal Transformer Benchmark Template

这个模板可以用于测试任何transformer架构的性能
只需替换或添加你想要测试的模型即可

支持的架构对比:
- Longformer vs Standard Transformer
- Linformer vs Standard Transformer
- Reformer vs Transformer
- Performer vs Transformer
- FNet vs Transformer
- 或者任何自定义架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import gc
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math

# ==================== 配置区域 ====================

# 在这里添加你要测试的模型导入
# EXAMPLE_IMPORTS = """
# from linformer import Linformer, LinformerSelfAttention
# from reformer import Reformer, ReformerSelfAttention
# from performer import Performer, PerformerSelfAttention
# """

# ==================== 设备设置 ====================

def get_device():
    """使用CPU运行"""
    device = 'cpu'
    print("💻 使用CPU运行")
    return device

DEVICE = get_device()

def synchronize_device():
    """CPU不需要同步"""
    pass

# ==================== 模型定义区域 ====================

class StandardTransformerSelfAttention(nn.Module):
    """标准自注意力 - 作为baseline"""
    def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, '维度必须能被头数整除'

        self.heads = heads
        dim_head = dim_head if dim_head is not None else dim // heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, **kwargs):
        b, n, d = x.shape
        h, d_h = self.heads, self.dim_head

        q = self.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)
        k = self.to_k(x).reshape(b, n, h, d_h).transpose(1, 2)
        v = self.to_v(x).reshape(b, n, h, d_h).transpose(1, 2)

        dots = torch.einsum('bhnd,bhmd->bhnm', q, k) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

# ==================== Longformer 实现 ====================

class LongformerSelfAttention(nn.Module):
    """
    Longformer自注意力机制

    结合了局部窗口注意力和全局注意力，支持长序列处理
    论文: Longformer: The Long-Document Transformer
    """
    def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.,
                 window_size=512, global_attention_indices=None):
        super().__init__()
        assert (dim % heads) == 0, '维度必须能被头数整除'

        self.heads = heads
        dim_head = dim_head if dim_head is not None else dim // heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.seq_len = seq_len

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

        # 全局注意力索引（如果未指定，使用第一个和最后一个token）
        if global_attention_indices is None:
            self.global_attention_indices = [0, seq_len - 1]
        else:
            self.global_attention_indices = global_attention_indices

    def forward(self, x, **kwargs):
        b, n, d = x.shape
        h, d_h = self.heads, self.dim_head

        # 计算Q, K, V
        q = self.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)  # (b, h, n, d_h)
        k = self.to_k(x).reshape(b, n, h, d_h).transpose(1, 2)
        v = self.to_v(x).reshape(b, n, h, d_h).transpose(1, 2)

        # 计算局部窗口注意力
        window_attn = self._compute_window_attention(q, k, v)

        # 计算全局注意力
        global_attn = self._compute_global_attention(q, k, v)

        # 合并局部和全局注意力
        combined_attn = torch.cat([window_attn, global_attn], dim=-1)

        return self.to_out(combined_attn)

    def _compute_window_attention(self, q, k, v):
        """计算滑动窗口注意力"""
        b, h, n, d_h = q.shape
        w = self.window_size

        outputs = []

        for i in range(n):
            # 定义窗口范围
            start = max(0, i - w // 2)
            end = min(n, i + w // 2 + 1)

            # 提取窗口内的queries, keys, values
            q_window = q[:, :, i:i+1, :]  # (b, h, 1, d_h)
            k_window = k[:, :, start:end, :]  # (b, h, window, d_h)
            v_window = v[:, :, start:end, :]  # (b, h, window, d_h)

            # 计算注意力分数
            dots = torch.einsum('bhid,bhjd->bhij', q_window, k_window) * (d_h ** -0.5)
            attn = dots.softmax(dim=-1)
            attn = self.dropout(attn)

            # 应用注意力到values
            out = torch.einsum('bhij,bhjd->bhid', attn, v_window)
            outputs.append(out)

        # 堆叠所有输出
        return torch.cat(outputs, dim=2)  # (b, h, n, d_h)

    def _compute_global_attention(self, q, k, v):
        """计算全局注意力（每个token attend到全局token）"""
        b, h, n, d_h = q.shape

        # 提取全局token的keys和values
        global_indices = self.global_attention_indices
        k_global = k[:, :, global_indices, :]  # (b, h, num_global, d_h)
        v_global = v[:, :, global_indices, :]  # (b, h, num_global, d_h)

        # 所有queries attend到全局keys
        dots = torch.einsum('bhnd,bhmd->bhnm', q, k_global) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v_global)  # (b, h, n, d_h)

        return out

class LongformerSlidingWindowAttention(nn.Module):
    """
    简化版Longformer滑动窗口注意力

    更高效的实现，使用矩阵操作代替循环
    """
    def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.,
                 window_size=512):
        super().__init__()
        assert (dim % heads) == 0, '维度必须能被头数整除'

        self.heads = heads
        dim_head = dim_head if dim_head is not None else dim // heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.seq_len = seq_len

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, **kwargs):
        b, n, d = x.shape
        h, d_h = self.heads, self.dim_head
        w = self.window_size

        q = self.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)
        k = self.to_k(x).reshape(b, n, h, d_h).transpose(1, 2)
        v = self.to_v(x).reshape(b, n, h, d_h).transpose(1, 2)

        # 创建局部注意力掩码
        attn_mask = self._create_sliding_window_mask(n, w).to(q.device)

        # 计算注意力分数
        dots = torch.einsum('bhnd,bhmd->bhnm', q, k) * (d_h ** -0.5)

        # 应用掩码（将不需要注意的位置设为负无穷）
        dots = dots.masked_fill(attn_mask == 0, float('-inf'))

        # 计算注意力权重
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)

    def _create_sliding_window_mask(self, seq_len, window_size):
        """创建滑动窗口掩码"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

class LongformerDilatedAttention(nn.Module):
    """
    Longformer扩展窗口注意力（Dilated Attention）

    使用扩展窗口来捕获更长距离的依赖关系
    """
    def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.,
                 window_size=512, dilation=2):
        super().__init__()
        assert (dim % heads) == 0, '维度必须能被头数整除'

        self.heads = heads
        dim_head = dim_head if dim_head is not None else dim // heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.dilation = dilation
        self.seq_len = seq_len

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, **kwargs):
        b, n, d = x.shape
        h, d_h = self.heads, self.dim_head
        w = self.window_size

        q = self.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)
        k = self.to_k(x).reshape(b, n, h, d_h).transpose(1, 2)
        v = self.to_v(x).reshape(b, n, h, d_h).transpose(1, 2)

        # 创建扩展窗口掩码
        attn_mask = self._create_dilated_mask(n, w, self.dilation).to(q.device)

        # 计算注意力分数
        dots = torch.einsum('bhnd,bhmd->bhnm', q, k) * (d_h ** -0.5)

        # 应用掩码
        dots = dots.masked_fill(attn_mask == 0, float('-inf'))

        # 计算注意力权重
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)

    def _create_dilated_mask(self, seq_len, window_size, dilation):
        """创建扩展窗口掩码"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # 中心窗口
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1

            # 扩展窗口
            for d in range(1, dilation + 1):
                # 左侧扩展
                left_start = max(0, i - window_size // 2 - d * window_size)
                left_end = max(0, i - window_size // 2 - (d-1) * window_size)
                if left_start < left_end:
                    mask[i, left_start:left_end] = 1

                # 右侧扩展
                right_start = min(seq_len, i + window_size // 2 + (d-1) * window_size)
                right_end = min(seq_len, i + window_size // 2 + d * window_size)
                if right_start < right_end:
                    mask[i, right_start:right_end] = 1

        return mask.unsqueeze(0).unsqueeze(0)

# ==================== 在这里添加你的模型 ====================

# 示例：添加自定义模型
# class YourCustomAttention(nn.Module):
#     def __init__(self, dim, seq_len, heads=8, **kwargs):
#         super().__init__()
#         # 你的模型定义
#         pass
#
#     def forward(self, x, **kwargs):
#         # 你的前向传播
#         pass

# ==================== Benchmark框架 ====================

class UniversalBenchmark:
    """通用Transformer性能测试框架"""

    def __init__(self, model_name="Transformer"):
        self.model_name = model_name
        self.results = defaultdict(list)
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self):
        """获取CPU内存使用量 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_model_memory(self, model, input_tensor):
        """
        更准确的模型内存测量方法
        测量模型本身的内存占用
        """
        try:
            # 创建模型的参数内存
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

            # 测量前向传播的内存增长
            gc.collect()
            mem_before = self.get_memory_usage()

            # 执行前向传播
            with torch.no_grad():
                _ = model(input_tensor)

            # 强制同步确保计算完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            mem_after = self.get_memory_usage()
            forward_memory = max(0, mem_after - mem_before)  # 确保非负

            total_memory = param_memory + forward_memory
            return total_memory

        except Exception as e:
            print(f"内存测量错误: {str(e)[:30]}")
            return 0

    def benchmark_model(self, model_class, model_params, seq_lengths, num_runs=10):
        """
        测试单个模型

        参数:
            model_class: 模型类
            model_params: 模型参数字典
            seq_lengths: 要测试的序列长度列表
            num_runs: 每个测试的运行次数
        """
        print(f"\n测试模型: {self.model_name}")
        print("-" * 40)

        dim = model_params.get('dim', 512)
        times = []
        memories = []

        for seq_len in seq_lengths:
            print(f"序列长度: {seq_len}", end=" ")

            try:
                # 创建模型
                current_params = {**model_params, 'seq_len': seq_len}
                model = model_class(**current_params)
                x = torch.randn(1, seq_len, dim)

                # 预热
                for _ in range(3):
                    _ = model(x)

                # 内存测量 - 使用新的更准确的方法
                mem_used = self.get_model_memory(model, x)
                if mem_used <= 0:
                    raise Exception("内存测量失败")

                # 计时
                run_times = []
                for _ in range(num_runs):
                    start = time.time()
                    _ = model(x)
                    run_times.append(time.time() - start)

                avg_time = np.mean(run_times)

                print(f"✅ 时间: {avg_time:.4f}s, 内存: {mem_used:.1f}MB")

                times.append(avg_time)
                memories.append(mem_used)

                # 清理
                del model, x

            except Exception as e:
                print(f"❌ 失败: {str(e)[:50]}")
                times.append(float('inf'))
                memories.append(0)

                # 清理
                try:
                    del model, x
                except:
                    pass

        return times, memories

    def compare_models(self, models_config, seq_lengths):
        """
        比较多个模型

        参数:
            models_config: 模型配置字典
                {
                    'Model1': {'class': Model1Class, 'params': {...}},
                    'Model2': {'class': Model2Class, 'params': {...}},
                }
            seq_lengths: 序列长度列表
        """
        print(f"\n{'='*50}")
        print(f"测试 {len(models_config)} 个模型，序列长度: {seq_lengths}")
        print(f"{'='*50}\n")

        all_results = {}

        for model_name, config in models_config.items():
            model_class = config['class']
            model_params = config['params']

            times, memories = self.benchmark_model(
                model_class, model_params, seq_lengths
            )

            all_results[model_name] = {
                'times': times,
                'memories': memories
            }

        return all_results

    def plot_comparison(self, results_dict, seq_lengths, save_name="benchmark_comparison.png"):
        """绘制性能对比图"""
        print("\n生成对比图...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 时间对比
        ax1 = axes[0]
        for model_name, results in results_dict.items():
            times = results['times']
            valid_indices = [i for i, t in enumerate(times) if t != float('inf')]
            valid_seq_lengths = [seq_lengths[i] for i in valid_indices]
            valid_times = [times[i] for i in valid_indices]

            if valid_times:
                ax1.plot(valid_seq_lengths, valid_times, 'o-', label=model_name, linewidth=2)

        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')

        # 内存对比
        ax2 = axes[1]
        for model_name, results in results_dict.items():
            memories = results['memories']
            valid_indices = [i for i, m in enumerate(memories) if m > 0]
            valid_seq_lengths = [seq_lengths[i] for i in valid_indices]
            valid_memories = [memories[i] for i in valid_indices]

            if valid_memories:
                ax2.plot(valid_seq_lengths, valid_memories, 's-', label=model_name, linewidth=2)

        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)

        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ 图像已保存: {save_name}")

        return fig

    def print_results_summary(self, results_dict, seq_lengths, models_config):
        """打印结果摘要"""
        print(f"\n{'='*60}")
        print("性能测试结果汇总")
        print(f"{'='*60}")

        # 时间性能表格
        print("\n时间对比 (秒):")
        print("-" * 60)
        header = f"{'序列长度':<12}"
        for model_name in models_config.keys():
            short_name = model_name[:18]
            header += f"{short_name:<12}"
        print(header)
        print("-" * 60)

        for idx, seq_len in enumerate(seq_lengths):
            row = f"{seq_len:<12}"
            for model_name in models_config.keys():
                if idx < len(results_dict[model_name]['times']):
                    time_result = results_dict[model_name]['times'][idx]
                    if time_result != float('inf'):
                        row += f"{time_result:<12.4f}"
                    else:
                        row += f"{'失败':<12}"
                else:
                    row += f"{'N/A':<12}"
            print(row)

        # 内存性能表格
        print("\n内存对比 (MB):")
        print("-" * 60)
        print(header)
        print("-" * 60)

        for idx, seq_len in enumerate(seq_lengths):
            row = f"{seq_len:<12}"
            for model_name in models_config.keys():
                if idx < len(results_dict[model_name]['memories']):
                    mem_result = results_dict[model_name]['memories'][idx]
                    if mem_result > 0:
                        row += f"{mem_result:<12.1f}"
                    else:
                        row += f"{'失败':<12}"
                else:
                    row += f"{'N/A':<12}"
            print(row)

        # 性能排名
        print(f"\n性能排名 (按序列长度 {seq_lengths[0]}):")
        print("-" * 40)

        performance_data = []
        for model_name in models_config.keys():
            if results_dict[model_name]['times'][0] != float('inf'):
                performance_data.append({
                    'name': model_name,
                    'time': results_dict[model_name]['times'][0],
                    'memory': results_dict[model_name]['memories'][0]
                })

        if performance_data:
            performance_data.sort(key=lambda x: x['time'])
            for rank, data in enumerate(performance_data, 1):
                print(f"{rank}. {data['name']:<25} 时间: {data['time']:.4f}s, 内存: {data['memory']:.1f}MB")
        else:
            print("没有成功完成的测试")

        print(f"{'='*60}")

# ==================== 使用示例 ====================

def example_usage():
    """基础测试 - 标准Transformer vs Longformer"""

    models_to_test = {
        'Standard Transformer': {
            'class': StandardTransformerSelfAttention,
            'params': {'dim': 512, 'heads': 8}
        },
        'Longformer (Window=256)': {
            'class': LongformerSlidingWindowAttention,
            'params': {'dim': 512, 'heads': 8, 'window_size': 256}
        },
        'Longformer (Window=512)': {
            'class': LongformerSlidingWindowAttention,
            'params': {'dim': 512, 'heads': 8, 'window_size': 512}
        },
    }

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    benchmark = UniversalBenchmark("基础性能测试")
    results = benchmark.compare_models(models_to_test, seq_lengths)
    benchmark.plot_comparison(results, seq_lengths, "benchmark_results.png")
    benchmark.print_results_summary(results, seq_lengths, models_to_test)

    print("\n✅ 测试完成！结果图像: benchmark_results.png")
    return results

def window_test():
    """窗口大小对比测试"""

    window_sizes = [128, 256, 512]
    models_to_test = {}

    for window_size in window_sizes:
        models_to_test[f'Longformer (Window={window_size})'] = {
            'class': LongformerSlidingWindowAttention,
            'params': {'dim': 512, 'heads': 8, 'window_size': window_size}
        }

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    benchmark = UniversalBenchmark("窗口大小对比")
    results = benchmark.compare_models(models_to_test, seq_lengths)
    benchmark.plot_comparison(results, seq_lengths, "window_comparison.png")
    benchmark.print_results_summary(results, seq_lengths, models_to_test)

    print("\n✅ 窗口测试完成！结果图像: window_comparison.png")
    return results

def dilation_test():
    """扩展窗口对比测试"""

    models_to_test = {
        'Sliding Window (512)': {
            'class': LongformerSlidingWindowAttention,
            'params': {'dim': 512, 'heads': 8, 'window_size': 512}
        },
        'Dilated (512, d=2)': {
            'class': LongformerDilatedAttention,
            'params': {'dim': 512, 'heads': 8, 'window_size': 512, 'dilation': 2}
        },
    }

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    benchmark = UniversalBenchmark("扩展窗口对比")
    results = benchmark.compare_models(models_to_test, seq_lengths)
    benchmark.plot_comparison(results, seq_lengths, "dilation_comparison.png")
    benchmark.print_results_summary(results, seq_lengths, models_to_test)

    print("\n✅ 扩展窗口测试完成！结果图像: dilation_comparison.png")
    return results

# ==================== 添加新模型的指南 ====================

ADD_MODEL_GUIDE = """
添加自定义模型:

1. 创建模型类:
class YourModel(nn.Module):
    def __init__(self, dim, seq_len, **kwargs):
        super().__init__()
        # 你的模型代码

    def forward(self, x, **kwargs):
        # 你的前向传播
        return output

2. 在测试配置中添加:
'YourModel': {
    'class': YourModel,
    'params': {'dim': 512, 'heads': 8}
}
"""

LONGFORMER_FEATURES = """
Longformer特性:
- 内存效率: O(n×w) vs 标准Transformer的O(n²)
- 长序列支持: 可处理超过8000 token
- 滑动窗口: 每个token只attend固定窗口内的邻居
- 扩展窗口: 捕获更长距离的依赖关系
"""

if __name__ == "__main__":
    import sys

    print("Longformer性能测试框架")
    print("=" * 50)
    print(f"设备: {DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")
    print("=" * 50)

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = "1"

    print("\n选择测试模式:")
    print("1. 基础测试 (推荐)")
    print("2. 窗口大小测试")
    print("3. 扩展窗口测试")
    print("=" * 50)

    try:
        if choice == "1":
            print("\n开始基础测试...")
            example_usage()
        elif choice == "2":
            print("\n开始窗口测试...")
            window_test()
        elif choice == "3":
            print("\n开始扩展窗口测试...")
            dilation_test()
        else:
            print("\n使用方法:")
            print("python universal_benchmark.py 1  # 基础测试")
            print("python universal_benchmark.py 2  # 窗口测试")
            print("python universal_benchmark.py 3  # 扩展窗口测试")
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()