"""
Universal Transformer Benchmark Template

这个模板可以用于测试任何transformer架构的性能
只需替换或添加你想要测试的模型即可

支持的架构对比:
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
import platform
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置区域 ====================

# 在这里添加你要测试的模型导入
# EXAMPLE_IMPORTS = """
# from linformer import Linformer, LinformerSelfAttention
# from reformer import Reformer, ReformerSelfAttention
# from performer import Performer, PerformerSelfAttention
# """

# ==================== 设备设置 ====================

def get_device():
    """自动选择最佳设备"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("使用CPU (PyTorch将使用多核心)")
    return device

DEVICE = get_device()


@dataclass
class BenchmarkResult:
    """Structured measurements with legacy tuple-unpacking compatibility."""

    records: List[Dict[str, Any]]

    @property
    def times(self):
        return [record["mean_seconds"] for record in self.records]

    @property
    def memories(self):
        return [record["peak_memory_mb"] for record in self.records]

    def __iter__(self) -> Iterator[List[float]]:
        yield self.times
        yield self.memories


def get_environment_metadata():
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": DEVICE,
        "dtype": "float32",
    }
    if DEVICE == "cuda":
        metadata.update({"gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda})
    return metadata

def synchronize_device():
    """同步设备操作 (仅GPU)"""
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

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
        """获取当前内存使用量 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024

    def benchmark_model(self, model_class, model_params, seq_lengths, num_runs=10, warmup_runs=3):
        """
        测试单个模型

        参数:
            model_class: 模型类
            model_params: 模型参数字典
            seq_lengths: 要测试的序列长度列表
            num_runs: 每个测试的运行次数
        """
        print(f"\n{'='*60}")
        print(f"测试模型: {self.model_name}")
        print(f"{'='*60}")

        records = []

        for seq_len in seq_lengths:
            print(f"\n序列长度: {seq_len}")
            record = {
                "seq_len": seq_len,
                "status": "error",
                "mean_seconds": None,
                "median_seconds": None,
                "std_seconds": None,
                "peak_memory_mb": None,
                "run_times_seconds": [],
                "error": None,
            }
            model = None
            x = None
            try:
                current_params = {**model_params, "seq_len": seq_len}
                model = model_class(**current_params).to(DEVICE)
                x = torch.randn(1, seq_len, model_params.get("dim", 512), device=DEVICE)
                model.eval()

                with torch.inference_mode():
                    for _ in range(warmup_runs):
                        _ = model(x)
                synchronize_device()

                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                mem_before = self.get_memory_usage()

                run_times = []
                with torch.inference_mode():
                    for _ in range(num_runs):
                        if DEVICE == "cuda":
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            start_event.record()
                            _ = model(x)
                            end_event.record()
                            end_event.synchronize()
                            run_times.append(start_event.elapsed_time(end_event) / 1000.0)
                        else:
                            start = time.perf_counter()
                            _ = model(x)
                            run_times.append(time.perf_counter() - start)

                if DEVICE == "cuda":
                    mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
                else:
                    mem_used = max(0.0, self.get_memory_usage() - mem_before)

                record.update({
                    "status": "ok",
                    "mean_seconds": float(np.mean(run_times)),
                    "median_seconds": float(np.median(run_times)),
                    "std_seconds": float(np.std(run_times)),
                    "peak_memory_mb": float(mem_used),
                    "run_times_seconds": [float(value) for value in run_times],
                })
                print(f"   平均时间: {record['mean_seconds']:.4f}s")
                print(f"   内存使用: {mem_used:.2f}MB")
            except torch.cuda.OutOfMemoryError as e:
                record["status"] = "oom"
                record["error"] = str(e)
                print(f"   CUDA OOM: {e}")
            except Exception as e:
                print(f"   测试失败: {e}")
                record["error"] = str(e)
            finally:
                records.append(record)
                del model, x
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        return BenchmarkResult(records)

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
        print("=" * 60)
        print("多模型性能对比")
        print("=" * 60)

        all_results = {}

        for model_name, config in models_config.items():
            print(f"\n{'='*50}")
            print(f"测试: {model_name}")
            print(f"{'='*50}")

            model_class = config['class']
            model_params = config['params']

            result = self.benchmark_model(
                model_class, model_params, seq_lengths
            )

            all_results[model_name] = {
                'times': result.times,
                'memories': result.memories,
                'records': result.records,
            }

        return all_results

    def plot_comparison(self, results_dict, seq_lengths, save_name="benchmark_comparison.png"):
        """绘制性能对比图"""
        print(f"\n{'='*60}")
        print("生成可视化对比图")
        print(f"{'='*60}")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 时间对比
        ax1 = axes[0]
        for model_name, results in results_dict.items():
            times = np.asarray(results['times'], dtype=float)
            times[~np.isfinite(times)] = np.nan
            ax1.plot(seq_lengths, times, 'o-', label=model_name,
                    linewidth=2, markersize=8)

        ax1.set_xlabel('Sequence length', fontsize=12)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_title('Inference time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')

        # 内存对比
        ax2 = axes[1]
        for model_name, results in results_dict.items():
            memories = np.asarray(results['memories'], dtype=float)
            memories[~np.isfinite(memories)] = np.nan
            ax2.plot(seq_lengths, memories, 's-', label=model_name,
                    linewidth=2, markersize=8)

        ax2.set_xlabel('Sequence length', fontsize=12)
        ax2.set_ylabel('Peak memory (MB)', fontsize=12)
        ax2.set_title('Peak memory', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)

        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"图像已保存: {save_name}")

        return fig

# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""

    # 配置要测试的模型
    models_to_test = {
        'Standard Transformer': {
            'class': StandardTransformerSelfAttention,
            'params': {
                'dim': 512,
                'heads': 8,
            }
        },
        # 添加你想要测试的模型
        # 'Linformer': {
        #     'class': LinformerSelfAttention,
        #     'params': {
        #         'dim': 512,
        #         'k': 256,
        #         'heads': 8,
        #         'one_kv_head': True,
        #         'share_kv': True
        #     }
        # },
    }

    # 测试配置
    seq_lengths = [256, 512, 1024, 2048, 4096]

    # 创建benchmark
    benchmark = UniversalBenchmark("通用Transformer测试")

    # 运行对比测试
    results = benchmark.compare_models(models_to_test, seq_lengths)

    # 生成对比图
    benchmark.plot_comparison(results, seq_lengths, "my_benchmark_results.png")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

# ==================== 添加新模型的指南 ====================

ADD_MODEL_GUIDE = """
添加新模型的步骤:

1. 导入你的模型:
   from your_module import YourModel

2. (可选) 如果模型接口不同，创建适配器:
   class YourModelAdapter(nn.Module):
       def __init__(self, dim, seq_len, **kwargs):
           super().__init__()
           self.model = YourModel(
               dim=dim,
               max_seq_len=seq_len,
               # 其他参数映射
           )

       def forward(self, x, **kwargs):
           return self.model(x)

3. 在models_to_test中添加配置:
   'Your Model Name': {
       'class': YourModelAdapter,  # 或直接用YourModel
       'params': {
           'dim': 512,
           'heads': 8,
           # 其他必需参数
       }
   }

4. 运行benchmark！

支持的模型接口要求:
- __init__(self, dim, seq_len, **kwargs)
- forward(self, x, **kwargs) -> x (相同形状的输出)
"""

if __name__ == "__main__":
    print("🎯 通用Transformer Benchmark模板")
    print("=" * 60)
    print(ADD_MODEL_GUIDE)
    print("\n要运行示例，取消注释 example_usage() 中的代码")
    print("=" * 60)
