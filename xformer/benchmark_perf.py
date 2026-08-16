"""
Universal Transformer Benchmark Template with xFormers Support

This template can be used to benchmark the performance of any transformer architecture
Now supports multiple optimized components from xFormers, compared with standard Transformer

Supported architecture comparisons:
- xFormers (Memory Efficient Attention) vs Standard Transformer
- xFormers (Scaled Dot Product Attention) vs Transformer
- xFormers (Flash Attention) vs Transformer
- Linformer vs Transformer
- Reformer vs Transformer
- Performer vs Transformer
- Or any custom architecture
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

# ==================== xFormers Import ====================
try:
    import xformers.ops as xops
    # Note: xformers.components is deprecated, mainly using xformers.ops
    XFORMERS_AVAILABLE = True
    print("✅ xFormers installed and available")
except ImportError as e:
    XFORMERS_AVAILABLE = False
    print(f"⚠️  xFormers not installed: {e}")
    print("   Install command: pip install xformers")

# ==================== Configuration Area ====================

# Add your model imports here
# EXAMPLE_IMPORTS = """
# from linformer import Linformer, LinformerSelfAttention
# from reformer import Reformer, ReformerSelfAttention
# from performer import Performer, PerformerSelfAttention
# """

# ==================== Device Setup ====================

def get_device():
    """Automatically select the best device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        if XFORMERS_AVAILABLE:
            # Display xFormers supported features
            print(f"xFormers available features:")
            if torch.cuda.is_available():
                try:
                    from xformers import get_cuda_version
                    print(f"  - CUDA version: {get_cuda_version()}")
                except:
                    pass
    else:
        device = 'cpu'
        print("Using CPU (PyTorch will use multiple cores)")
        if XFORMERS_AVAILABLE:
            print("  Note: xFormers functionality is limited on CPU, GPU recommended")
    return device

DEVICE = get_device()

def synchronize_device():
    """Synchronize device operations (GPU only)"""
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

# ==================== Model Definition Area ====================

class StandardTransformerSelfAttention(nn.Module):
    """Standard self-attention - as baseline"""
    def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'Dimension must be divisible by number of heads'

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

# ==================== xFormers Model Definitions ====================

if XFORMERS_AVAILABLE:
    class XFormersMemoryEfficientAttention(nn.Module):
        """xFormers memory-efficient attention"""
        def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.):
            super().__init__()
            assert (dim % heads) == 0, 'Dimension must be divisible by number of heads'

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

            q = self.to_q(x).reshape(b, n, h, d_h)
            k = self.to_k(x).reshape(b, n, h, d_h)
            v = self.to_v(x).reshape(b, n, h, d_h)

            # Use xFormers memory-efficient attention
            try:
                # Merge attention head dimension into batch dimension
                q = q.transpose(1, 2).contiguous()  # [b, h, n, d_h]
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                # Use xFormers ops
                out = xops.memory_efficient_attention(q, k, v, attn_bias=None)

                # Rearrange output
                out = out.transpose(1, 2).reshape(b, n, -1)
                return self.to_out(out)
            except Exception as e:
                # Fall back to standard attention if xFormers fails
                print(f"xFormers failed, using standard attention: {e}")
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                dots = torch.einsum('bhnd,bhmd->bhnm', q, k) * (d_h ** -0.5)
                attn = dots.softmax(dim=-1)
                attn = self.dropout(attn)
                out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
                out = out.transpose(1, 2).reshape(b, n, -1)
                return self.to_out(out)

    class XFormersScaledDotProduct(nn.Module):
        """xFormers scaled dot-product attention (uses memory_efficient_attention)"""
        def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.):
            super().__init__()
            assert (dim % heads) == 0, 'Dimension must be divisible by number of heads'

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

            q = self.to_q(x).reshape(b, n, h, d_h)
            k = self.to_k(x).reshape(b, n, h, d_h)
            v = self.to_v(x).reshape(b, n, h, d_h)

            # Use xFormers memory-efficient attention (replaces SDPA)
            try:
                q = q.transpose(1, 2).contiguous()  # [b, h, n, d_h]
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                # Use memory_efficient_attention as SDPA replacement
                out = xops.memory_efficient_attention(q, k, v, attn_bias=None)

                out = out.transpose(1, 2).reshape(b, n, -1)
                return self.to_out(out)
            except Exception as e:
                # Fall back to standard attention
                print(f"xFormers SDPA failed, using standard attention: {e}")
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                dots = torch.einsum('bhnd,bhmd->bhnm', q, k) * (d_h ** -0.5)
                attn = dots.softmax(dim=-1)
                out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
                out = out.transpose(1, 2).reshape(b, n, -1)
                return self.to_out(out)

    class XFormersFlashAttention(nn.Module):
        """xFormers Flash Attention (requires specific hardware support)"""
        def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.):
            super().__init__()
            assert (dim % heads) == 0, 'Dimension must be divisible by number of heads'

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

            q = self.to_q(x).reshape(b, n, h, d_h)
            k = self.to_k(x).reshape(b, n, h, d_h)
            v = self.to_v(x).reshape(b, n, h, d_h)

            try:
                q = q.transpose(1, 2).contiguous()  # [b, h, n, d_h]
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                # Try to use Flash Attention
                out = xops.memory_efficient_attention(q, k, v, attn_bias=None)

                out = out.transpose(1, 2).reshape(b, n, -1)
                return self.to_out(out)
            except Exception as e:
                # Fall back to standard attention
                print(f"Flash Attention unavailable, using standard attention: {e}")
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                dots = torch.einsum('bhnd,bhmd->bhnm', q, k) * (d_h ** -0.5)
                attn = dots.softmax(dim=-1)
                out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
                out = out.transpose(1, 2).reshape(b, n, -1)
                return self.to_out(out)

    # Note: XFormersMultiHeadDispatch removed because xformers.components is deprecated
    # Mainly using optimized attention mechanisms in xformers.ops

# ==================== Other Custom Model Examples ====================

# class YourCustomAttention(nn.Module):
#     def __init__(self, dim, seq_len, heads=8, **kwargs):
#         super().__init__()
#         # Your model definition
#         pass
#
#     def forward(self, x, **kwargs):
#         # Your forward pass
#         pass

# ==================== Benchmark Framework ====================

class UniversalBenchmark:
    """Universal Transformer Performance Benchmark Framework"""

    def __init__(self, model_name="Transformer"):
        self.model_name = model_name
        self.results = defaultdict(list)
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self):
        """Get current memory usage (MB)"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_gpu_memory(self):
        """Get GPU memory usage (MB)"""
        if DEVICE == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def benchmark_model(self, model_class, model_params, seq_lengths, num_runs=10, warmup_runs=3):
        """
        Benchmark a single model

        Args:
            model_class: Model class
            model_params: Model parameter dictionary
            seq_lengths: List of sequence lengths to test
            num_runs: Number of runs per test
            warmup_runs: Number of warmup runs
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking model: {self.model_name}")
        print(f"{'='*60}")

        times = []
        memories = []
        gpu_memories = []

        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")

            # Clean environment
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Get baseline memory
            baseline_mem = self.get_memory_usage()
            baseline_gpu_mem = self.get_gpu_memory()

            # Create model - don't modify original params, create copy
            current_params = {**model_params, 'seq_len': seq_len}
            model = model_class(**current_params).to(DEVICE)
            x = torch.randn(1, seq_len, model_params.get('dim', 512)).to(DEVICE)

            # Model creation memory cost
            model_mem = self.get_memory_usage() - baseline_mem
            model_gpu_mem = self.get_gpu_memory() - baseline_gpu_mem

            # Warmup
            try:
                for _ in range(warmup_runs):
                    _ = model(x)
                synchronize_device()
            except Exception as e:
                print(f"   ❌ Warmup failed: {e}")
                del model, x
                continue

            # Ensure all operations complete
            synchronize_device()

            # Timing
            try:
                run_times = []
                for _ in range(num_runs):
                    start = time.time()
                    _ = model(x)
                    synchronize_device()
                    run_times.append(time.time() - start)

                avg_time = np.mean(run_times)
                std_time = np.std(run_times)

                print(f"   Average time: {avg_time:.4f}s (±{std_time:.4f}s)")
                print(f"   Memory usage: {model_mem:.2f}MB")
                if DEVICE == 'cuda':
                    print(f"   GPU memory: {model_gpu_mem:.2f}MB")

                times.append(avg_time)
                memories.append(model_mem)
                gpu_memories.append(model_gpu_mem)

            except Exception as e:
                print(f"   ❌ Benchmark failed: {e}")
                times.append(float('inf'))
                memories.append(0)
                gpu_memories.append(0)

            # Cleanup
            del model, x
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

        return times, memories, gpu_memories

    def compare_models(self, models_config, seq_lengths):
        """
        Compare multiple models

        Args:
            models_config: Model configuration dictionary
                {
                    'Model1': {'class': Model1Class, 'params': {...}},
                    'Model2': {'class': Model2Class, 'params': {...}},
                }
            seq_lengths: List of sequence lengths
        """
        print("=" * 60)
        print("Multi-model Performance Comparison")
        print("=" * 60)

        all_results = {}

        for model_name, config in models_config.items():
            print(f"\n{'='*50}")
            print(f"Testing: {model_name}")
            print(f"{'='*50}")

            model_class = config['class']
            model_params = config['params']

            times, memories, gpu_memories = self.benchmark_model(
                model_class, model_params, seq_lengths
            )

            all_results[model_name] = {
                'times': times,
                'memories': memories,
                'gpu_memories': gpu_memories
            }

        return all_results

    def plot_comparison(self, results_dict, seq_lengths, save_name="benchmark_comparison.png"):
        """Plot performance comparison charts"""
        print(f"\n{'='*60}")
        print("Generating visualization comparison charts")
        print(f"{'='*60}")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Time comparison
        ax1 = axes[0]
        for model_name, results in results_dict.items():
            times = results['times']
            ax1.plot(seq_lengths, times, 'o-', label=model_name,
                    linewidth=2, markersize=8)

        ax1.set_xlabel('Sequence Length', fontsize=12)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_title('Time Performance Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')

        # Memory comparison
        ax2 = axes[1]
        for model_name, results in results_dict.items():
            memories = results['memories']
            ax2.plot(seq_lengths, memories, 's-', label=model_name,
                    linewidth=2, markersize=8)

        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax2.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)

        # GPU memory comparison
        ax3 = axes[2]
        if DEVICE == 'cuda':
            for model_name, results in results_dict.items():
                gpu_memories = results['gpu_memories']
                ax3.plot(seq_lengths, gpu_memories, '^-', label=model_name,
                        linewidth=2, markersize=8)

            ax3.set_xlabel('Sequence Length', fontsize=12)
            ax3.set_ylabel('GPU Memory Usage (MB)', fontsize=12)
            ax3.set_title('GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log', base=2)
        else:
            ax3.text(0.5, 0.5, 'GPU Not Available', ha='center', va='center',
                    fontsize=16, transform=ax3.transAxes)
            ax3.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved: {save_name}")

        return fig

    def print_summary(self, results_dict, seq_lengths):
        """Print benchmark results summary"""
        print(f"\n{'='*80}")
        print("Performance Benchmark Results Summary")
        print(f"{'='*80}")
        print(f"{'Model':<30} | {'Seq Len':<10} | {'Time(s)':<10} | {'Memory(MB)':<10} | {'GPU(MB)':<10}")
        print("-" * 80)

        for model_name, results in results_dict.items():
            times = results['times']
            memories = results['memories']
            gpu_memories = results['gpu_memories']

            for i, seq_len in enumerate(seq_lengths):
                if i < len(times):
                    print(f"{model_name:<30} | {seq_len:<10} | {times[i]:<10.4f} | {memories[i]:<10.2f} | {gpu_memories[i]:<10.2f}")

        print("-" * 80)

        # Calculate speedup
        baseline_name = list(results_dict.keys())[0]
        baseline_times = results_dict[baseline_name]['times']

        print(f"\nSpeedup relative to {baseline_name}:")
        print(f"{'Model':<30} | {'Avg Speedup':<15} | {'Max Speedup':<15}")
        print("-" * 60)

        for model_name, results in results_dict.items():
            if model_name == baseline_name:
                continue

            times = results['times']
            speedups = [baseline_times[i] / times[i] if times[i] > 0 else 0
                       for i in range(min(len(times), len(baseline_times)))]

            if speedups:
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                print(f"{model_name:<30} | {avg_speedup:<15.2f}x | {max_speedup:<15.2f}x")

# ==================== Usage Examples ====================

def example_usage():
    """Usage example - xFormers vs Standard Transformer"""

    # Configure models to test
    models_to_test = {
        'Standard Transformer': {
            'class': StandardTransformerSelfAttention,
            'params': {
                'dim': 512,
                'heads': 8,
            }
        },
    }

    # Add xFormers models if available
    if XFORMERS_AVAILABLE:
        models_to_test.update({
            'xFormers Memory Efficient': {
                'class': XFormersMemoryEfficientAttention,
                'params': {
                    'dim': 512,
                    'heads': 8,
                }
            },
            'xFormers SDPA': {
                'class': XFormersScaledDotProduct,
                'params': {
                    'dim': 512,
                    'heads': 8,
                }
            },
            'xFormers Flash Attention': {
                'class': XFormersFlashAttention,
                'params': {
                    'dim': 512,
                    'heads': 8,
                }
            },
        })
    else:
        print("⚠️  xFormers not available, only testing Standard Transformer")

    # Test configuration - sequence length range 256~8192
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    # Create benchmark
    benchmark = UniversalBenchmark("xFormers vs Transformer Performance Test")

    # Run comparison test
    results = benchmark.compare_models(models_to_test, seq_lengths)

    # Generate comparison charts
    benchmark.plot_comparison(results, seq_lengths, "xformer_vs_transformer_benchmark.png")

    # Print summary
    benchmark.print_summary(results, seq_lengths)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)

# ==================== Quick Test Example ====================

def quick_test():
    """Quick test - using smaller configuration"""
    print("🚀 Quick test mode")

    models_to_test = {
        'Standard Transformer': {
            'class': StandardTransformerSelfAttention,
            'params': {
                'dim': 256,
                'heads': 4,
            }
        },
    }

    if XFORMERS_AVAILABLE:
        models_to_test['xFormers Memory Efficient'] = {
            'class': XFormersMemoryEfficientAttention,
            'params': {
                'dim': 256,
                'heads': 4,
            }
        }

    # CPU-friendly sequence lengths - smaller range
    seq_lengths = [128, 256, 512]

    benchmark = UniversalBenchmark("Quick Test")
    results = benchmark.compare_models(models_to_test, seq_lengths)
    benchmark.print_summary(results, seq_lengths)

# ==================== Guide for Adding New Models ====================

ADD_MODEL_GUIDE = """
Steps to add a new model:

1. Import your model:
   from your_module import YourModel

2. (Optional) If the model interface differs, create an adapter:
   class YourModelAdapter(nn.Module):
       def __init__(self, dim, seq_len, **kwargs):
           super().__init__()
           self.model = YourModel(
               dim=dim,
               max_seq_len=seq_len,
               # Other parameter mappings
           )

       def forward(self, x, **kwargs):
           return self.model(x)

3. Add configuration in models_to_test:
   'Your Model Name': {
       'class': YourModelAdapter,  # or directly use YourModel
       'params': {
           'dim': 512,
           'heads': 8,
           # Other required parameters
       }
   }

4. Run the benchmark!

Required model interface:
- __init__(self, dim, seq_len, **kwargs)
- forward(self, x, **kwargs) -> x (output with same shape)

xFormers installation:
pip install xformers
or
conda install -c xformers xformers
"""

# ==================== Main Program ====================

if __name__ == "__main__":
    print("🎯 xFormers vs Transformer Benchmark")
    print("=" * 60)

    # Display environment information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"xFormers available: {XFORMERS_AVAILABLE}")
    print("=" * 60)

    print("\nSelect test mode:")
    print("1. Full test (example_usage)")
    print("2. Quick test (quick_test)")

    # Run full test
    print("\n🚀 Running full test...")
    example_usage()

    # To run quick test, uncomment the line below
    # quick_test()
    print("=" * 60)
    print(ADD_MODEL_GUIDE)