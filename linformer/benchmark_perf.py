"""
Linformer Performance Benchmark and Reproduction Script

This script reproduces the key experiments from the Linformer paper:
1. Complexity analysis (time and space)
2. Performance comparison with standard Transformer
3. Quality evaluation on various sequence lengths
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
from linformer import Linformer, LinformerSelfAttention, LinformerLM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device detection - automatically select CPU or GPU
def get_device():
    """Automatically select the best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU (PyTorch will use multiple cores)")
    return device

# Global device variable
DEVICE = get_device()

def synchronize_device():
    """Synchronize device operations (GPU only)"""
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

class StandardTransformerSelfAttention(nn.Module):
    """Standard Self-Attention for comparison"""
    def __init__(self, dim, seq_len, heads=8, dim_head=None, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

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

        # Standard O(n²) attention
        dots = torch.einsum('bhnd,bhmd->bhnm', q, k) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

# Remove the leftover synchronize line (was line 43)
class PerformanceBenchmark:
    """Benchmark Linformer vs Standard Transformer"""

    def __init__(self):
        self.results = defaultdict(list)
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def benchmark_attention_complexity(self):
        """Benchmark 1: Time and Space Complexity Analysis"""
        print("=" * 60)
        print("Benchmark 1: Time and Space Complexity Analysis")
        print("=" * 60)

        seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
        dim = 512
        heads = 8
        k = 256  # Linformer projection dimension
        num_runs = 10

        std_attn_times = []
        linformer_times = []
        std_attn_memory = []
        linformer_memory = []

        for seq_len in seq_lengths:
            print(f"\nSequence Length: {seq_len}")

            # Standard Attention
            std_attn = StandardTransformerSelfAttention(dim, seq_len, heads).to(DEVICE)
            x = torch.randn(1, seq_len, dim).to(DEVICE)

            # Warmup
            for _ in range(3):
                _ = std_attn(x)
            synchronize_device()

            # Memory before
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            mem_before = self.get_memory_usage()

            # Timing
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = std_attn(x)
                synchronize_device()
                times.append(time.time() - start)

            std_time = np.mean(times)
            std_mem = self.get_memory_usage() - mem_before

            # Linformer Attention
            linformer_attn = LinformerSelfAttention(dim, seq_len, k=k, heads=heads,
                                                    one_kv_head=True, share_kv=True).to(DEVICE)

            # Warmup
            for _ in range(3):
                _ = linformer_attn(x)
            synchronize_device()

            # Memory before
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            mem_before = self.get_memory_usage()

            # Timing
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = linformer_attn(x)
                synchronize_device()
                times.append(time.time() - start)

            linf_time = np.mean(times)
            linf_mem = self.get_memory_usage() - mem_before

            print(f"  Standard Attention: {std_time:.4f}s, Memory: {std_mem:.2f}MB")
            print(f"  Linformer:          {linf_time:.4f}s, Memory: {linf_mem:.2f}MB")
            print(f"  Speedup:            {std_time/linf_time:.2f}x")

            std_attn_times.append(std_time)
            linformer_times.append(linf_time)
            std_attn_memory.append(std_mem)
            linformer_memory.append(linf_mem)

            # Cleanup
            del std_attn, linformer_attn, x
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

        self.results['seq_lengths'] = seq_lengths
        self.results['std_attn_times'] = std_attn_times
        self.results['linformer_times'] = linformer_times
        self.results['std_attn_memory'] = std_attn_memory
        self.results['linformer_memory'] = linformer_memory

        return seq_lengths, std_attn_times, linformer_times, std_attn_memory, linformer_memory

    def benchmark_quality_comparison(self):
        """Benchmark 2: Quality Comparison on Synthetic Task"""
        print("\n" + "=" * 60)
        print("Benchmark 2: Quality Comparison on Synthetic Task")
        print("=" * 60)

        # Synthetic task: sequence classification
        seq_len = 1024
        dim = 512
        num_classes = 10
        num_samples = 100

        print(f"\nTask: Sequence Classification")
        print(f"Sequence Length: {seq_len}, Dimension: {dim}, Classes: {num_classes}")

        # Generate synthetic data
        X_train = torch.randn(num_samples, seq_len, dim).to(DEVICE)
        y_train = torch.randint(0, num_classes, (num_samples,)).to(DEVICE)

        X_test = torch.randn(20, seq_len, dim).to(DEVICE)
        y_test = torch.randint(0, num_classes, (20,)).to(DEVICE)

        # Standard Transformer Model
        class TransformerClassifier(nn.Module):
            def __init__(self, dim, seq_len, num_classes, heads=8):
                super().__init__()
                self.attn = StandardTransformerSelfAttention(dim, seq_len, heads)
                self.norm = nn.LayerNorm(dim)
                self.fc = nn.Linear(dim, num_classes)

            def forward(self, x):
                x = self.attn(x)
                x = self.norm(x.mean(dim=1))  # Pooling
                return self.fc(x)

        # Linformer Model
        class LinformerClassifier(nn.Module):
            def __init__(self, dim, seq_len, num_classes, k=256, heads=8):
                super().__init__()
                self.attn = LinformerSelfAttention(dim, seq_len, k=k, heads=heads,
                                                 one_kv_head=True, share_kv=True)
                self.norm = nn.LayerNorm(dim)
                self.fc = nn.Linear(dim, num_classes)

            def forward(self, x):
                x = self.attn(x)
                x = self.norm(x.mean(dim=1))  # Pooling
                return self.fc(x)

        # Train Standard Transformer
        print("\nTraining Standard Transformer...")
        std_model = TransformerClassifier(dim, seq_len, num_classes).to(DEVICE)
        optimizer = torch.optim.Adam(std_model.parameters(), lr=0.001)

        std_train_losses = []
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = std_model(X_train)
            loss = F.cross_entropy(outputs, y_train)
            loss.backward()
            optimizer.step()
            std_train_losses.append(loss.item())

        # Train Linformer
        print("Training Linformer...")
        linf_model = LinformerClassifier(dim, seq_len, num_classes, k=256).to(DEVICE)
        optimizer = torch.optim.Adam(linf_model.parameters(), lr=0.001)

        linf_train_losses = []
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = linf_model(X_train)
            loss = F.cross_entropy(outputs, y_train)
            loss.backward()
            optimizer.step()
            linf_train_losses.append(loss.item())

        # Evaluate
        with torch.no_grad():
            std_model.eval()
            linf_model.eval()

            std_outputs = std_model(X_test)
            linf_outputs = linf_model(X_test)

            std_acc = (std_outputs.argmax(dim=1) == y_test).float().mean().item()
            linf_acc = (linf_outputs.argmax(dim=1) == y_test).float().mean().item()

        print(f"\nResults:")
        print(f"  Standard Transformer Test Accuracy: {std_acc:.3f}")
        print(f"  Linformer Test Accuracy:           {linf_acc:.3f}")
        print(f"  Accuracy Gap:                      {abs(std_acc - linf_acc):.3f}")

        self.results['std_train_losses'] = std_train_losses
        self.results['linf_train_losses'] = linf_train_losses
        self.results['std_acc'] = std_acc
        self.results['linf_acc'] = linf_acc

        return std_train_losses, linf_train_losses, std_acc, linf_acc

    def benchmark_scaling_behavior(self):
        """Benchmark 3: Scaling Behavior with Different k Values"""
        print("\n" + "=" * 60)
        print("Benchmark 3: Scaling Behavior with Different k Values")
        print("=" * 60)

        seq_len = 4096
        dim = 512
        heads = 8
        k_values = [64, 128, 256, 512, 1024]

        print(f"\nSequence Length: {seq_len}, Dimension: {dim}")

        results = []
        for k in k_values:
            print(f"\nk = {k}:")

            linf_attn = LinformerSelfAttention(dim, seq_len, k=k, heads=heads,
                                              one_kv_head=True, share_kv=True).to(DEVICE)
            x = torch.randn(1, seq_len, dim).to(DEVICE)

            # Warmup
            for _ in range(3):
                _ = linf_attn(x)
            synchronize_device()

            # Timing
            times = []
            for _ in range(10):
                start = time.time()
                _ = linf_attn(x)
                synchronize_device()
                times.append(time.time() - start)

            avg_time = np.mean(times)
            print(f"  Average Time: {avg_time:.4f}s")

            results.append(avg_time)

            del linf_attn, x
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

        self.results['k_values'] = k_values
        self.results['k_scaling_results'] = results

        return k_values, results

def plot_results(benchmark):
    """Create visualization of benchmark results"""
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Time Complexity
    if 'seq_lengths' in benchmark.results:
        ax = axes[0, 0]
        seq_lens = benchmark.results['seq_lengths']
        std_times = benchmark.results['std_attn_times']
        linf_times = benchmark.results['linformer_times']

        ax.plot(seq_lens, std_times, 'o-', label='Standard Attention', linewidth=2, markersize=8)
        ax.plot(seq_lens, linf_times, 's-', label='Linformer', linewidth=2, markersize=8)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Time Complexity Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

    # Plot 2: Memory Usage
    if 'seq_lengths' in benchmark.results:
        ax = axes[0, 1]
        std_mem = benchmark.results['std_attn_memory']
        linf_mem = benchmark.results['linformer_memory']

        ax.plot(seq_lens, std_mem, 'o-', label='Standard Attention', linewidth=2, markersize=8)
        ax.plot(seq_lens, linf_mem, 's-', label='Linformer', linewidth=2, markersize=8)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    # Plot 3: Training Loss
    if 'std_train_losses' in benchmark.results:
        ax = axes[1, 0]
        std_losses = benchmark.results['std_train_losses']
        linf_losses = benchmark.results['linf_train_losses']

        ax.plot(std_losses, 'o-', label='Standard Transformer', linewidth=2, markersize=6)
        ax.plot(linf_losses, 's-', label='Linformer', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Convergence', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # Plot 4: Scaling with k
    if 'k_values' in benchmark.results:
        ax = axes[1, 1]
        k_vals = benchmark.results['k_values']
        k_results = benchmark.results['k_scaling_results']

        ax.plot(k_vals, k_results, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('k (projection dimension)', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Linformer Scaling with k', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig('linformer_benchmark_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: linformer_benchmark_results.png")

    return fig

def main():
    """Main benchmark execution"""
    print("=" * 60)
    print("Linformer Performance Benchmark and Reproduction")
    print("=" * 60)

    # Device selection is already done at module level
    print(f"Using device: {DEVICE}")

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    try:
        # Run benchmarks
        benchmark.benchmark_attention_complexity()
        benchmark.benchmark_quality_comparison()
        benchmark.benchmark_scaling_behavior()

        # Generate visualizations
        plot_results(benchmark)

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if 'seq_lengths' in benchmark.results:
            max_seq_len = benchmark.results['seq_lengths'][-1]
            speedup = benchmark.results['std_attn_times'][-1] / benchmark.results['linformer_times'][-1]
            print(f"\nAt sequence length {max_seq_len}:")
            print(f"  Speedup: {speedup:.2f}x")

            # Memory reduction calculation with safety check
            std_mem = benchmark.results['std_attn_memory'][-1]
            linf_mem = benchmark.results['linformer_memory'][-1]
            if linf_mem > 0 and std_mem > 0:
                print(f"  Memory reduction: {std_mem / linf_mem:.2f}x")
            elif DEVICE == 'cpu':
                print(f"  Memory reduction: N/A (CPU memory measurement less accurate)")
            else:
                print(f"  Memory reduction: N/A (measurement error)")

        if 'std_acc' in benchmark.results:
            print(f"\nQuality Comparison:")
            print(f"  Standard Transformer Accuracy: {benchmark.results['std_acc']:.3f}")
            print(f"  Linformer Accuracy:            {benchmark.results['linf_acc']:.3f}")
            print(f"  Performance Gap:                {abs(benchmark.results['std_acc'] - benchmark.results['linf_acc']):.3f}")

        print("\n" + "=" * 60)
        print("Benchmark completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()