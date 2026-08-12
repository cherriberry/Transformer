# Paper and source repository map

The three papers requested are:

| Model | Original paper | Author / official source | Runnable source used here |
|---|---|---|---|
| Memformer | [Memformer: A Memory-Augmented Transformer for Sequence Modeling](https://arxiv.org/abs/2010.06891), AACL-IJCNLP 2022 | [qywu/memformers](https://github.com/qywu/memformers) | `memformers` |
| Performer | [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794), ICLR 2021 | [google-research/google-research/performer](https://github.com/google-research/google-research/tree/master/performer) | [lucidrains/performer-pytorch](https://github.com/lucidrains/performer-pytorch), in `performer-pytorch` |
| Reformer | [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451), ICLR 2020 | [google/trax](https://github.com/google/trax) (the authors' Trax implementation) | [lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch), in `reformer-pytorch` |

The two `lucidrains` projects are PyTorch reimplementations used only because
the local benchmark is PyTorch-based. The official links above are retained
for provenance and paper-faithful reference.

The sandbox blocked direct `git clone`, so the three runnable repositories were
downloaded from GitHub's official codeload endpoint and unpacked locally. The
folders preserve the upstream source files and README/licenses, but do not
contain `.git` metadata.

## Reproduction command

Install the dependencies listed in `requirements-benchmark.txt`, then run:

```powershell
python run_benchmark.py --seq-lengths 128 256 512 --dim 256 --heads 8 --runs 3
```

The runner calls `universal_benchmark.py`, measures inference latency and CUDA
peak allocated memory (or process RSS on CPU), and writes
`benchmark_results.png` plus `benchmark_results.json`.
