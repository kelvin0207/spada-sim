# Spada simulator
## Install
Please first install the [Rust toolchain](https://www.rust-lang.org/tools/install).

The simulator interacts with [python3](https://www.python.org/downloads/) for parsing sparse matrices:
```bash
$ python3 -m venv spadaenv
$ source spadaenv/bin/activate
$ pip install -U pip numpy scipy
```

## Build
```bash
$ cargo build --no-default-features
```

## Workload
The simulator accepts both MatrixMarket (.mtx) and numpy formatted matrices, with the latter ones packed as a pickle file (.pkl). The folder containing these matrices is specified in the config file under `config`.

## Simulate
First ensure the created python virtual environment is activated. The following command simulates SpGEMM of [cari](https://sparse.tamu.edu/Meszaros/cari) on Spada with the configuration specified in `config/config_1mb_row1.json`.
```bash
(spadaenv) $ ./target/debug/spada-sim accuratesimu spada ss cari config/config_1mb_row1.json
```
## Reference

If you use this tool in your research, please kindly cite the following paper.

Zhiyao Li, Jiaxiang Li, Taijie Chen, Dimin Niu, Hongzhong Zheng, Yuan Xie, and Mingyu Gao.
Spada: Accelerating Sparse Matrix Multiplication with Adaptive Dataflow.
In *Proceedings of the 28th International Conference on Architectural Support for Programming Languages and Operating Systems* (ASPLOS), 2023.

---

# SPADA 模拟器的解读分析

该项目是 SpGEMM（稀疏矩阵乘法）加速器模拟器，支持多种数据流架构（如 Ip、MultiRow、Op、Spada）。
核心目标是通过周期精确的模拟，评估不同数据流在稀疏矩阵乘法中的性能（如执行周期、存储访问等）。