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

## 简介

该项目是 SpGEMM（稀疏矩阵乘法）加速器模拟器，支持多种数据流架构（如 Ip、MultiRow、Op、Spada）。核心目标是通过周期精确的模拟，评估不同数据流在稀疏矩阵乘法中的性能（如执行周期、存储访问等）。

目录结构

- 配置文件：`config/` 目录下存放配置文件（如 config_1mb_row1.json）

- 源代码：src/

  目录包含主要实现，关键模块包括：

  - storage.rs：实现稀疏矩阵的存储和读写操作
  - simulator.rs：模拟器核心逻辑
  - gemm.rs：稀疏矩阵乘法相关实现
  - scheduler.rs：调度器实现
  - block_topo_tracker.rs：块拓扑跟踪器

- 矩阵数据：`matrices/` 目录存放示例矩阵（如 `cari.mtx`）

---

## 程序执行流程

### Step 1：主函数入口(main.rs)

1. 解析命令行参数与json文件的配置，加载矩阵数据并打印出来

```rust
// 解析命令行参数
let cli: Cli = Cli::from_args();
// 解析json文件的config
let spada_config = parse_config(&cli.configuration).unwrap();
//  加载矩阵数据，更具workload类型加载不同的矩阵
let gemm: GEMM;
match cli.category {
    WorkloadCate::NN => {gemm = xx;}
    WorkloadCate::SS => {gemm = XX;}
};
```

2. 进入周期精确模拟（核心逻辑）

* 初始化存储（DRAM 中的矩阵）
* 矩阵预处理，优化A矩阵的数据局部性
* 确定默认块的形状
* 初始化模拟器并执行
* 输出结果

下面这一部分我们将逐个拆解周期精确模拟部分的核心逻辑

### Step 2: 初始化存储（storage.rs)

`main.rs` 通过调用下面的代码，初始化 DRAM

```rust
let (mut dram_a, mut dram_b) = CsrMatStorage::init_with_gemm(gemm);
let mut dram_psum = VectorStorage::new();
```

`CsrMatStorage`是基于**CSR 格式**的稀疏矩阵存储类型。

`init_with_gemm(gemm)`是其关联函数，从`gemm`中提取矩阵 A 和 B，转换为`CsrMatStorage`类型，并存入 “DRAM”（模拟器中模拟的内存）。主要存储了如下信息：

```rust
CsrMatStorage {
    data: gemm.a.data().to_vec(),
    indptr: gemm.a.indptr().as_slice().unwrap().to_vec(),
    indices: gemm.a.indices().to_vec(),
    read_count: 0,
    write_count: 0,
    remapped: false,
    row_remap: HashMap::new(),
    track_count: true,
    mat_shape: [gemm.a.shape().1, gemm.a.shape().0],
},
```

`VectorStorage::new()`创建一个新的向量存储（`dram_psum`），用于存储部分和（Partial Sum，稀疏矩阵乘法中的中间结果）。

```rust
VectorStorage {
    data: HashMap::new(),
    read_count: 0,
    write_count: 0,
    track_count: true,
}
```

我们发现，`CsrMatStorage`和`VectorStorage`都使用了`read_count`与`write_count`记录访存信息，通过`track_count`作为开关。

### Step 3: 矩阵预处理

`main.rs` 文件通过下面的代码对A矩阵进行重组

```python
if cli.preprocess {
    let rowmap = sort_by_length(&mut dram_a);
    dram_a.reorder_row(rowmap);
}
```

`sort_by_length(&mut dram_a)`：对`dram_a`（矩阵 A）的行按长度（非零元素数量）排序，返回一个`rowmap`（行索引映射表，记录排序后的行与原行的对应关系）。

`dram_a.reorder_row(rowmap)`：根据`rowmap`重新排序矩阵 A 的行（物理上调整行的顺序）。

### Step 4: **确定默认块形状（数据流核心参数）**

`main.rs` 文件通过下面的代码确定默认块

```rust
let output_base_addr = dram_b.indptr.len();
let default_block_shape = match cli.accelerator {
    Accelerator::Ip => spada_config.block_shape,
    Accelerator::MultiRow => [spada_config.block_shape[0], spada_config.block_shape[1]],
    Accelerator::Op => [spada_config.lane_num, 1],
    Accelerator::Spada => spada_config.block_shape,
};
```

**块形状（block_shape）是数据流的核心参数**,对应WA的 $$\alpha \times \beta$$ 规模，按 “块” 划分后流入 PE 计算。块的大小和形状直接决定了数据如何在存储和计算单元间流动（如行块、列块、混合块）。

### Step 5: **初始化模拟器并执行模拟**

`main.rs`通过下面代码初始化模拟器并执行

```rust
let mut cycle_simu = Simulator::new(
    spada_config.pe_num,
    spada_config.at_num,
    spada_config.lane_num,
    spada_config.cache_size,
    spada_config.word_byte,
    output_base_addr,
    default_block_shape,
    &mut dram_a,
    &mut dram_b,
    &mut dram_psum,
    cli.accelerator.clone(),
    spada_config.mem_latency,
    spada_config.cache_latency,
    spada_config.freq,
    spada_config.channel,
    spada_config.bandwidth_per_channel,
);

cycle_simu.execute();
```

`Simulator::new`是模拟器的构造函数，接收大量参数初始化模拟器状态（硬件配置、数据存储、数据流类型等）。

`cycle_simu.execute()`调用模拟器的执行方法，启动周期精确模拟（核心逻辑，后续会深入`Simulator`结构体分析）。

---

## 核心数据结构与硬件抽象

数据流的模拟依赖于对硬件组件的抽象，核心定义在 `src/simulator.rs` 中：

### Simulator

定义

```rust
pub struct Simulator<'a> {
    pe_num: usize,
    adder_tree_num: usize,
    lane_num: usize,
    fiber_cache: LatencyPriorityCache<'a>,
    pes: Vec<PE>,
    a_matrix: &'a mut CsrMatStorage,
    exec_cycle: usize,
    scheduler: Scheduler,
    adder_trees: Vec<AdderTree>,
    // Storage access latency related.
    pub a_pending_cycle: Vec<usize>,
    pub channel: usize,
    pub word_cycle_chan_bw: f32,
    // Debug info.
    pub drain_cycles: Vec<usize>,
    pub mult_util: Vec<f32>,
    pub active_cycle: Vec<usize>,
}
```



1. **PE（处理单元）**：`PE` 结构体是数据流处理的核心，包含：
   - `stream_buffers`：输入数据缓冲区（存储待处理的矩阵元素）。
   - `multiplier_array`：乘法器阵列（执行元素乘法）。
   - `sorting_network` 和 `merge_tree`：处理乘法结果的排序与合并（稀疏矩阵乘法中关键步骤）。
   - `task`：当前执行的任务（由调度器分配，包含数据流相关配置，如 `group_size`）。

2. **关键组件逻辑**：
   - `MultiplierArray`：实现矩阵元素的乘法，其 `multiply` 方法中通过 `group_size` 控制数据匹配逻辑（不同数据流可能有不同的分组策略）。
   - `SortingNetwork` 和 `MergeTree`：处理乘法结果的排序和累加（稀疏矩阵中需按列索引合并相同位置的结果），其延迟和并行度直接影响数据流的效率。

重点关注：`PE` 的 `idle` 方法（判断当前处理单元是否空闲）、`update_tail_flags` 方法（跟踪数据处理进度），这些逻辑反映了数据流的推进方式。


### **第四步：分析调度器与任务分配（数据流的驱动逻辑）**
不同数据流的核心差异体现在任务的调度方式上，对应 `src/scheduler.rs`（虽未完全提供代码，但可结合 `simulator.rs` 推测）：
- **任务（Task）**：每个任务包含矩阵块的范围、分组大小（`group_size`）等信息，决定了数据如何被分配到 PE 中处理。
- **调度策略**：调度器根据加速器类型（如 `Spada`）将矩阵划分为块，分配给不同 PE，并协调数据在 PE 间的流动（如行优先、列优先或自适应方式）。

结合 `simulator.rs` 中 `PE` 的 `task` 字段，可推断：不同数据流（如 `Ip` 与 `Spada`）的任务结构不同，导致数据加载、乘法、合并的顺序和并行度存在差异。


### **第五步：跟踪数据在存储与计算间的流动**
数据流的效率很大程度依赖于存储访问模式，相关逻辑在 `src/storage.rs` 和 `src/storage_traffic_model.rs` 中：
1. **存储访问**：`storage.rs` 中的 `request_read_scalars` 方法处理数据读取请求，根据缓存命中情况（`rowmap`）决定从缓存还是内存加载数据，不同数据流会有不同的访问模式（如是否连续访问、是否复用数据）。
2. **流量模型**：`storage_traffic_model.rs` 中的 `TrafficModel` 跟踪数据块的访问、复用和缓存行为，通过 `exec_trackers` 记录不同数据流的存储流量指标（如 `c_reuse`、`b_reuse` 复用率）。

重点关注：不同数据流如何通过 `block_shape`（块形状）影响数据块的划分，进而影响存储访问的局部性和效率。


### **第六步：聚焦特定数据流的差异化实现**
项目支持多种加速器架构（数据流），其差异化逻辑主要体现在：
1. **frontend.rs**：`Accelerator` 枚举定义了支持的数据流类型（`Ip`、`MultiRow`、`Op`、`Spada`），初始化时会根据此类型配置模拟器参数（如 `default_block_shape`）。
2. **simulator.rs**：`Simulator` 的 `execute` 方法（未完全展示）会根据加速器类型选择不同的任务调度和数据处理流程。例如，`Spada` 的自适应数据流可能会动态调整 `block_shape` 或 `group_size`，而 `Ip`（内积）数据流可能采用固定的块划分。
3. **调整策略**：`rowwise_adjust.rs`、`colwise_reg_adjust.rs` 等文件可能包含针对行优先、列优先数据流的调整逻辑（如块大小适配、延迟优化）。


### **总结阅读路径**
1. **整体流程**：`main.rs` → 理解程序入口和初始化逻辑。
2. **硬件抽象**：`simulator.rs` → 掌握 PE 及核心组件（乘法器、排序网络等）的工作方式。
3. **调度与任务**：`scheduler.rs`（结合 `simulator.rs` 的 `task` 处理）→ 分析不同数据流的任务分配策略。
4. **存储交互**：`storage.rs` + `storage_traffic_model.rs` → 理解数据访问模式与流量模型。
5. **差异化实现**：`frontend.rs`（加速器类型）+ 调整策略文件 → 对比不同数据流的核心差异。

通过以上步骤，可逐步理清不同数据流在模拟过程中的数据路径、调度逻辑和性能优化点。