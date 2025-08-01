// 这两行是Rust 特性标志，用于启用 Rust 标准库中尚未稳定的实验性功能
#![feature(drain_filter)]
// 为集合类型（如Vec、HashMap）提供drain_filter方法，用于过滤并移除元素（返回迭代器，可处理被移除的元素），类似 “边遍历边删除符合条件的元素”。
#![feature(hash_drain_filter)]
// HashMap专属的drain_filter实现（更高效）

// mod是 Rust 中定义模块的关键字（类似include），用于将代码拆分成独立的逻辑单元
// （类似其他语言的 “包” 或 “命名空间”），避免命名冲突，提高可读性。
// 每个mod 模块名;对应一个代码文件：比如mod adder_tree;
// 会去寻找adder_tree.rs文件（与main.rs同目录），该文件中包含adder_tree模块的具体实现。
mod adder_tree;
mod block_topo_tracker;
mod colwise_irr_adjust;
mod colwise_reg_adjust;
mod frontend;
mod gemm;
mod preprocessing;
mod py2rust;
mod rowwise_adjust;
mod rowwise_perf_adjust;
mod scheduler;
mod simulator;
mod storage;
mod util;

// use的作用是导入外部的类型、函数、模块，
// 避免每次使用时写完整路径（类似 Python 的from ... import ...）。

// 导入 Rust 标准库（std）中cmp模块的min函数（用于取两个值的最小值）
use std::cmp::min;

// 导入当前项目中gemm模块（对应gemm.rs）定义的GEMM类型
// （可能是一个 trait 接口，定义了稀疏矩阵乘法的核心行为）。
use gemm::GEMM;

// crate是 Rust 的关键字，代表 “当前项目”（Rust 中 “项目” 称为crate，是编译的基本单元）。
// crate::frontend表示当前项目下的frontend模块（对应frontend.rs）。
use crate::frontend::{parse_config, Accelerator, Cli, Mode, WorkloadCate};
use crate::preprocessing::sort_by_length;
use crate::py2rust::{load_mm_mat, load_pickled_gemms};
use crate::simulator::Simulator;
use crate::storage::{CsrMatStorage, VectorStorage};
use structopt::StructOpt;

fn main() {
    // 解析命令行参数
    // 命令行输入：./spada-sim (1)accuratesimu (2)spada (3)ss (4)cari (5)config/config_1mb_row1.json
    // 对应匹配：Cli结构里面 (1)Simulate Mode, (2)Accelerator, (3)Workload Category, (4)Workload Name
    // (5) Configuration File path, (6) preprocessing
    let cli: Cli = Cli::from_args();

    // 解析json文件的config
    let spada_config = parse_config(&cli.configuration).unwrap();
    
    //  加载矩阵数据，更具workload类型加载不同的矩阵
    let gemm: GEMM;
    match cli.category {
        WorkloadCate::NN => {
            gemm = load_pickled_gemms(&spada_config.nn_filepath, &cli.workload).unwrap();
        }
        WorkloadCate::SS => {
            let mat = load_mm_mat(&spada_config.ss_filepath, &cli.workload).unwrap();
            gemm = GEMM::from_mat(&cli.workload, mat);
        }
    };

    let a_avg_row_len = gemm.a.nnz() / gemm.a.rows();
    let b_avg_row_len = gemm.b.nnz() / gemm.b.rows();
    println!("Get GEMM {}", gemm.name);
    println!("{}", &gemm);
    println!(
        "Avg row len of A: {}, Avg row len of B: {}",
        a_avg_row_len, b_avg_row_len
    );

    // 执行周期精确模拟（核心逻辑）
    match cli.simulator {
        Mode::AccurateSimu => {
            // Cycle-accurate simulator.
            // Step1. 初始化存储（DRAM 中的矩阵）
            let (mut dram_a, mut dram_b) = CsrMatStorage::init_with_gemm(gemm);
            let mut dram_psum = VectorStorage::new();

            // Preprocessing.
            // Step2. 矩阵预处理，目的是优化数据局部性
            if cli.preprocess {
                let rowmap = sort_by_length(&mut dram_a);
                dram_a.reorder_row(rowmap);
            }

            // Step3. 确定默认块形状（数据流核心参数）
            let output_base_addr = dram_b.indptr.len();
            // Determine the default window & block shape.
            let default_block_shape = match cli.accelerator {
                Accelerator::Ip => spada_config.block_shape,
                Accelerator::MultiRow => [spada_config.block_shape[0], spada_config.block_shape[1]],
                Accelerator::Op => [spada_config.lane_num, 1],
                Accelerator::Spada => spada_config.block_shape,
            };
            
            // Step4. 初始化模拟器并执行模拟
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

            // Step5. 输出模拟结果
            let result = cycle_simu.get_exec_result();
            let a_count = cycle_simu.get_a_mat_stat();
            let b_count = cycle_simu.get_b_mat_stat();
            let c_count = cycle_simu.get_c_mat_stat();
            let exec_count = cycle_simu.get_exec_cycle();
            let cache_count = cycle_simu.get_cache_stat();

            println!("-----Result-----");
            println!("-----Access count");
            println!("Execution count: {}", exec_count);
            println!("A matrix count: read {} write {}", a_count[0], a_count[1]);
            println!("B matrix count: read {} write {}", b_count[0], b_count[1]);
            println!("C matrix count: read {} write {}", c_count[0], c_count[1]);
            println!(
                "Cache count: read {} write {}",
                cache_count[0], cache_count[1]
            );

            println!("-----Output product matrix");
            for idx in 0..min(result.len(), 10) {
                println!("{}", &result[idx]);
            }
        }

        _ => panic!("Unimplemented simulator {}", cli.simulator),
    }
}
