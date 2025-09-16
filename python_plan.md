你已经非常清晰地分析了 `spada-sim` 模拟器的核心逻辑。我们现在的目标是：

> **以 Python 语言重构一个适配你研究的稀疏加速需求的模拟器框架，参考 spada-sim 的结构设计与数据流模拟机制。**

下面我为你 **用 Python 逻辑重新梳理** 这个模拟器的大纲。重点突出模块职责划分、主流程结构，以及你可以如何模块化编写、便于后续扩展支持多种数据流和配置。

------

## 🧠 Python版本 Spada模拟器设计大纲

### 📦 模块划分（仿照 Rust 项目结构）

| 模块文件        | 职责说明                                              |
| --------------- | ----------------------------------------------------- |
| `main.py`       | 程序入口，加载配置，读取矩阵，初始化组件，启动模拟器  |
| `config.py`     | JSON 配置文件解析器                                   |
| `matrix.py`     | 处理稀疏矩阵的加载与封装（支持CSR格式）               |
| `storage.py`    | 模拟DRAM存储：CSR矩阵、向量存储、访问统计             |
| `simulator.py`  | 核心模拟器类，周期推进、调度、执行等主逻辑            |
| `scheduler.py`  | 调度器：任务生成与分配，任务状态跟踪                  |
| `pe.py`         | 处理单元（PE）类：stream buffer、乘法器、psum缓冲区等 |
| `adder_tree.py` | 加法树处理：跨 PE 结果合并                            |
| `cache.py`      | 模拟缓存层（可选）                                    |
| `utils.py`      | 工具函数，打印、debug、可视化等                       |

------

### 🧭 主流程（main.py）

```python
from config import load_config
from matrix import load_matrix
from storage import CSRMatrixStorage, VectorStorage
from simulator import Simulator

def main():
    config = load_config("config/config_1mb_row1.json")
    matrix_a, matrix_b = load_matrix("matrices/cari.mtx")

    dram_a = CSRMatrixStorage(matrix_a)
    dram_b = CSRMatrixStorage(matrix_b)
    dram_psum = VectorStorage()

    # Optional: 对 A 矩阵按非零行数排序以增强局部性
    if config["preprocess"]:
        dram_a.sort_by_row_length()

    # 初始化模拟器
    sim = Simulator(config, dram_a, dram_b, dram_psum)
    sim.execute()

    sim.report()

if __name__ == "__main__":
    main()
```

------

### 🧩 Simulator类结构（simulator.py）

```python
class Simulator:
    def __init__(self, config, dram_a, dram_b, dram_psum):
        self.config = config
        self.dram_a = dram_a
        self.dram_b = dram_b
        self.dram_psum = dram_psum
        self.exec_cycle = 0
        self.pes = [PE(config) for _ in range(config["pe_num"])]
        self.scheduler = Scheduler(config, self.pes, dram_a)
        self.adder_trees = [AdderTree(config) for _ in range(config["adder_tree_num"])]

    def execute(self):
        while not self.finished():
            for pe in self.pes:
                pe.tick(self.exec_cycle, self.scheduler, self.dram_b, self.dram_psum)

            for at in self.adder_trees:
                at.tick()

            self.scheduler.tick()
            self.exec_cycle += 1

    def finished(self):
        return self.scheduler.done() and all(pe.idle() for pe in self.pes)

    def report(self):
        print(f"Total cycles: {self.exec_cycle}")
        # 可打印 DRAM 读写统计、利用率等
```

------

### 🛠️ PE类结构（pe.py）

```python
class PE:
    def __init__(self, config):
        self.config = config
        self.stream_buffers = [[] for _ in range(config["lane_num"])]
        self.psum_buffers = []
        self.current_task = None
        self.drain_cycle = None

    def tick(self, cycle, scheduler, dram_b, dram_psum):
        if self.current_task is None:
            self.current_task = scheduler.assign_task(self)
            if self.current_task is None:
                return

        self.fetch_b_data(dram_b)
        self.multiply()
        self.collect_psum(dram_psum)

    def idle(self):
        return self.current_task is None and all(len(buf) == 0 for buf in self.stream_buffers)

    def fetch_b_data(self, dram_b):
        # 根据当前任务，从 dram_b 中读数据填入 stream buffer
        pass

    def multiply(self):
        # 取出 A 和 B，执行乘法，放入 psum buffer
        pass

    def collect_psum(self, dram_psum):
        # 将 psum buffer 的值写回 dram_psum
        pass
```

------

## ✅ 你可以立即做的事：

### 1. 基于上述框架，搭建最小可运行模拟器：

- 支持一个简单数据流（如 Gustavson Row-based）
- 加载 CSR 稀疏矩阵
- 每个 PE 模拟简单计算与部分和合并
- 统计周期与访存读写次数

### 2. 后续逐步加上扩展功能：

- 支持多数据流（InP / OutP / Row / Spada）
- 模拟 bandwidth、cache、pending latency 等
- 支持任务 tiling 与 partitioning
- 可视化结果：bar chart / Gantt 图

------

如果你愿意，我可以帮你：

- 搭建这个 Python 框架的基础代码
- 编写最小工作示例
- 或者给出接口文档/模块说明，便于你自己扩展

是否需要我帮你起草这一基础代码骨架？
