# CAE Graph Surrogate

面向结构方案筛选的图代理模型工具链。输入结构参数、连接、材料、载荷与约束，
预测扭转刚度和节点位移，通过风险检查后排序，将 Top-K 与风险候选交给求解器复核。

内置数据由合成弹性网络生成，来源记为 `synthetic-only`；预测位移分量为 `u_z`，
刚度单位为 `N·m/rad`。商业 OptiStruct 适配器仍是扩展接口。

## 安装

Python 3.10+，支持 CPU。以下命令在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pytest -q
```

安装后可用 `cae` 或 `python -m src.cli` 调用。

## 批量筛选

仓库包含 100 个可复现的候选及配套 checkpoint，可直接运行：

```bash
cae screen \
  --checkpoint experiments/results/batch_physics.pt \
  --input data/synthetic/candidates.json \
  --policy configs/screening.yaml \
  --top-k 10 \
  --output outputs/screening/run-001
```

输出目录必须为空。每次运行产生独立的结果和版本信息。

| 文件 | 内容 |
|---|---|
| `summary.json` | 模型/数据/输入哈希、策略、分流计数、求解调用次数和阶段耗时 |
| `candidates.csv` / `predictions.json` | 刚度预测、风险原因、处理状态、代理排名及复核排名 |
| `displacements.npz` | 节点位移；按预测记录的 `displacement_key` 读取 |
| `verification_queue.json` | 需要求解复核的完整候选参数 |
| `verified.csv` | 求解成功的候选，按复核刚度排序 |
| `events.jsonl` | 候选分流、求解状态和错误记录 |

候选文件使用 `schema_version: 1`，每条记录包含唯一的 `sample_id`、`mesh_seed`、
`mode`、`grid_shape` 和 `parameters`。随机种子固定网格扰动与连接，输入不携带标签。
生成另一批候选：

```bash
cae candidates --config configs/demo.yaml --count 500 \
  --candidate-seed 2027 --ood-fraction 0.2 --output outputs/candidates-500.json
```

## 分流策略

`configs/screening.yaml` 控制风险阈值，独立于模型权重：

- 参数超出训练集经验范围、出现未覆盖的网格模式或节点规模，进入求解回退。
- 位移缺失、固定端位移非零、扭转角非正或 `|K_pred × θ / T − 1| > 0.25`，进入求解回退。
- 可信候选按预测刚度排序，前 K 个进入求解复核，其余记为 `screened_out`。
- 单批推理失败时，该批合法候选全部回退；单个求解失败记为 `solver_failed`，不进入最终排名。
- 非法参数、重复 ID 和不兼容的特征契约会被检查。模型/预处理契约不匹配时停止任务。

训练域只由训练输入计算；`domain_margin: 0.02` 表示在各参数经验跨度两端扩展 2%。
它是范围检查，不是经过校准的置信区间。默认要求位移输出，纯刚度模型需明确调整策略。

全量标签仅在显式审计模式下生成：

```bash
cae screen --checkpoint experiments/results/batch_physics.pt \
  --input data/synthetic/candidates.json --top-k 10 --audit \
  --output outputs/screening/audit-001
```

审计模式另写 `oracle.csv`，单独统计附加调用；仅当整个候选池标签及预测完整时计算
全池回归和排名指标。审计结果不会改变正常分流或复核排名。

## 训练与评估

```bash
cae generate --config configs/demo.yaml
cae baseline --config configs/demo.yaml
cae train --model gcn --config configs/demo.yaml
cae train --model meshgraphnet --config configs/demo.yaml
cae train --model multitask --config configs/demo.yaml
cae train --model physics --config configs/demo.yaml --output outputs/physics
cae evaluate --checkpoint outputs/physics/best.pt --split test
cae evaluate --checkpoint outputs/physics/best.pt --split ood
```

MeshGraphNet 使用 18 维节点特征、8 维边特征，经编码、消息传递后分别输出图级刚度
与节点位移。固定边界位移直接归零。训练使用 train-only normalization、物理损失
warm-up、转角下限、梯度裁剪和验证集早停。

`best.pt` 保存权重、归一化统计、特征顺序及单位、训练域、配置和数据哈希。
`history.json` 记录刚度/位移/物理损失、裁剪前梯度范数和每轮耗时；评估同时输出
MAPE、R²、P90/P95、位移误差、物理残差和排序指标。checkpoint 与 PyG 数据文件仅加载可信来源。
历史 v0 checkpoint 可继续评估；批量筛选要求重新训练生成 v1 契约。

## 实验记录

当前批处理模型：seed 42，train/val/test/OOD = 120/30/30/30，hidden dim 32，
3 轮消息传递，30 个 epoch，CPU。原始指标在
[`batch_training_metrics.json`](experiments/results/batch_training_metrics.json)。

| 测试集 | R² | MAPE | P90 相对误差 |
|---|---:|---:|---:|
| ID，30 个候选 | 0.9482 | 6.15% | 11.88% |
| 板厚 OOD，30 个候选 | 0.7049 | 10.29% | 21.97% |

固定候选池 seed 2026，共 100 个，其中 20 个板厚 OOD：

| 项目 | 本次运行 |
|---|---:|
| 排名选中 / 风险回退 | 10 / 35 |
| 正常筛选求解调用 | 45 / 100 |
| 筛选阶段调用减少 | 55% |
| 全量审计附加调用 | 55 |
| 全池 Spearman / 预测 Top-10 Recall | 0.9819 / 80% |
| 45 个复核候选覆盖真实 Top-10 | 10 / 10 |

运行记录见 [`batch_screening`](experiments/results/batch_screening) 和
[`batch_audit`](experiments/results/batch_audit)。这些数值来自本仓库合成样本的实际运行；
减少求解次数不等于同等比例的耗时收益，当前小规模合成求解器本身很快。
历史七模型对照保存在 [`comparison.csv`](experiments/results/comparison.csv)。

## 代码结构

| 模块 | 职责 |
|---|---|
| `src/data` | 合成网格、求解标签、构图、候选文件与特征契约 |
| `src/models` | MLP、GCN、MeshGraphNet 与多任务输出 |
| `src/losses` | 监督损失与扭矩/转角一致性 |
| `src/training` | 训练、评估、全池实验与基准计时 |
| `src/inference` | 风险检查、Top-K、回退和批处理结果 |
| `src/cae` | 求解器接口与合成适配器 |
| `tests` | 图完整性、批处理、数值稳定性、契约及故障分流测试 |

完整对照实验使用 `python scripts/run_experiments.py --config configs/demo.yaml`；
小规模链路验证使用 `configs/smoke.yaml`。技术细节见
[数据设计](docs/DATA_DESIGN_CN.md)、[模型设计](docs/MODEL_DESIGN_CN.md)和
[CAE 接入](docs/REAL_CAE_MIGRATION_GUIDE_CN.md)。

License: [MIT](LICENSE).
