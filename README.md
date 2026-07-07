# FMGAD — local_prior 极性校准精简版

基于 `/home/yehang/0703coex/submission` 精简而来，**原仓库不做任何改动**。

本版本只保留一种无监督极性机制：**用 `local_prior`（节点特征与邻居均值之差的 L2 范数）作为唯一探针**，通过 rank consensus 决定主分数是否 flip。

## 与完整版的区别

| 项目 | 完整版 submission | 本精简版 |
|------|-------------------|----------|
| 极性探针 | 5 个（ae_self, flow_residual, latent_energy, local_prior, nk_prior） | **仅 local_prior** |
| 极性开关 | `polarity_adapter: consensus_rank \| none` | `polarity_enabled: true \| false` |
| 分数模式 | 多种 score_mode / 环境变量覆盖 | 固定：Flow 重建误差 + 图平滑 + 极性校准 |
| 开发脚本 | scripts/dev/, ablation, results/ | 已移除 |

## 极性机制

1. 预计算 `local_prior` = ‖xᵢ − mean(neighbors)‖₂  
2. 主分数（Flow Matching 重建误差，可选图平滑）转秩  
3. 与 `local_prior` 秩的相关性 ≥ 0.70 → keep，否则 flip  
4. 最终分数 = 0.90 × 定向主分数秩 + 0.10 × local_prior 秩  

全程不使用标签。

## 快速开始

```bash
cd /home/yehang/fmgad_local_prior

# 单数据集单 seed
python main_train.py --config configs/books.yaml --seed 42 --device 0

# 或使用 wrapper
python scripts/run_single.py --dataset books --seed 42 --device 0 --deterministic
```

环境变量：

- `FMGAD_MODEL_ROOT`：checkpoint 目录（默认 `./models`）
- `FMGAD_RUN_TAG_SUFFIX`：多进程隔离 checkpoint（如 `seed42`）
- `FMGAD_REUSE_CHECKPOINTS=1`：跳过训练，复用已有权重
- `FMGAD_INFERENCE_SEED`：推理噪声种子（默认 1729）

## 配置

各数据集 YAML 位于 `configs/`。极性相关字段：

```yaml
polarity_enabled: true
polarity_consensus_threshold: 0.70
polarity_consensus_score_weight: 0.90
```

关闭极性校准：`polarity_enabled: false`

## 测试

```bash
python tests/test_consensus_polarity.py
```

## 文件结构

```
main_train.py          # 训练/评估入口
model.py               # ResFlowGAD 模型
utils.py               # local_prior + 秩共识校准
auto_encoder.py        # 图自编码器
encoder.py             # 双残差特征
flow_matching_model.py # Flow Matching
FMloss.py              # FM 损失
configs/               # 五数据集配置
scripts/run_single.py  # 单 run wrapper
tests/                 # 极性单元测试
```

## 依赖

见 `requirements.txt`（Python 3.8+，PyTorch，PyGOD，PyG）。
