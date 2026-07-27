# RALFlow-GAD / FMGAD 五数据集配置表

来源：`~/newbase/better/configs/*.yaml`（当前仓库最佳配置）。  
论文对应：*RALFlow-GAD: Residual-Augmented Latent Flow …*（正文 Sec. 3，附录 Appendix B；主表为 **Table 7**）。  
说明：你附的 `AAAI_2026_Press_Formatting_Instructions_…pdf` 是 AAAI 排版说明，不含方法超参，故下列「论文位置」均指向方法稿而非该排版 PDF。

「—」表示 yaml 未写明，代码使用默认值（见 `fmgad/cli.py`）。

来源说明：五数据集关键超参已按 **AutoGAD 风格 CST**（无标签）搜索覆盖；`disney.weight` 现为 **1.25**（原 2.5）。

| 配置项 (yaml) | 含义 | 论文位置 / 符号 | weibo | reddit | enron | books | disney |
|---|---|---|---:|---:|---:|---:|---:|
| `weight` | 条件 FM / prototype 融合权重 | Sec. 3.2 | 0.0 | 0.0 | 0.0 | 1.25 | **1.25** |
| `ae_alpha` | 图自编码器属性/结构重建权重；=1 时关闭结构项 | Sec. 3.1 式 \(L_{\mathrm{AE}}\) 中 \(\alpha_{\mathrm{AE}}\)；App. B.2 / **Table 7** | 1.0 | 1.0 | 1.0 | 0.75 | 0.8 |
| `ae_lr` | 图自编码器学习率 | App. B.2 / **Table 7**（实现细节，正文无独立符号） | 0.005 | 0.003 | 0.003 | 0.015 | 0.05 |
| `ae_dropout` | 图自编码器 dropout | App. B.2 / **Table 7** | 0.1 | 0.3 | 0.2 | 0.25 | 0.3 |
| `flow_t_sampling` | ~~可配~~ → **代码固定** `logit_normal` | Sec. 3.2 | — | — | — | — | — |
| `use_score_smoothing` | ~~可配~~ → **代码固定开启** | Sec. 3.3 | — | — | — | — | — |
| `score_smoothing_alpha` | 邻居平滑系数 \(\mu\)（\(s=(1-\mu)s^{\mathrm{raw}}+\mu\,\mathrm{mean}(s^{\mathrm{raw}})\)） | Sec. 3.3 中 \(\mu\) | **0.5** | **0.5** | **0.45** | **0.5** | 0.35 |
| `use_virtual_neighbors` | 是否启用 DB-LVN 虚拟邻居 | Sec. 3.1 **DB-LVN**；Table 3 消融 *w/o Virtual Neighbor* | true | true | true | **false** | true |
| `virtual_degree_threshold` | DB-LVN 度阈值 \(b\)（低于 \(b\) 的节点补虚拟邻居） | Sec. 3.1 阈值 \(b\)；App. B.2 门控 \(\gamma_i=\sigma((d_i-b)\kappa)\) | — (默认 5) | — (默认 5) | — (默认 5) | —（已关 virt） | **4** |
| `virtual_k` | 低度节点补充的 top-\(k\) 潜空间近邻数 | Sec. 3.1 DB-LVN「top-\(k\) latent nearest neighbors」 | — (默认 5) | — (默认 5) | — (默认 5) | —（已关 virt） | **6** |
| `polarity_enabled` | 是否启用无标签分数极性校准 | Sec. 3.4；App. **B.5**；Table 3 消融 *w/o Polarity* | true | true | true | true | true |
| `polarity_consensus_threshold` | local_prior 秩共识门限：相关 ≥ 该值则 keep，否则 flip | Sec. 3.4 / App. B.5（实现为 local_prior 共识；论文表述为 confidence-weighted gate） | 0.7 | 0.65 | 0.7 | 0.7 | 0.7 |
| `polarity_consensus_score_weight` | 最终分中主分数秩的权重（其余为 local_prior 秩） | Sec. 3.4 校准后最终分；实现默认约 0.9×score + 0.1×prior | 0.9 | **0.7** | 0.95 | **0.95** | 0.95 |
| `num_trial` | 同 seed 内集成 trial 次数 | 实现/评估设置；**Table 7** 未单列（当前均为 1） | 1 | 1 | 1 | 1 | 1 |
| `hid_dim` | 隐层维度；`null` 表示按实现默认 | App. B.2 实现细节 | null | null | null | null | null |

## 与论文 Table 7 的差异（便于写论文时核对）

| 数据集 | 当前 yaml 相对 Table 7 / 旧最佳的主要变化 |
|--------|----------------------------------|
| **books** | `score_smoothing_alpha`→**0.5**；`use_virtual_neighbors`→**false**；`polarity_consensus_score_weight`→**0.95** |
| **reddit** | `score_smoothing_alpha`→**0.5**；`polarity_consensus_score_weight`→**0.7** |
| **enron** | CST：`score_smoothing_alpha`→**0.45**；`residual_scale=15.0`（Table 7 为 20.0） |
| **disney** | CST：`weight` 2.5→**1.25**；显式 `virtual_degree_threshold=4`, `virtual_k=6` |
| **weibo** | CST：`score_smoothing_alpha`→**0.5** |

## 近期调参 5-seed 结果（仅 books / reddit 本次重跑）

| 数据集 | 配置文件 | AUROC (mean±std) | AP (mean±std) |
|--------|----------|------------------|---------------|
| books | `books.yaml` | 0.6239±0.0157 | 0.0323±0.0031 |
| reddit | `reddit.yaml` | 0.5735±0.0157 | 0.0385±0.0027 |
| weibo / enron / disney | 对应 yaml | 本次未重跑 | 本次未重跑 |

种子：`{0,1,2,3,42}`（与论文 App. B.2 / Table 8 一致）。标准差为样本标准差（\(n=5\), ddof=1）。
