# MCAR 桥接结果（2026-09-17，job 381564）

一条对照臂，回答审计设定的校准问题：**这套代码栈在参考配方下能否得到
可信的 baseline**。通过标准（按审计定义）："实现、数据、评估和参考配方
都能明确对齐，结果差距能够解释"——**不是**"必须达到 0.76"。

## 运行身份

- mask：冻结 E1_v1 `random_moderate_s0`（**新 mask 校准**，非 W3 重放）
- 配方：D96×H128×W128 官方裁剪、batch=4（25 更新/epoch×500=12,500 更新）、
  plateau mode=max 跟验证 all17-Dice（每 2 epoch，stride 128/96 选模）、
  Adam(0.9,0.99) lr 0.01、无 clip、best/final 双存
- 最终比较：best 与 final 都按**锁定 64/64 evaluator**重评（20 例同集合）
- 完整性：无 DIAGNOSTIC、500 epoch 零跳过、运行 12h37m COMPLETED

## 三方对照（全部 fg16 @ stride 64/64，同 20 locked cases）

| 臂 | 配方 | mask | checkpoint | fg16 |
|---|---|---|---|---|
| 旧 E1 random_s0 | 旧（D128×128×96, batch1×200/ep, plateau-min-on-train-loss, clip5） | E1 random_s0 | final | 0.6784 |
| **桥接** | **参考配方** | E1 random_s0 | final | **0.7319** |
| **桥接** | 参考配方 | E1 random_s0 | best | **0.7394** |
| W3 | 官方原实现 | 官方 16r uniform | best | 0.7598 |

## 读法（严格限定的结论）

1. **恢复参考训练配方后，同 mask 同评估下 baseline 提升 +5.3pp（final 口径）
   / +6.1pp（best 口径）。** 这是整套配方修正的综合效果；patch 失配的独立
   贡献未被分离（如需归因，另做只改 patch 的控制臂，非当前前置）。
2. **本次运行中 best 与 final 相差 0.75pp**——与 W3 那次运行的 0.84pp 同
   量级，是这类运行的典型波动，不是固定常数。
3. **桥 best 与 W3 best 仍差 2.0pp**，候选来源（未分离）：mask realization
   不同（官方 16r vs E1 random pack；E1 三个 random seeds 间本身 ±0.8pp、
   AOVA random_b200 与桥同为 4/16 vs 2/16 不同预算不可直接比）、标签视图
   字节路径（预制 labelsTr_2 vs labelsTr_All+mask 动态应用）、其余实现微差。
   该差距在 mask-to-mask 方差的合理范围内，不构成"代码仍错"的证据。

## 对 E1/AOVA 既有数字的含义

- E1-v1 的 12 臂数字保留为"旧配方 wrapper 下的探索性结果"；**其绝对值不可
  与 W3/桥接口径混用**。pattern 间的配对比较要引用时必须带上 F2/F3 限制。
- AOVA pilot 同理（+7.1pp 等数字是旧配方口径）。
- 任何新实验（E1 补跑、AOVA 重评、propensity 对照）都应基于
  `calib/train_bridge.py` 的参考配方与共享 train_step（等价性/反事实均已
  认证）。

## 证据

`artifacts/calib_20260916/bridge/`：`final_eval_stride64.json`（双 checkpoint
64/64 全量）、`train.log`（500 epoch，零 DIAGNOSTIC）、`run_manifest.json`
（配方、断言、源码/资产 SHA、软件环境）、`DONE`。
