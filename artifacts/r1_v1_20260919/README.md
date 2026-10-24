# R1_v1 中止记录（2026-09-19，用户指示取消排队任务）

数组 383503（16 臂）：任务 7–15 与 7 已按用户指示 scancel；任务 6 意外
FAILED（exit 120，gpu04，36 min，无 traceback——疑似节点级问题，未复现）。

## 结果状态

| 臂 | 状态 | fg16 @64/64 |
|---|---|---|
| random_s0（task 0） | **训练崩塌**（见下） | 6.3e-10（无效） |
| random_s1（task 1） | 完成 | best **0.7474** / final 0.7398 |
| random_s2（task 2） | 完成 | best **0.7342** / final 0.7325 |
| longtail_s0/s1/s2（task 3/4/5） | 完成时被中止？仍在 RUNNING（用户未明确处置） | — |
| sitelike_s0（task 6） | FAILED 120 | — |
| 其余 | 已取消 | — |

## 关键事实

1. **random_s1/s2 与桥接臂（同 pack、同 opt-seed 42）高度一致**：
   0.7474 / 0.7342 / 0.7394（桥），±0.7pp 内——参考配方在多数运行中可复现。
2. **random_s0 崩塌（新问题）**：验证从 epoch 5 起恒定 all17=0.0558
   （=1/17，全背景输出）、fg16=0.000000 直到 500 epoch；训练 loss 全程
   "正常"（~1.1–1.3）。train/eval 通路从头分裂，非有限值保护未触发
   （参数有限但崩坏），plateau(mode=max) 把崩塌值锁为 BEST——
   **best-checkpoint 机制在从头崩塌模式下完全失效**。
   同 pack 同 seed 的桥接臂正常 → 偶发（本批 4 次同配置运行 1 次崩）。
   机制候选（未定位）：eval 通路统计量（BN running stats 类）被早期
   极端 batch 破坏；与三任务同节点并发的关系未知。
3. **影响**：R1 的 E1 配对门在 random_s0 无效数字下不可判读；且任何
   future 运行都必须先解决"从头崩塌 + best 污染"（如：验证早停断言
   [连续 N 次 val fg16==0 即 FAILED]、best 下限保护、或改用 InstanceNorm
   排查）。

## 处置（按用户指示）

排队任务全部取消，不重跑。在跑的 longtail 三臂去留待用户明确。
random_s1/s2 数字有效并已入仓（本目录），供后续任何参考配方分析使用。
