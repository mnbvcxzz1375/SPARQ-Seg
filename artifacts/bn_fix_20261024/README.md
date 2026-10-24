# "装睡" bug 修复记录（2026-10-24）

## 根因（实锤，calib/diag_collapse.py + collapse_diag.json）

R1 random_s0 崩塌模型的诊断：eval 模式 100% 全背景（bg_fraction=1.0），
同一模型 BN 换 batch 统计立即恢复 17 类预测（91.8% vs 健康 91.4%）；
BN 体检发现 **running_var 含 Inf（any_nonfinite=true, var_max=Infinity）**。

机制：极端 batch 使某层激活方差溢出 float32 → running_var 存入 Inf →
eval 前向 `x/sqrt(Inf)=0` 输出坍缩为全背景；训练前向用 batch 统计不受
影响 → train loss 正常、非有限 loss/grad 保护不触发、崩塌值被
plateau(max) 锁成 BEST。偶发（当时 1/4 运行），取决于早期是否抽到极端 crop。

## 修复（calib/train_bridge.py）

1. `sanitize_bn()`：每 epoch 检查全部 BN running stats，非有限层
   `reset_running_stats()`（权重无损，统计随后续 epoch 重新积累），
   事件写入日志行 + `bn_reset_events.json` 审计。
2. 崩塌熔断：连续 3 次验证 fg16==0 → `DIAGNOSTIC_collapsed` + 退出。
3. best 保存下限：v_fg16 > 0.01 才允许写 best_model.pth。
4. DONE 前终检：结束时 BN 统计仍污染 → DIAGNOSTIC + 不写 DONE。

## 验证（4 × 60-epoch 探针，random_s0 pack，seed 42–45，jobs 573893–96）

| seed | 冻结值出现 | BN 重置 | DIAGNOSTIC | 60ep best fg16 |
|---|---|---|---|---|
| 42 | 0 | 0 | 0 | 0.254 |
| 43 | 0 | 0 | 0 | 0.276 |
| 44 | 0 | 0 | 0 | 0.241 |
| 45 | 0 | 0 | 0 | 0.199 |

四探针轨迹全部正常爬升、全部干净 DONE。60/500 epoch 的 fg16 水平与
健康桥接同时段一致。本批未触发重置（消毒是保护网，Inf 事件本身偶发）。

## 状态

修复已部署学校 `calib/train_bridge.py`。按用户指示**暂不全量重跑**
R1 剩余臂；下次任何训练自动带上保护。全量补跑待用户放行。
