# E1-v1 现象门判定（FINAL，2026-09-16 冻结）

状态：**已完结并冻结**。不重抽 mask、不改门限、不回头修数。
本报告是 Stage I（SM-PLSeg / PA-HADFL）go/no-go 决策的正式记录。

## 协议

- 12 臂 = {random, longtail, sitelike, conditional} × mask seeds {0,1,2}，
  E1_v1 moderate packs（`ea4921f` 生成，SHA 校验，`docs/E1_V1_MASK_PACK_FREEZE.md`）。
- 训练：`sparq/e1/train_e1_arm.py` @ `8a4169d`——逐步镜像官方
  `train_PLSeg.py`（4×HADFL on clamped softmax、logit DistillationLoss(T=10) PSD、
  CO_Contrastive、xavier、patch 归一化）。此前批次（377527–378354）因手写损失
  偏离官方在 ~epoch 16 全 NaN、被 `c83b5cc` 记成假零损失空跑，全部作废；
  本判定只基于修复码重训结果。
- 评估：locked `imagesVal`（20 例），vendored 滑窗推理 + 官方 Dice
  （`tools/eval_e1_val.py`），checkpoint = `final_model.pth`，opt seed 42 全臂一致。
- 主指标（2026-09-16 预注册，先于本分析写定）：**DSC_difficult8**——WORD/PL-Seg
  官方困难组 {Gallbladder, Esophagus, Pancreas, Duodenum, Colon, Intestine,
  Adrenal, Rectum} 的均值；次指标 DSC_easy8、mean_fg16。
  random_s0-worst5 口径仅作 post-hoc 探索，不用于门判定。
- 明细：`artifacts/E1_ANALYSIS.md` / `.json`。

## 结果

| 臂 | difficult8 | easy8 | fg16 |
|---|---|---|---|
| random_s0/s1/s2（对照） | 0.5417 / 0.5337 / 0.5186 | 0.8151 / 0.7909 / 0.8121 | 0.6784 / 0.6623 / 0.6653 |
| longtail Δ（配对 pp） | −1.68 / −4.42 / +1.70（2 负 1 正翻转，均值 drop 1.47） | −1.66 / +1.89 / −3.56 混合 | −1.67 / −1.27 / −0.93（3/3 同向，std 0.30） |
| sitelike Δ | **+0.60 / +0.58 / +4.79（一致更好）** | 混合，s2 **−21.1** | +1.46 / −1.03 / **−8.15** |
| conditional Δ | −1.66 / −1.32 / +5.27 | 混合 | −0.48 / −1.52 / +1.87 |

判定（difficult8 主口径）：

- **longtail：UNRELIABLE**（均值 drop +1.47pp，但符号跨种子翻转，std 2.5pp）。
- **sitelike：NO PHENOMENON**（difficult8 上 structured 一致更好 +1.99pp；
  global −2.57pp 完全由 s2 的 **easy8 −21.1pp** 驱动——大器官标注被拿走，
  困难组反而受益）。
- **conditional：NO-GO**（|drop| 0.76pp < 1pp）。
- global mean_fg16：唯一稳定的信号是 longtail（3/3 同向、−1.29pp、std 0.30），
  落在预注册 1–2pp 灰区的下沿，够不到 2–3pp GO 门。

## 结论

1. **“structured missingness → 严重 global bias → 需要 propensity 全局纠偏”
   链条未被 E1-v1 支持。** Stage I PA-HADFL 开发暂停（2026-09-16 决策）。
2. E1 完成了它的诊断职能：平均 Dice 不崩，是因为器官级效应**互相对销**——
   longtail 下 Rectum −12.7pp 与 Adrenal +10.6pp 同现；sitelike_s2 灾难在
   easy8。**问题的真实形态是“有限标注预算分配给谁”，不是“全局偏差”。**
3. 这把研究主线推进到 **Stage II AOVA**：AOVA 要回答的“下一份 organ mask
   标给谁”，正是 E1 器官级对冲所暴露的杠杆。E1 结果本身可以写成论文的
   diagnostic/negative result（含 pre-specified groups 与 seed-variance 方法学）。
4. 已知限制（写论文时必须带）：moderate 档 pattern 对比度弱（freeze doc
   预告的风险兑现）；单臂单 opt-seed；imagesVal n=20；sitelike_s2 的 −8pp 说明
   单个 mask realization 可以很糟——但那是 acquisition 风险问题（→AOVA），
   不是 propensity 问题。

## 决策记录（2026-09-16）

| # | 决定 |
|---|---|
| 1 | E1-v1 结果冻结：不重抽、不改结论、不改门限 |
| 2 | propensity / PA-HADFL 暂停开发（保留代码占位，不删除） |
| 3 | 器官级分析改用预注册 difficult8/easy8；hard-5 仅作 post-hoc 探索 |
| 4 | longtail 扩种子（longtail s3/s4 + random s3/s4 共 4 臂）：可选支线，不挡主线 |
| 5 | severe 档只作 stress test（“多偏会崩”），不得反向用作 propensity 动机 |
| 6 | **主线转入 Stage II AOVA**，MVP 见 `docs/AOVA_MVP_PROTOCOL.md` |
