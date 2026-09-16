# AOVA MVP 协议（Stage II 第一轮，2026-09-16 立项）

## 要回答的唯一问题

> 在相同 mask budget 下，主动选 `(volume, organ)` 对，能不能稳定打败 random？

答不了这个问题，就不上 Fisher / kernel / acquisition-model 等任何复杂件。

## 出发点（承接 E1-v1 冻结结论，`docs/E1_V1_VERDICT.md`）

- E1 显示 global bias 不成立，但器官级效应巨大且互相抵消（Rectum −12.7 vs
  Adrenal +10.6；sitelike_s2 的 −21pp easy8 灾难）→ 真实杠杆是**预算分配**。
- AOVA 的评估协议直接继承 E1 的：locked `imagesVal` 20 例、vendored 滑窗
  （`tools/eval_e1_val.py`）、预注册主指标 **DSC_difficult8** + easy8 + fg16、
  opt seed 42、`train_e1_arm.py`（官方镜像码）零改动——增强后的标签以同合同
  mask pack 喂入既有 fail-closed 管线（`tools/aova_select.py` 生成 npz +
  manifest + SHA）。

## Phase A：single-shot acquisition（本轮）

- 初始标签：E1_v1 `random_moderate_s0`（2/16，SHA 冻结）。
- 预算阶梯：+2（→4/16）与 +4（→6/16），与 PL-Seg 官方 2/4/6/16 曲线对齐；
  **先只跑 4/16 一档**，有信号再补 6/16。
- Teacher（打分模型）：E1 `random_moderate_s0` 的 `final_model.pth`——已存在，
  **零新训练**。给每个未标注 `(v,c)` 算平均预测熵（体积内 argmax 区域，空则全卷）。
- 4 个选择策略（`sparq/active/strategies.py`）：
  1. `random`——uniform over un-annotated pairs；
  2. `class_balanced`——按 `1/(coverage+ε)` 加权的随机（先补欠标注器官）；
  3. `entropy`——teacher 熵 top-K；
  4. `entropy_x_coverage`——`entropy × (1 − coverage)`（route wiki MVP sanity）。
- 每策略 → 1 个增强 pack → 1 次训练 → 1 次评估。Phase A 预算 = 4 臂训练
  （random/class_balanced 不需要熵分数，可立即开跑；entropy 两臂等打分工具）。
- 对照：同预算随机选择本身就是 random 策略臂，不另设。

## 判定规则

- **先决条件**：`entropy_x_coverage` 相对 `random` 在 difficult8 上有提升且
  fg16 不掉——否则所有基于 teacher entropy 的打分路线（含 Fisher）都不成立，
  回头重想 acquisition score。
- `class_balanced` vs `random`：分离"纯覆盖率均衡"与"实例不确定性"两个来源。
- 过线 → Phase B：多轮序贯（acquisition model 逐轮更新）、B_90%FSL、
  product kernel Fisher；不过线 → AOVA 论文故事改为 risk/robustness
  （E1 的 sitelike_s2 型单实现灾难），不做通用 acquisition。

## 实现清单

- [x] `sparq/active/strategies.py` + 测试（9/9 本地）
- [x] `tools/aova_select.py`（增强 pack 生成器，manifest 合并式更新）
- [x] `tools/aova_score_entropy.py`（teacher 熵打分；2026-09-16 已跑，
      teacher=random_s0 final_model，100 卷 ~35 分钟，4090D）
- [x] 4/16 四臂训练已提交（2026-09-16，学校 4090D；B=200 全局新增 mask，
      per-volume 2→8/9 不均匀分布，策略特征显著）：
      `random_b200_r0` / `class_balanced_b200_r0` / `entropy_b200_r0` /
      `entropy_coverage_b200_r0`，out=`runs/AOVA_v1/<arm>`，
      packs=`annotation_masks/WORD/AOVA_v1/`
- [ ] 四臂评估（train DONE 后 `eval_e1_val.py` 同管线）+ 先决条件判定
- [ ] Phase B 门判定与预算阶梯扩展

已知教训：aova_select 早期版本覆盖写 manifest 导致两臂 KeyError（380511/
380539），已改合并式更新 + 回归测试，重提后四臂齐跑。

## 边界（继承设计锁）

- 不读 hidden `full_label` 进训练路径；选择器只能看 图像/teacher 预测/已有
  标注计数，不能看未标注器官的真值。
- AOVA packs 是新资产（`annotation_masks/WORD/AOVA_v1/`），**不改 E1_v1**。
- `inclusion_prob` 语义：AOVA 的 π 不是 IPW 用途，manifest 标
  `design="acquisition"`，沿用 uniform 近似仅为一键复用下游合同校验。
