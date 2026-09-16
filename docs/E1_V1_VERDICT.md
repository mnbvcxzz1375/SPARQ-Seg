# E1-v1 现象门判定（INTERIM — 2026-09-16 审计后降级，待校准实验后终判）

状态：**探索性结果，可保留；不作路线终判。**
2026-09-16 外部源码审计（REVIEW.md，快照 `0a6d573`/`8a4169d`）判定 **AUDIT HOLD**：
本文档原有的"propensity 路线 NO-GO"结论证据不足，降级为 interim。
下方判定与数字全部保留为原始观察（不重抽、不覆盖），但其**解释边界**见文末"审计限制"。

> 一句话：E1 没有证明 propensity 有必要，但也没能证明它没必要——当前实现相对
> 官方 PL-Seg 有训练配方与坐标约定的实质偏离，random 基线（0.6687）与此前锚点
> （~0.7598）的差距尚未归因。**先校准 baseline，再谈路线。**

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
- 主指标：mean_fg16（global mean Dice over 16 前景器官）为**原始预设 primary**。
  2026-09-16 因见器官级效应，追加 **DSC_difficult8 / easy8**（WORD/PL-Seg 官方
  困难组）作为**外部定义分组的敏感性分析**——注意：这不是本实验的事前预注册
  （fg16/hard5 结果在此之前已产出），只是分组边界取自官方论文而非事后挑点，
  审计据此纠正了"预注册"措辞。random_s0-worst5 仅作 post-hoc 探索。
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
  落在 1–2pp 灰区的下沿，够不到 2–3pp GO 门。

## 初步观察（原判定，解释边界见下节）

1. 在**本 wrapper 的训练配方下**，"structured missingness → 严重 global bias →
   需要 propensity 全局纠偏"链条未出现预期现象：global mean_fg16 唯一稳定信号是
   longtail（−1.29pp，3/3 同向）；conditional/mean 效应不成立；sitelike 均值由
   单种子驱动。**这些观察保留，但不构成对 propensity 路线的可靠 NO-GO 终判。**
2. 器官级效应远大于全局效应且互相对销（longtail Rectum −12.7 与 Adrenal +10.6
   同现；sitelike_s2 的 −8.15pp 集中在 easy8 −21.09）——提示值得研究
   器官级标注分配（AOVA 动机）。注意（审计 F10）：组均值分解是确定算术，
   但**不能单独识别成因**（标注分布 / 有效监督 / 优化塌陷均未排除）。

## 审计限制（AUDIT HOLD，2026-09-16；终判前必须解决）

| # | 事实（已核实） | 对解释的影响 |
|---|---|---|
| F3 | **训练窗口 128×128×96（D,H,W）≠ 官方/评估滑窗 96×128×128**。学校 vendor `val3D.py`（md5 2a47683e）实测按 `patch_size[2]` 切 D 向；WORD z spacing 2.5mm/xy 0.98mm，物理视野差 z +33%、W −25% | 训练/评估 mismatch + 与官方锚点不同配方；可能解释部分基线差距 |
| F2 | plateau `mode=min` 跟训练 loss（官方 `mode=max` 跟验证 Dice）；batch=1×200 样本/epoch vs 官方 4×25；grad clip 5 额外；PSD ramp 权重随 epoch 升 yet 参与 plateau 控制 | 全部 12 臂共享该配方，臂间比较仍自洽；但结果不能外推为"原版 PL-Seg 的能力边界" |
| 锚点 | random 均值 0.6687 vs W3 记录 0.7598：即使 BG=1，all17 上限 0.688——**口径差不能解释**，需对齐 checkpoint 类型/mask realization/evaluator/病例清单后归因 | 基线未校准前，任何"偏差不严重"的结论不稳 |
| F4 | conditional 三臂在本地机（与九臂异环境），s0 从 epoch 349 非等价续训（HADFL history/RNG/scheduler 未恢复） | conditional 的 −0.04pp 混入环境与续训混杂，解释受限 |
| F1 | Preflight 的 GO 基于 TinySeg/MiniHADFL 替身（且替身 PSD 恒零），未认证真实训练链 | "无泄漏"仅有代码审查级证据，无运行时认证 |
| F7 | conditional 三 seed 同时换机制（W,a 随 seed）与抽样；sitelike 组=`arange%3` 无域差异验证；random(rng.choice) 与 structured(Gumbel) 同 seed 不构成严格配对（均匀权重 seed0 下仅 1/100 行相同） | "pattern 主效应"的方差解释要带此限制；sitelike ≠ 已验证的跨医院域偏移 |
| F5 | vendor 源码不在 git 内（本次已收 md5：unet3d d27330d1、sing_class_loss 3ae8ab05、transforms 3608c510、val3D 2a47683e）；school 无 .git 时 code-pin 跳过 | 已在 evidence 收集中补救 |
| F8/F9 | AOVA 选择器不核对 entropy volume_ids（已修复：ids 门控+重排+防覆盖，测试 12/12）；**当前四臂 pack 已验证对齐，pilot 数据不废** | 扩预算前已修 |

**终判前置（三步校准，全部完成前不重跑 12 臂、不扩大 AOVA 预算）**：
① 每臂证据收集 + vendor/环境指纹（本轮执行，见 `artifacts/audit_evidence/`）；
② 真实链测试：train_step 抽为共享函数，真模型反事实 + 训练/评估实形打印 +
双 evaluator 对照同一 checkpoint；
③ **MCAR 桥接实验**：官方配方（96×128×128 窗口、batch=4、plateau 跟验证 Dice）
+ 一个冻结 random mask，检验能否回到 ~0.76 锚点——过桥才能谈基线校准，
然后才决定 E1 是否补跑与 AOVA 是否扩。

AOVA 四臂处理：pack 对齐已验证，训练**作为 pilot 继续**（不杀不删），但其
结论自动继承 F2/F3 限制；任何"entropy×coverage 打败 random"的先决条件判定
只在 pilot 层面解读，扩预算等桥接实验之后。

## 决策记录（2026-09-16 上午原记录 + 同日审计后修订）

原决策（上午，基于"门未过即转向"）：
1. E1-v1 结果冻结；2. PA-HADFL 暂停；3. difficult8 作为主分析口径；
4. longtail s3/s4 可选；5. severe 只作 stress test；6. 主线转 AOVA。

**审计后修订（同日，本文件生效版本）**：
| # | 决定 | 变化 |
|---|---|---|
| 1 | E1-v1 pack/结果**不重抽不覆盖**，但地位从"终判证据"降为"探索性 pilot" | 修订 |
| 2 | PA-HADFL 维持暂停，**AOVA 扩预算同步冻结**（先校准后投入） | 新增 |
| 3 | difficult8 改称外部分组敏感性分析，primary 回到 fg16，记录修订时间 | 修订 |
| 4 | longtail s3/s4 与严重度问题一律排到桥接实验之后 | 修订 |
| 5 | severe 只作 stress test | 不变 |
| 6 | AOVA 四臂继续为 pilot（pack 对齐已验证、F8 已修）；扩预算等桥接 | 修订 |
| 7 | **当前最高优先 = 三步校准**（证据收集→真实链测试→MCAR 桥接）| 新增 |
