# E1 phenomenon-gate paired analysis (v3)

primary per DESIGN_LOCKS: mean_fg16; DSC_difficult8 = external-definition grouping (WORD official difficult group) added as sensitivity analysis 2026-09-16 (post fg16 results; see E1_V1_VERDICT.md audit constraints). cases=20 locked imagesVal; missing arms: none

| arm | difficult8 | easy8 | fg16 |
|---|---|---|---|
| random_moderate_s0 | 0.5417 | 0.8151 | 0.6784 |
| random_moderate_s1 | 0.5337 | 0.7909 | 0.6623 |
| random_moderate_s2 | 0.5186 | 0.8121 | 0.6653 |
| longtail_moderate_s0 | 0.5249 | 0.7985 | 0.6617 |
| longtail_moderate_s1 | 0.4895 | 0.8098 | 0.6497 |
| longtail_moderate_s2 | 0.5356 | 0.7766 | 0.6561 |
| sitelike_moderate_s0 | 0.5477 | 0.8383 | 0.6930 |
| sitelike_moderate_s1 | 0.5395 | 0.7646 | 0.6521 |
| sitelike_moderate_s2 | 0.5665 | 0.6012 | 0.5838 |
| conditional_moderate_s0 | 0.5251 | 0.8222 | 0.6736 |
| conditional_moderate_s1 | 0.5205 | 0.7737 | 0.6471 |
| conditional_moderate_s2 | 0.5712 | 0.7968 | 0.6840 |

## longtail vs random (paired by mask seed)
- DSC_difficult8 (sensitivity): UNRELIABLE: mean drop +1.47pp but signs flip across seeds (std 2.50pp) {'mean_drop_pp': 1.466, 'seed_std_pp': 2.503, 'deltas_pp': [-1.678, -4.42, 1.699], 'n_pairs': 3, 'sign_consistent': False}
- easy8: UNRELIABLE: mean drop +1.11pp but signs flip across seeds (std 2.26pp) {'mean_drop_pp': 1.108, 'seed_std_pp': 2.257, 'deltas_pp': [-1.656, 1.89, -3.557], 'n_pairs': 3, 'sign_consistent': False}
- **mean_fg16 (primary): GRAY: drop +1.29pp in -2.0..-1.0pp band, consistent** {'mean_drop_pp': 1.287, 'seed_std_pp': 0.302, 'deltas_pp': [-1.667, -1.265, -0.929], 'n_pairs': 3, 'sign_consistent': True}
- per-organ delta (difficult8, pp): {'Gallbladder': 1.24, 'Esophagus': -5.027, 'Pancreas': 0.26, 'Duodenum': -4.163, 'Colon': -2.303, 'Intestine': 0.404, 'Adrenal': 10.588, 'Rectum': -12.727}
- largest drop on seed1 (longtail_moderate_s1), difficult8 per-organ (pp): {'Gallbladder': -14.707, 'Esophagus': -3.682, 'Pancreas': 10.519, 'Duodenum': -16.033, 'Colon': -3.5, 'Intestine': 0.985, 'Adrenal': -3.656, 'Rectum': -5.282}

## sitelike vs random (paired by mask seed)
- DSC_difficult8 (sensitivity): NO PHENOMENON: structured BETTER by 1.99pp, sign-consistent (std 1.98pp) {'mean_drop_pp': -1.991, 'seed_std_pp': 1.979, 'deltas_pp': [0.602, 0.581, 4.79], 'n_pairs': 3, 'sign_consistent': True}
- easy8: UNRELIABLE: mean drop -7.13pp >= gate but seed std 10.07pp exceeds it (single-seed-driven risk; audit F6) {'mean_drop_pp': 7.135, 'seed_std_pp': 10.072, 'deltas_pp': [2.318, -2.633, -21.089], 'n_pairs': 3, 'sign_consistent': False}
- **mean_fg16 (primary): UNRELIABLE: mean drop -2.57pp >= gate but seed std 4.07pp exceeds it (single-seed-driven risk; audit F6)** {'mean_drop_pp': 2.572, 'seed_std_pp': 4.073, 'deltas_pp': [1.46, -1.026, -8.149], 'n_pairs': 3, 'sign_consistent': False}
- per-organ delta (difficult8, pp): {'Gallbladder': 0.995, 'Esophagus': -2.46, 'Pancreas': 3.629, 'Duodenum': 1.14, 'Colon': 2.499, 'Intestine': 4.335, 'Adrenal': 6.752, 'Rectum': -0.963}
- largest drop on seed1 (sitelike_moderate_s1), difficult8 per-organ (pp): {'Gallbladder': -13.124, 'Esophagus': -1.938, 'Pancreas': 11.749, 'Duodenum': -2.188, 'Colon': 2.155, 'Intestine': 4.905, 'Adrenal': -6.272, 'Rectum': 9.36}

## conditional vs random (paired by mask seed)
- DSC_difficult8 (sensitivity): NO-GO: |drop| 0.76pp < 1.0pp {'mean_drop_pp': -0.763, 'seed_std_pp': 3.189, 'deltas_pp': [-1.66, -1.319, 5.268], 'n_pairs': 3, 'sign_consistent': False}
- easy8: NO-GO: |drop| 0.85pp < 1.0pp {'mean_drop_pp': 0.847, 'seed_std_pp': 1.103, 'deltas_pp': [0.709, -1.723, -1.528], 'n_pairs': 3, 'sign_consistent': False}
- **mean_fg16 (primary): NO-GO: |drop| 0.04pp < 1.0pp** {'mean_drop_pp': 0.042, 'seed_std_pp': 1.418, 'deltas_pp': [-0.476, -1.521, 1.87], 'n_pairs': 3, 'sign_consistent': False}
- per-organ delta (difficult8, pp): {'Gallbladder': 2.794, 'Esophagus': 0.278, 'Pancreas': 2.325, 'Duodenum': -4.839, 'Colon': 0.169, 'Intestine': 1.67, 'Adrenal': 5.106, 'Rectum': -1.4}
- largest drop on seed0 (conditional_moderate_s0), difficult8 per-organ (pp): {'Gallbladder': -2.088, 'Esophagus': 0.194, 'Pancreas': -3.306, 'Duodenum': 3.739, 'Colon': -4.104, 'Intestine': -0.743, 'Adrenal': 10.453, 'Rectum': -17.426}

## Exploratory (post-hoc, NOT for gate claims)
- longtail random_s0-worst5 {'Adrenal': 10.588, 'Duodenum': -4.163, 'Gallbladder': 1.24, 'Esophagus': -5.027, 'Rectum': -12.727}
- sitelike random_s0-worst5 {'Adrenal': 6.752, 'Duodenum': 1.14, 'Gallbladder': 0.995, 'Esophagus': -2.46, 'Rectum': -0.963}
- conditional random_s0-worst5 {'Adrenal': 5.106, 'Duodenum': -4.839, 'Gallbladder': 2.794, 'Esophagus': 0.278, 'Rectum': -1.4}
