# E1 phenomenon-gate paired analysis

metric: `mean_fg16` over 20 locked imagesVal cases; missing arms: none

| arm | dice |
|---|---|
| random_moderate_s0 | 0.6784 |
| random_moderate_s1 | 0.6623 |
| random_moderate_s2 | 0.6653 |
| longtail_moderate_s0 | 0.6617 |
| longtail_moderate_s1 | 0.6497 |
| longtail_moderate_s2 | 0.6561 |
| sitelike_moderate_s0 | 0.6930 |
| sitelike_moderate_s1 | 0.6521 |
| sitelike_moderate_s2 | 0.5838 |
| conditional_moderate_s0 | 0.6736 |
| conditional_moderate_s1 | 0.6471 |
| conditional_moderate_s2 | 0.6840 |

## longtail vs random (paired by mask seed)
- paired deltas (pp): [-1.667, -1.265, -0.929]
- mean delta: **-1.29 pp** (seed std 0.30 pp, n=3)
- hard-5 organ deltas: {'Adrenal': 10.588, 'Duodenum': -4.163, 'Gallbladder': 1.24, 'Esophagus': -5.027, 'Rectum': -12.727}
- verdict: GRAY ZONE: 1.29pp drop in 1.0-2.0pp band, direction consistent; extend seeds before deciding

## sitelike vs random (paired by mask seed)
- paired deltas (pp): [1.46, -1.026, -8.149]
- mean delta: **-2.57 pp** (seed std 4.07 pp, n=3)
- hard-5 organ deltas: {'Adrenal': 6.752, 'Duodenum': 1.14, 'Gallbladder': 0.995, 'Esophagus': -2.46, 'Rectum': -0.963}
- verdict: PHENOMENON (2.57pp drop >= 2.0pp gate) BUT seed std exceeds gate - unreliable, extend seeds

## conditional vs random (paired by mask seed)
- paired deltas (pp): [-0.476, -1.521, 1.87]
- mean delta: **-0.04 pp** (seed std 1.42 pp, n=3)
- hard-5 organ deltas: {'Adrenal': 5.106, 'Duodenum': -4.839, 'Gallbladder': 2.794, 'Esophagus': 0.278, 'Rectum': -1.4}
- verdict: NO-GO: |delta| < 1pp, no usable phenomenon

