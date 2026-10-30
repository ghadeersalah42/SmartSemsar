# Furnishing: real furniture photos

- Success (whole set): **21 / 30 (70 %)**
- Pieces recognised and turned into 3D: 15 / 24 (62 %)
- Photos correctly refused (room, several pieces, kitchen): 6 / 6 (100 %)
- Vision model: gemini-flash-lite-latest; median time per photo: 4.5 s

| Condition | Success |
|---|---:|
| clutter | 13 / 19 (68 %) |
| cut_off | 0 / 2 (0 %) |
| glare | 2 / 2 (100 %) |
| low_light | 9 / 14 (64 %) |
| normal | 3 / 3 (100 %) |
| odd_angle | 3 / 6 (50 %) |

| Outcome | Photos |
|---|---:|
| 3d_ok | 15 |
| refused | 6 |
| bad_3d_size | 3 |
| 3d_error | 2 |
| refused:several_items | 2 |
| refused:single_item | 1 |
| wrong_kind | 1 |

## 3D pieces

| Generator | Calls | Generated | Mean s | Median s | P95 s | Cache hit rate |
|---|---:|---:|---:|---:|---:|---:|
| colab | 20 | 18 | 263.9 | 246.9 | 496.01 | 0 % |
