# Furnishing: real furniture photos

- Success (whole set): **22 / 30 (73 %)**
- Pieces recognised and turned into 3D: 17 / 24 (71 %)
- Photos correctly refused (room, several pieces, kitchen): 5 / 6 (83 %)
- Vision model: gemini-flash-lite-latest; median time per photo: 4.4 s

| Condition | Success |
|---|---:|
| clutter | 15 / 19 (79 %) |
| cut_off | 0 / 2 (0 %) |
| glare | 2 / 2 (100 %) |
| low_light | 11 / 14 (79 %) |
| normal | 2 / 3 (67 %) |
| odd_angle | 3 / 6 (50 %) |

| Outcome | Photos |
|---|---:|
| 3d_ok | 17 |
| refused | 5 |
| bad_3d_size | 4 |
| 3d_error | 1 |
| wrong_kind | 1 |
| wrongly_accepted | 1 |
| refused:several_items | 1 |

## 3D pieces

| Generator | Calls | Generated | Mean s | Median s | P95 s | Cache hit rate |
|---|---:|---:|---:|---:|---:|---:|
| colab | 22 | 3 | 241.3 | 232.8 | 260.03 | 82 % |
