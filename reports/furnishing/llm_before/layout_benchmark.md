# Furnishing: layouts vs. the validator

## rules

- Layouts: 36 (12 plans x 3 customer profiles)
- **Pass on the first try: 36 / 36 (100.0 %)**
- Pass after repair (what the user sees): 36 / 36 (100.0 %)
- Mean time per layout: 1.71 s

| Rejection reason | Count | Share |
|---|---:|---:|
| (none) | 0 | - |

## llm

- Layouts: 36 (36 where a model answered)
- **Pass on the first try: 2 / 36 (5.6 %)**
- Pass after repair (what the user sees): 36 / 36 (100.0 %)
- Mean time per layout: 22.172 s

| Rejection reason | Count | Share |
|---|---:|---:|
| no_free_wall | 25 | 25.0 % |
| doors_clear | 24 | 24.0 % |
| inside_room | 17 | 17.0 % |
| bad_reference | 17 | 17.0 % |
| front_clearance | 8 | 8.0 % |
| no_overlaps | 7 | 7.0 % |
| wrong_room_type | 1 | 1.0 % |
| fixtures_clear | 1 | 1.0 % |
