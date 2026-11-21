# Furnishing: layouts vs. the validator

## rules

- Layouts: 36 (12 plans x 3 customer profiles)
- **Pass on the first try: 36 / 36 (100.0 %)**
- Pass after repair (what the user sees): 36 / 36 (100.0 %)
- Mean time per layout: 1.604 s

| Rejection reason | Count | Share |
|---|---:|---:|
| (none) | 0 | - |

## llm

- Layouts: 36 (36 where a model answered)
- **Pass on the first try: 4 / 36 (11.1 %)**
- Pass after repair (what the user sees): 36 / 36 (100.0 %)
- Mean time per layout: 30.078 s

| Rejection reason | Count | Share |
|---|---:|---:|
| no_free_wall | 23 | 26.7 % |
| inside_room | 17 | 19.8 % |
| doors_clear | 14 | 16.3 % |
| bad_reference | 12 | 14.0 % |
| no_overlaps | 9 | 10.5 % |
| front_clearance | 8 | 9.3 % |
| no_open_floor | 2 | 2.3 % |
| wrong_room_type | 1 | 1.2 % |
