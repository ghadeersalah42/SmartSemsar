# Grounded replies report

Checked **300 replies** built for the 100 listings in `data/final_merged_dataset.csv`, with the same functions the app uses.
Every number and compound name in each reply was compared with the listing's database record.

| Reply type | Replies | Mismatches |
|---|---|---|
| chat_card | 100 | 0 |
| rent_details | 100 | 0 |
| telegram_report | 100 | 0 |

**Total mismatches: 0 / 300**

## Negative control

The same replies with one value changed (price +10%, area +15 m², or another listing's compound).
The check caught **300 / 300** tampered replies.
Missed: none

**Result: PASS**

