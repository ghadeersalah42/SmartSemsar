# Round 2, requirement 8: Furnishing and 3D

Measured on 2026-10-30. Commands and how each number is counted: see the main README, section
"Round 2: Furnishing and 3D numbers". Every run is in the `.json` files next to this one.

## 1. Layouts that pass the validator on the first try

| Planner | Set | First try: before | First try: after | Shown to the user (after repair) |
|---|---|---:|---:|---:|
| Rules (`furniture_placer`) | 100 plans x 5 customer profiles = 500 layouts | 500 / 500 (100 %) | unchanged | 500 / 500 (100 %) |
| LLM (`llm_furnisher`, gemini-flash-lite) | 12 plans x 3 profiles = 36 layouts | 2 / 36 (6 %) | 4 / 36 (11 %) | 36 / 36 (100 %) |

- The rule placer checks every spot with the validator's own rules before it uses it, so its layouts pass
  by construction; `keep_paths_open` never had to drop an item in the 500 layouts.
- The LLM almost never gets a whole apartment right in one answer. Nothing it gets wrong reaches the user:
  each instruction is validated, failures go back to the model (up to 3 rounds), and the rules fill any room left empty.
- Instructions that failed in round 1: 100 before, 86 after.

Most common rejection reasons (LLM, round 1, after the fix):

| Reason | Share | What happened |
|---|---:|---|
| no_free_wall | 27 % | piece put on a wall with no free stretch as wide as the piece |
| inside_room | 20 % | piece would stick out of the room or into a wall |
| doors_clear | 16 % | piece within 0.7 m of a door |
| bad_reference | 14 % | "beside / in front of X" where X was never named |
| no_overlaps | 11 % | two pieces in the same place |
| front_clearance | 9 % | no free floor in front of a wardrobe / dresser |

What changed:
- The room description the model reads gave "free wall" without the pieces already placed and counted the
  stretch under a window for wardrobes too. It now subtracts both (`furniture_actions.RoomView.describe`).
- The prompt now spells out the checks behind the top reasons, and the catalog lists each item's front clearance.

## 2. Real furniture photos (poor lighting, odd angles, clutter)

30 photos from the workshop folders, labelled by hand in `data/eval/furniture_photos.csv`:
24 single pieces and 6 that must be refused (a pair of chairs, two beds, a bed with its nightstands,
loose parts, a screen showing a drawing, two cabinets). Same model for both runs (gemini-flash-lite).

| | Before | After |
|---|---:|---:|
| **Whole set** | **23 / 30 (77 %)** | **26 / 30 (87 %)** |
| Pieces recognised with the right kind | 17 / 24 (71 %) | 21 / 24 (88 %) |
| Photos correctly refused | 6 / 6 | 5 / 6 |
| Low light (14 photos) | 11 / 14 | 12 / 14 |
| Clutter (19) | 14 / 19 | 17 / 19 |
| Odd angle (6) | 2 / 6 | 5 / 6 |

Failures before: clutter in the background read as "several pieces", a corner out of frame read as
"cut off", a bed frame without a headboard in a dark workshop read as a wardrobe or as not furniture.
Fixes in `vision_service`: phone photos are turned upright (EXIF rotation was ignored), dark photos are
brightened before the model reads them, and the prompt says how to treat a main subject with clutter
around it, unfinished pieces, and small cut-offs.
Cost: the bed with two nightstands is now accepted as a bed instead of being refused.
This is recognition only (photo -> right catalog kind, usable or not). The 3D step was not in this run (see 3).
With 30 photos, one photo is 3 points.

## 3. Time per 3D piece and cache hit rate

Every photo -> 3D call now writes its time and whether it came from the cache to `data/metrics/furnishing.jsonl`
(`multi_furnish.raw_model`). `python -m backend.services.furnishing_metrics` gives mean / median / P95 seconds
per generated piece and the hit rate per generator.

Not measured yet: the Kaggle/Colab 3D server was offline on 2026-10-30 (ngrok address expired). To get the numbers,
start `cv_service/furniture_server_kaggle.ipynb`, put its address in `.env`, and run the photo benchmark twice with
`--generator colab`. The first run times every piece, and the second shows the cache (same photo, same model).
