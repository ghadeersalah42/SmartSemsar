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
This table is recognition only (photo -> right catalog kind, usable or not); section 3 adds the 3D step.
With 30 photos, one photo is 3 points, and the same photos give ±1 photo between runs even at temperature 0
(the 3D runs below: 21/30 and 22/30 at the recognition + 3D level).

## 3. Photo -> 3D, time per piece and cache hit rate

Server: `cv_service/furniture_server_kaggle.ipynb` (Hunyuan3D-2 mini, textured, Tesla T4 on Kaggle, reached through ngrok).
The same 30 photos, two runs (`photo_3d_run1`, `photo_3d_run2`). Every call is logged by `multi_furnish.raw_model`.

| | Number |
|---|---:|
| Pieces generated | 21 |
| **Time per piece: median / mean / P95** | **246 s / 261 s / 311 s** |
| Server calls that failed (all ngrok gateway errors, not the model) | 3 |
| **Cache hit rate, second run of the same photos** | **18 / 22 = 82 %** |
| Cache hit rate over both runs | 18 / 42 = 43 % |

From the Kaggle log: shape takes about 5 s of diffusion plus 78 s of volume decoding. The remaining ~2.5 min is texturing.
The cache key is the photo's bytes, so a re-run after changing a width or the wishes costs nothing. The same piece photographed
again, or resent through WhatsApp (recompressed), is a miss.

End to end (photo -> right kind -> 3D model with a sensible size), run 2: **22 / 30 (73 %)**, pieces 17 / 24.
"Sensible size" is not enough, so every model that passed was also checked by eye (`data/eval/furniture_3d_eyecheck.csv`):
**14 / 17 look right.** The 3 that don't: the dark gap between a table's legs became solid walls, a sofa photographed at night
came out torn, and a glass cabinet merged with the box behind it. Background removal is the weak point in dark, cluttered photos.
Size check failures (3): a long L-shaped sofa frame, a table photographed from its short end, and loose headboard/footboard parts.
Photo 2 failed with an ngrok 503 in both runs after ~340 s: its response does not get through the free tunnel.

Pictures: `before_after/NN_<photo>.jpg` (photo next to its 3D model), `before_after/all.jpg` (all 30),
`before_after/photo_to_3d.mp4` (every photo, then its model turning).
