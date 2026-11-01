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

Rule-based planner: 500 layouts (100 plans x 5 profiles). LLM planner: 36 layouts (12 plans x 3 profiles), limited by
API quota. The rule planner is offline and free; the LLM costs quota and ~30 s per layout, so it was not run on all 500.

Rejection reasons (LLM, round 1, after the fix): 86 failed instructions, each counted once under its first reason.

| Reason | Count | Share | What happened |
|---|---:|---:|---|
| no_free_wall | 23 | 26.7 % | piece put on a wall with no free stretch as wide as the piece |
| inside_room | 17 | 19.8 % | piece would stick out of the room or into a wall |
| doors_clear | 14 | 16.3 % | piece within 0.7 m of a door |
| bad_reference | 12 | 14.0 % | "beside / in front of X" where X was never named |
| no_overlaps | 9 | 10.5 % | two pieces in the same place |
| front_clearance | 8 | 9.3 % | no free floor in front of a wardrobe / dresser |
| no_open_floor, wrong_room_type | 3 | 3.5 % | no open floor for a free-standing piece; piece not meant for that room |

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
| Pieces generated | 21 (18 in run 1, 3 new in run 2) |
| **Time per piece: median / mean / P95** | **246 s / 261 s / 311 s** (initial benchmark: the 21 successful generations only) |
| Server calls that failed (all ngrok gateway errors, not the model; not in the times) | 3 |
| **Cache hit rate when the same photos run again** | **18 / 22 = 82 %** |

| Run | Calls | Cache hits | Made | Failed |
|---|---:|---:|---:|---:|
| 1 (cold) | 20 | 0 | 18 | 2 |
| 2 (same photos) | 22 | 18 | 3 | 1 |

All 18 pieces successfully generated in run 1 were reused (cache hits) in run 2. Run 2's 4 misses are 3 photos the vision step accepted only in
run 2 and 1 retry of a failed call. (Both runs pooled give 18 / 42, but that mixes in the cold run, so 82 % is the figure quoted.)

The P95 comes from only 21 timed pieces, so it is an initial figure: the slowest piece (496 s) sits just above it,
and more runs are needed before the P95 is stable.

From the Kaggle log: shape takes about 5 s of diffusion plus 78 s of volume decoding. The remaining ~2.5 min is texturing.
The cache key is the photo's bytes, so a re-run after changing a width or the wishes costs nothing. The same piece photographed
again, or resent through WhatsApp (recompressed), is a miss.

What happened to every photo in run 2 (section 2's 87 % stops at recognition; this follows the photos through 3D):

| Step | Photos | Lost here |
|---|---:|---|
| Photos | 30 | 6 should be refused: 5 refused correctly, 1 accepted by mistake (bed with nightstands) |
| Single pieces | 24 | 2 not recognised: 1 wrong kind, 1 refused as several pieces |
| Recognised correctly | 22 | 1 lost after recognition: the server ran 342 s, then its reply came back broken through ngrok (transport failure, not the model) |
| 3D models generated | 21 | 4 failed the size check (L-shaped sofa frame, table shot end-on, folding table, loose bed parts) |
| Passed the size check | 17 | 3 look wrong by eye |
| Look right | 14 | |

- **3D success on furniture photos: 14 / 24 (58 %)**: good models (by eye, `data/eval/furniture_3d_eyecheck.csv`)
  out of the 24 single-piece photos. This is the success rate for turning a real furniture photo into a 3D piece.
- Right outcome over all 30 photos: 19 / 30 (63 %) = 14 good models + 5 correct refusals. 19 does not mean 19 photos
  became 3D models.
- Automatic checks only: 22 / 30 (73 %) = 17 models + 5 correct refusals.
- The 3 that look wrong: the dark gap between a table's legs became solid walls, a sofa photographed at night came out torn,
  and a glass cabinet merged with the box behind it. Background removal is the weak point in dark, cluttered photos.
- The recognition step in this run got 27 / 30 right (26 / 30 in section 2's run): about one photo of run-to-run variation.
Photo 2 failed with an ngrok 503 in both runs after ~340 s: its response does not get through the free tunnel.

Pictures: `before_after/NN_<photo>.jpg` (photo next to its 3D model), `before_after/all.jpg` (all 30),
`before_after/photo_to_3d.mp4` (every photo, then its model turning).
