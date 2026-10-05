# Furniture models / مصادر موديلات الفرش

Both sources are **Creative Commons Zero (CC0)**: free for personal, educational and
commercial use, no credit required. http://creativecommons.org/publicdomain/zero/1.0/

Every `.glb` here follows one convention, so the viewer places it with no scaling:
meters, y up, standing on y = 0, footprint centred on the origin, front toward +z.

## Poly Haven (textured models)

https://polyhaven.com/models/furniture — built by `data/scripts/build_polyhaven_models.py`,
which downloads each model's glTF with 1k textures, shrinks the textures to 512 px JPEG and
packs everything into one GLB. Models keep their **true size**; the script writes that size
into `data/catalog/furniture.json`.

    python data/scripts/build_polyhaven_models.py

| catalog id | Poly Haven asset | size (w x d x h, m) |
|---|---|---|
| sofa_3_seat | sofa_02 | 1.81 x 0.82 x 0.71 |
| sofa_2_seat | Sofa_01 | 1.57 x 0.66 x 0.80 |
| armchair | modern_arm_chair_01 | 0.82 x 0.99 x 1.02 |
| coffee_table | modern_coffee_table_01 | 1.20 x 0.60 x 0.39 |
| tv_unit | modern_wooden_cabinet (+ a plain flat TV added by the script) | 2.44 x 0.52 x 0.68 |
| bookshelf | wooden_display_shelves_01 | 1.08 x 0.37 x 1.56 |
| dining_table_6 | dining_table | 2.26 x 1.39 x 0.88 |
| dining_table_4 | round_wooden_table_01 | 1.40 x 1.40 x 1.01 |
| dining_chair | painted_wooden_chair_01 | 0.43 x 0.54 x 0.96 |
| nightstand | side_table_01 | 0.55 x 0.45 x 0.55 |
| wardrobe | drawer_cabinet | 1.14 x 0.49 x 1.88 |
| dresser | WoodenTable_03 | 1.33 x 0.58 x 0.83 |
| desk | metal_office_desk | 2.00 x 0.95 x 0.79 |
| console_table | chinese_console_table | 1.72 x 0.34 x 0.66 |
| shoe_cabinet | vintage_wooden_drawer_01 | 0.86 x 0.46 x 0.55 |

## Kenney Furniture Kit 2.0 (simple models)

https://kenney.nl/assets/furniture-kit — used only where Poly Haven has no suitable model
(it has no modern bed). Built by `data/scripts/build_furniture_models.py`, which stretches the
kit model to the catalog size and replaces the kit's colours with a calmer palette.

    python data/scripts/build_furniture_models.py "<kit>/Models/GLTF format"

| catalog id | kit model |
|---|---|
| bed_double | bedDouble |
| bed_single | bedSingle |
