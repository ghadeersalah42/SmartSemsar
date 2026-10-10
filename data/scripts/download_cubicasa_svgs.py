"""
Download only the model.svg files we need from Kaggle (qmarva/cubicasa5k),
never the full ~5 GB dataset. Reads the token from KAGGLE_API_TOKEN (env only).

Usage (from repo root):
    export KAGGLE_API_TOKEN=...
    python data/scripts/download_cubicasa_svgs.py
"""
import glob
import os
import shutil
import subprocess
import tempfile
import zipfile

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET = "qmarva/cubicasa5k"
SUBFOLDERS = ["high_quality_architectural", "high_quality", "colorful"]


def download_svg(cubicasa_id, out_dir):
    """Try each CubiCasa subfolder; return the one that worked or None."""
    for sub in SUBFOLDERS:
        with tempfile.TemporaryDirectory() as tmp:
            remote = f"cubicasa5k/cubicasa5k/{sub}/{cubicasa_id}/model.svg"
            r = subprocess.run(["kaggle", "datasets", "download", DATASET, "-f", remote, "-p", tmp],
                               capture_output=True, text=True)
            if r.returncode != 0:
                continue
            for z in glob.glob(os.path.join(tmp, "*.zip")):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(tmp)
            svgs = glob.glob(os.path.join(tmp, "**", "*.svg"), recursive=True)
            if svgs and os.path.getsize(svgs[0]) > 0:
                shutil.copy(svgs[0], os.path.join(out_dir, f"{cubicasa_id}.svg"))
                return sub
    return None


# if __name__ == "__main__":
#     if not os.environ.get("KAGGLE_API_TOKEN"):
#         raise SystemExit("Set KAGGLE_API_TOKEN in the environment first.")
#     df = pd.read_csv(os.path.join(REPO_ROOT, "data", "final_merged_dataset.csv"), encoding="utf-8-sig")
#     out_dir = os.path.join(REPO_ROOT, "data", "cubicasa_svg")
#     os.makedirs(out_dir, exist_ok=True)
#     for cid in sorted(df["cubicasa_id"].astype(int).unique()):
#         print(cid, "->", download_svg(cid, out_dir) or "NOT FOUND")
