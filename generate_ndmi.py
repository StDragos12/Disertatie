from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

BANDS_DIR = BASE_DIR / "data" / "SitsPerBands"

B8_DIR = BANDS_DIR / "B8A_SR"
B11_DIR = BANDS_DIR / "B11_SR"

OUTPUT_DIR = BASE_DIR / "data" / "Indices" / "NDMI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROIS = ["roi1", "roi2"]

for roi in ROIS:
    b8_path = B8_DIR / f"{roi}.npy"
    b11_path = B11_DIR / f"{roi}.npy"

    b8 = np.load(b8_path).astype(np.float32)
    b11 = np.load(b11_path).astype(np.float32)

    denominator = b8 + b11

    ndmi = np.where(
        denominator == 0,
        0,
        (b8 - b11) / denominator
    )

    output_path = OUTPUT_DIR / f"{roi}.npy"

    np.save(output_path, ndmi)

    print(f"Saved NDMI for {roi}: {ndmi.shape}")