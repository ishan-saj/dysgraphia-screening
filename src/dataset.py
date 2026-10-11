from pathlib import Path
import math

import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BBOX_CSV = _PROJECT_ROOT / "data" / "metadata" / "form_handwriting_bboxes.csv"
_GLOBAL_HANDWRITING_BBOXES = {}

if _DEFAULT_BBOX_CSV.exists():
    _bbox_df = pd.read_csv(_DEFAULT_BBOX_CSV)
    for _, _r in _bbox_df.iterrows():
        _GLOBAL_HANDWRITING_BBOXES[str(_r["form_id"]).strip()] = (
            int(_r["x1"]),
            int(_r["y1"]),
            int(_r["x2"]),
            int(_r["y2"]),
        )


def get_handwriting_crop_coords(form_id: str, img_shape: tuple, padding: int = 30) -> tuple:
    """Return (crop_x1, crop_y1, crop_x2, crop_y2) for form_id on an image of shape img_shape."""
    if form_id is None:
        raise ValueError("form_id must not be None when cropping handwriting.")
    norm_id = FORM_ID_MAP.get(str(form_id).strip(), str(form_id).strip())
    if norm_id not in _GLOBAL_HANDWRITING_BBOXES:
        raise KeyError(f"Handwriting bounding box not found for form_id='{form_id}' (normalized='{norm_id}').")
    x1, y1, x2, y2 = _GLOBAL_HANDWRITING_BBOXES[norm_id]
    img_h, img_w = img_shape[:2]
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(img_w, x2 + padding)
    crop_y2 = min(img_h, y2 + padding)
    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        raise ValueError(f"Invalid crop coordinates for form_id='{form_id}': {(crop_x1, crop_y1, crop_x2, crop_y2)}")
    return crop_x1, crop_y1, crop_x2, crop_y2


def load_handwriting_crop(form_id: str, img: np.ndarray, padding: int = 30) -> np.ndarray:
    """
    Single shared crop function used across all pipelines and baselines.
    Crops `img` to the handwriting region of `form_id` (excluding printed IAM headers/footers).
    For synthetic images, pass the paired clean form's `form_id`.
    """
    if img is None:
        raise ValueError(f"Input image is None for form_id='{form_id}'.")
    crop_x1, crop_y1, crop_x2, crop_y2 = get_handwriting_crop_coords(form_id, img.shape, padding=padding)
    return img[crop_y1:crop_y2, crop_x1:crop_x2]


def resize_handwriting_crop(crop_gray: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Single shared resize function used across all pipelines, baselines, and caches.
    Always resizes `crop_gray` to `(size, size)` using `cv2.INTER_AREA`.
    """
    if crop_gray is None or crop_gray.size == 0:
        raise ValueError("crop_gray must be a non-empty array when resizing.")
    return cv2.resize(crop_gray, (int(size), int(size)), interpolation=cv2.INTER_AREA)


def load_and_resize_handwriting(
    form_id: str, img: np.ndarray, size: int = 224, padding: int = 30
) -> tuple:
    """
    Single shared crop + INTER_AREA resize entry point.
    Returns `(crop_gray, resized_224)` where `crop_gray` is at native crop resolution
    and `resized_224` is resized to `(size, size)` via `cv2.INTER_AREA`.
    """
    crop_gray = load_handwriting_crop(form_id, img, padding=padding)
    resized_224 = resize_handwriting_crop(crop_gray, size=size)
    return crop_gray, resized_224


def crop_to_resnet_tensor(crop_gray: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Convert a native handwriting crop `crop_gray` into a `[3, size, size]` float32 tensor in `[0, 1]`
    using the shared `cv2.INTER_AREA` resize path.
    """
    resized = resize_handwriting_crop(crop_gray, size=size)
    if resized.ndim == 2:
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        rgb = resized
    return torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0


def fold_angle_45(angle_deg: float) -> float:
    """
    Fold any angle in degrees into [-45.0, 45.0] without boundary wraparound at +45.0.
    Prevents OpenCV minAreaRect wraparound where -1 deg is reported as +89 deg.
    """
    a = float(angle_deg)
    while a > 45.0:
        a -= 90.0
    while a < -45.0:
        a += 90.0
    return float(a)


def detect_line_bands_from_binary(cleaned: np.ndarray) -> list:
    """Detect horizontal text line bands (y1, y2) purely from horizontal projection profile."""
    H, W = cleaned.shape[:2]
    h_proj = (cleaned > 0).sum(axis=1).astype(np.float64)
    smooth_k = 15
    if H >= smooth_k:
        smooth_proj = np.convolve(
            h_proj, np.ones(smooth_k, dtype=np.float64) / smooth_k, mode="same"
        )
    else:
        smooth_proj = h_proj

    pos_vals = smooth_proj[smooth_proj > 0]
    row_thresh = max(4.0, float(np.mean(pos_vals)) * 0.14) if len(pos_vals) > 0 else 4.0
    active_rows = smooth_proj > row_thresh

    line_bands = []
    in_band = False
    b_start = 0
    for r_i, is_act in enumerate(active_rows):
        if is_act and not in_band:
            in_band = True
            b_start = r_i
        elif not is_act and in_band:
            in_band = False
            if r_i - b_start >= 16:
                line_bands.append((b_start, r_i))
    if in_band and (H - b_start >= 16):
        line_bands.append((b_start, H))

    refined_bands = []
    for y1, y2 in line_bands:
        bh = y2 - y1
        if bh > 155:
            n_splits = max(2, int(round(bh / 105.0)))
            step = bh / n_splits
            curr_y = y1
            for s_i in range(1, n_splits):
                target = int(round(y1 + s_i * step))
                win_l = max(y1 + 15, target - 25)
                win_r = min(y2 - 15, target + 25)
                if win_r > win_l:
                    split_y = win_l + int(np.argmin(smooth_proj[win_l:win_r]))
                else:
                    split_y = target
                if split_y - curr_y >= 16:
                    refined_bands.append((curr_y, split_y))
                curr_y = split_y
            if y2 - curr_y >= 16:
                refined_bands.append((curr_y, y2))
        else:
            refined_bands.append((y1, y2))
    return refined_bands if refined_bands else [(0, H)]


def _estimate_line_slant_fast(binary_line: np.ndarray) -> float:
    """
    Estimate stroke slant angle in [-45, 45] degrees using vertical projection variance
    under shear transforms at native/48px-height line scale.
    """
    h, w = binary_line.shape[:2]
    if h < 8 or w < 15 or np.count_nonzero(binary_line) < 20:
        return 0.0
    if h > 48:
        scale = 48.0 / h
        small = cv2.resize(
            binary_line,
            (max(8, int(round(w * scale))), 48),
            interpolation=cv2.INTER_AREA,
        )
        small = (small > 64).astype(np.uint8) * 255
    else:
        small = (binary_line > 0).astype(np.uint8) * 255

    sh, sw = small.shape
    best_angle = 0.0
    best_var = -1.0
    for angle in range(-45, 46, 3):
        shear = math.tan(math.radians(angle))
        pad = int(abs(shear) * sh) + 1
        new_w = sw + pad
        tx = pad if shear < 0 else 0
        M = np.float32([[1.0, shear, float(tx)], [0.0, 1.0, 0.0]])
        sheared = cv2.warpAffine(
            small, M, (new_w, sh), flags=cv2.INTER_NEAREST, borderValue=0
        )
        col_sums = sheared.sum(axis=0, dtype=np.float64)
        var = float(col_sums.var())
        if var > best_var:
            best_var = var
            best_angle = float(angle)
    return fold_angle_45(best_angle)


def extract_pixel_features(crop_gray: np.ndarray) -> np.ndarray:
    """
    Extract the 9 clinical/geometric handwriting features purely from a cropped grayscale
    handwriting image (`crop_gray`) at native crop resolution prior to 224x224 resizing.
    Uses ONLY `crop_gray` — zero XML annotations, zero metadata_all.csv lookups, and zero
    external word/line bounding boxes.

    Per-Feature Algorithmic Documentation (matches DysgraphiaDataset.FEATURE_COLUMNS order):
    ----------------------------------------------------------------------------------------
    1. `skew` (index 0):
       - Lines are segmented via horizontal ink projection profile (`detect_line_bands_from_binary`).
       - Within each line, words are formed via morphological closing `(24, 3)` and character
         connected components (`cv2.connectedComponentsWithStats`).
       - Each word's baseline point `(center_x, bottom)` uses the median of its character component
         bottoms (reducing descender bias). Ordinary least squares fits `bottom = slope * center_x + b`
         per line, converts `-atan(slope)` to degrees, folds into `[-45, 45]` via `fold_angle_45`,
         and scales by `100.0` (hundredths of a degree, matching IAM scale).
    2. `baseline_deviation` (index 1):
       - Combines in quadrature: (a) the mean per-line linear regression RMS residual (px) of word
         bottoms around the fitted line baseline, and (b) the inter-line angular dispersion excursion
         `400 * tan(std(line_angles_rad))` so that both within-line word bounce and per-line tilt
         irregularity increase `baseline_deviation`.
    3. `word_spacing_cv` (index 2):
       - Within each line band, words are sorted left-to-right by `x` and inter-word horizontal
         whitespace gaps `gap = next_x - (curr_x + curr_w)` with `gap >= 16 px` are collected.
       - Computes the coefficient of variation `std(gaps) / mean(gaps)` per line and averages across lines.
    4. `character_height_cv` (index 3):
       - Extracts character connected components (`area >= 10, height >= 5, width >= 2`) from the
         adaptive-thresholded (`blockSize=31, C=15`), `2x2` morphologically opened binary crop.
       - Combines in quadrature the component height CV `std(char_heights) / mean(char_heights)` and
         `1.8 * std(line_mean_char_heights) / mean(line_mean_char_heights)` to capture both letter-level
         and line-to-line size irregularity.
    5. `average_word_height` (index 4):
       - Mean bounding-box height (px) of all morphologically grouped word components across lines.
    6. `average_word_width` (index 5):
       - Mean bounding-box width (px) of all morphologically grouped word components across lines.
    7. `stroke_density` (index 6):
       - Ratio of total foreground ink pixels (`cleaned > 0`) to the tight ink bounding-box area
         `(max_x - min_x + 1) * (max_y - min_y + 1)` at native crop resolution.
    8. `slant_angle` (index 7):
       - For each detected line band, shears the binary line image across `[-45, 45]` deg (step `3` deg)
         and selects the angle maximizing vertical projection variance (`_estimate_line_slant_fast`),
         folded into `[-45, 45]` deg and averaged across lines.
    9. `writing_area` (index 8):
       - Sum of active line bounding-box areas `(line_x_max - line_x_min) * (line_y2 - line_y1)`
         normalized by standard IAM page area (`2479 * 3542`).
    """
    if crop_gray is None or crop_gray.size == 0:
        raise ValueError("crop_gray must be a non-empty 2D grayscale array.")
    if crop_gray.ndim == 3:
        crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)

    H, W = crop_gray.shape[:2]
    if H < 10 or W < 10:
        return np.zeros(9, dtype=np.float32)

    binary = cv2.adaptiveThreshold(
        crop_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    open_k = np.ones((2, 2), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    char_comps = []
    char_heights = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 10 or h < 5 or w < 2:
            continue
        if h > int(0.48 * H) or w > int(0.65 * W):
            continue
        char_comps.append((x, y, w, h, area))
        char_heights.append(float(h))

    line_bands = detect_line_bands_from_binary(cleaned)

    line_char_lists = [[] for _ in range(len(line_bands))]
    for c in char_comps:
        cy = c[1] + c[3] / 2.0
        best_li = 0
        best_dist = float("inf")
        for li, (ly1, ly2) in enumerate(line_bands):
            if ly1 <= cy <= ly2:
                best_li = li
                best_dist = 0.0
                break
            d = min(abs(cy - ly1), abs(cy - ly2))
            if d < best_dist:
                best_dist = d
                best_li = li
        if best_dist <= 45.0:
            line_char_lists[best_li].append(c)

    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 3))
    word_heights = []
    word_widths = []
    line_spacing_cvs = []
    line_baseline_rms = []
    line_angles = []
    line_slants = []
    line_mean_char_heights = []
    active_line_area_sum = 0.0

    for li, (ly1, ly2) in enumerate(line_bands):
        line_img = cleaned[ly1:ly2, :]
        if np.count_nonzero(line_img) < 25:
            continue

        line_slants.append(_estimate_line_slant_fast(line_img))

        c_list = line_char_lists[li]
        if c_list:
            line_mean_char_heights.append(float(np.mean([c[3] for c in c_list])))

        closed_line = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, word_kernel)
        wnum, _, wstats, _ = cv2.connectedComponentsWithStats(closed_line, connectivity=8)

        line_word_dicts = []
        for wi in range(1, wnum):
            wx = int(wstats[wi, cv2.CC_STAT_LEFT])
            wy = int(wstats[wi, cv2.CC_STAT_TOP])
            ww = int(wstats[wi, cv2.CC_STAT_WIDTH])
            wh = int(wstats[wi, cv2.CC_STAT_HEIGHT])
            warea = int(wstats[wi, cv2.CC_STAT_AREA])
            if warea < 45 or ww < 14 or wh < 10:
                continue

            wx2 = wx + ww
            sub_bottoms = [
                float(c[1] + c[3])
                for c in c_list
                if (c[0] + c[2] / 2.0) >= wx - 4 and (c[0] + c[2] / 2.0) <= wx2 + 4
            ]
            if sub_bottoms:
                w_bottom = float(np.median(sub_bottoms))
            else:
                w_bottom = float(ly1 + wy + wh)

            word_heights.append(float(wh))
            word_widths.append(float(ww))
            line_word_dicts.append(
                {
                    "x": float(wx),
                    "y": float(ly1 + wy),
                    "w": float(ww),
                    "h": float(wh),
                    "center_x": float(wx) + float(ww) / 2.0,
                    "bottom": w_bottom,
                }
            )

        if not line_word_dicts:
            continue

        line_word_dicts.sort(key=lambda d: d["x"])
        lx_min = min(d["x"] for d in line_word_dicts)
        lx_max = max(d["x"] + d["w"] for d in line_word_dicts)
        active_line_area_sum += float(max(0.0, (lx_max - lx_min) * (ly2 - ly1)))

        gaps = []
        for w1, w2 in zip(line_word_dicts, line_word_dicts[1:]):
            gap = w2["x"] - (w1["x"] + w1["w"])
            if gap >= 16.0:
                gaps.append(float(gap))
        if len(gaps) >= 2:
            g_arr = np.array(gaps, dtype=np.float64)
            g_mean = float(np.mean(g_arr))
            if g_mean > 1e-6:
                line_spacing_cvs.append(float(np.std(g_arr) / g_mean))

        if len(line_word_dicts) >= 2:
            xs = np.array([d["center_x"] for d in line_word_dicts], dtype=np.float64)
            ys = np.array([d["bottom"] for d in line_word_dicts], dtype=np.float64)
            x_bar = float(np.mean(xs))
            y_bar = float(np.mean(ys))
            denom = float(np.sum((xs - x_bar) ** 2))
            if denom > 1e-6:
                slope = float(np.sum((xs - x_bar) * (ys - y_bar)) / denom)
                intercept = y_bar - slope * x_bar
                residuals = ys - (slope * xs + intercept)
                rms_fit = float(np.sqrt(np.mean(residuals ** 2)))
                angle_deg = fold_angle_45(-math.degrees(math.atan(slope)))
                line_angles.append(angle_deg)
                line_baseline_rms.append(rms_fit)

    if len(char_heights) >= 2:
        ch_arr = np.array(char_heights, dtype=np.float64)
        comp_cv = float(np.std(ch_arr) / max(1e-6, np.mean(ch_arr)))
    else:
        comp_cv = 0.0
    if len(line_mean_char_heights) >= 2:
        lh_arr = np.array(line_mean_char_heights, dtype=np.float64)
        line_scale_cv = float(np.std(lh_arr) / max(1e-6, np.mean(lh_arr)))
    else:
        line_scale_cv = 0.0
    character_height_cv = float(math.sqrt(comp_cv ** 2 + (1.8 * line_scale_cv) ** 2))

    if len(line_angles) >= 2:
        angle_std_rad = math.radians(float(np.std(line_angles)))
        inter_line_excursion = 400.0 * math.tan(min(math.pi / 4, angle_std_rad))
    else:
        inter_line_excursion = 0.0

    mean_angle_deg = fold_angle_45(float(np.mean(line_angles))) if line_angles else 0.0
    skew = float(mean_angle_deg * 100.0)

    within_rms = float(np.mean(line_baseline_rms)) if line_baseline_rms else 0.0
    baseline_deviation = float(math.sqrt(within_rms ** 2 + inter_line_excursion ** 2))

    word_spacing_cv = float(np.mean(line_spacing_cvs)) if line_spacing_cvs else 0.0
    average_word_height = float(np.mean(word_heights)) if word_heights else 0.0
    average_word_width = float(np.mean(word_widths)) if word_widths else 0.0

    ys_all, xs_all = np.where(cleaned > 0)
    if len(xs_all) > 0:
        min_x, max_x = int(xs_all.min()), int(xs_all.max())
        min_y, max_y = int(ys_all.min()), int(ys_all.max())
        bbox_area = float(max(1, (max_x - min_x + 1) * (max_y - min_y + 1)))
        ink_pixels = float(len(xs_all))
        stroke_density = float(ink_pixels / bbox_area)
        writing_area = float(active_line_area_sum / float(2479.0 * 3542.0))
    else:
        stroke_density = 0.0
        writing_area = 0.0

    slant_angle = float(np.mean(line_slants)) if line_slants else 0.0

    vec = np.array(
        [
            skew,
            baseline_deviation,
            word_spacing_cv,
            character_height_cv,
            average_word_height,
            average_word_width,
            stroke_density,
            slant_angle,
            writing_area,
        ],
        dtype=np.float32,
    )
    assert not np.isnan(vec).any() and not np.isinf(vec).any(), "NaN/Inf in extract_pixel_features!"
    return vec


FEATURE_COLUMNS = [
    "skew",
    "baseline_deviation",
    "word_spacing_cv",
    "character_height_cv",
    "average_word_height",
    "average_word_width",
    "stroke_density",
    "slant_angle",
    "writing_area",
]


class DysgraphiaDataset(Dataset):
    """
    Dataset for handwriting samples.

    Each sample returns:
        image           -> [3, 224, 224]
        opencv_features -> [9]
        ink_mask        -> [1, 224, 224]
        sample_id       -> sample identifier
        image_path      -> actual image path
    """

    # --------------------------------------------------
    # 9 handcrafted OpenCV features
    # --------------------------------------------------

    FEATURE_COLUMNS = FEATURE_COLUMNS

    def __init__(
        self,
        metadata_csv,
        image_root=None,
        image_size=224,
        transform=None,
        crop_handwriting_only=True,
        padding=30,
        bboxes_csv=None,
    ):
        """
        Parameters
        ----------
        metadata_csv : str
            Path to metadata_all.csv

        image_root : str
            Root directory containing handwriting images.

        image_size : int
            Image size used for the model.

        transform : optional
            Future image transformation pipeline.

        crop_handwriting_only : bool
            If True, automatically crops the image to isolate genuine handwriting,
            discarding the printed prompt text header and signature footer.

        padding : int
            Padding in pixels around the handwriting bounding box.

        bboxes_csv : str or Path, optional
            Path to form_handwriting_bboxes.csv.
        """

        self.metadata_csv = Path(metadata_csv)

        self.image_root = (
            Path(image_root)
            if image_root
            else None
        )

        self.image_size = image_size
        self.transform = transform
        self.crop_handwriting_only = crop_handwriting_only
        self.padding = padding
        self.handwriting_bboxes = dict(_GLOBAL_HANDWRITING_BBOXES)
        if self.crop_handwriting_only:
            print(
                f"Loaded {len(self.handwriting_bboxes)} handwriting bounding boxes for cropping."
            )

        # --------------------------------------------------
        # Load CSV
        # --------------------------------------------------

        project_root = Path(__file__).resolve().parent.parent
        if not self.metadata_csv.exists() and (project_root / self.metadata_csv).exists():
            self.metadata_csv = project_root / self.metadata_csv

        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found:\n"
                f"{self.metadata_csv}"
            )

        self.df = pd.read_csv(self.metadata_csv)

        print(
            f"Loaded metadata: {len(self.df)} samples"
        )

        # --------------------------------------------------
        # Check required columns
        # --------------------------------------------------

        required_columns = (
            ["sample_id"] + self.FEATURE_COLUMNS
        )

        missing_columns = [
            col
            for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"\nMissing required columns:\n"
                f"{missing_columns}\n\n"
                f"Available columns:\n"
                f"{list(self.df.columns)}"
            )

        # --------------------------------------------------
        # Convert feature columns to numbers
        # --------------------------------------------------

        self.df[self.FEATURE_COLUMNS] = (
            self.df[self.FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        # --------------------------------------------------
        # Keep only samples whose images exist locally
        # --------------------------------------------------

        valid_rows = []

        for _, row in self.df.iterrows():

            try:
                self._find_image(row)
                valid_rows.append(row)

            except FileNotFoundError:
                pass

        self.df = (
            pd.DataFrame(valid_rows)
            .reset_index(drop=True)
        )

        print(
            f"Images available locally: "
            f"{len(self.df)} samples"
        )

        if len(self.df) == 0:
            raise RuntimeError(
                "\nNo valid images were found.\n"
                "Check your image_root and image paths."
            )

    # --------------------------------------------------
    # Number of samples
    # --------------------------------------------------

    def __len__(self):
        return len(self.df)

    # --------------------------------------------------
    # Find image
    # --------------------------------------------------

    def _find_image(self, row):
        """
        Find the image belonging to a sample.

        Search order:
        1. image_path from CSV
        2. image_root/sample_id
        3. recursive search inside image_root
        """

        sample_id = str(
            row["sample_id"]
        ).strip()

        # --------------------------------------------------
        # 1. Try image_path from CSV
        # --------------------------------------------------

        if "image_path" in self.df.columns:

            csv_path = str(
                row["image_path"]
            ).strip()

            if (
                csv_path
                and csv_path.lower() != "nan"
            ):

                path = Path(csv_path)

                # Absolute path
                if (
                    path.is_absolute()
                    and path.exists()
                ):
                    return path

                # Relative to project root
                project_root = (
                    Path(__file__)
                    .resolve()
                    .parent
                    .parent
                )

                candidate = (
                    project_root / path
                )

                if candidate.exists():
                    return candidate

                # Relative to image root
                if self.image_root:

                    candidate = (
                        self.image_root / path
                    )

                    if candidate.exists():
                        return candidate

        # --------------------------------------------------
        # 2. Search directly in image_root
        # --------------------------------------------------

        if self.image_root:

            extensions = [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
            ]

            for ext in extensions:

                candidate = (
                    self.image_root
                    / f"{sample_id}{ext}"
                )

                if candidate.exists():
                    return candidate

        # --------------------------------------------------
        # 3. Recursive search
        # --------------------------------------------------

        if (
            self.image_root
            and self.image_root.exists()
        ):

            extensions = [
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.bmp",
                "*.tif",
                "*.tiff",
            ]

            for ext in extensions:

                matches = list(
                    self.image_root.rglob(
                        f"{sample_id}{ext}"
                    )
                )

                if matches:
                    return matches[0]

        # --------------------------------------------------
        # Image not found
        # --------------------------------------------------

        raise FileNotFoundError(
            f"\nImage not found for sample: "
            f"{sample_id}\n"
            f"CSV image_path: "
            f"{row.get('image_path', 'N/A')}\n"
            f"Image root: "
            f"{self.image_root}\n"
        )

    # --------------------------------------------------
    # Load and preprocess image
    # --------------------------------------------------

    def _load_image(self, image_path, form_id=None):

        # Read grayscale
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"OpenCV could not read image:\n"
                f"{image_path}"
            )

        # --------------------------------------------------
        # Crop pure handwriting (exclude printed prompt text)
        # --------------------------------------------------
        if self.crop_handwriting_only and form_id is not None:
            image = load_handwriting_crop(form_id, image, padding=self.padding)

        # --------------------------------------------------
        # Resize to 224 × 224 via shared INTER_AREA helper
        # --------------------------------------------------

        image = resize_handwriting_crop(
            image,
            size=self.image_size
        )

        # --------------------------------------------------
        # Create ink mask
        # --------------------------------------------------

        _, binary = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU
        )

        # --------------------------------------------------
        # Convert grayscale → RGB
        # ResNet50 expects 3 channels
        # --------------------------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        # --------------------------------------------------
        # Convert image to tensor
        # --------------------------------------------------

        image_tensor = torch.from_numpy(
            image_rgb
        ).float() / 255.0

        # HWC → CHW
        image_tensor = (
            image_tensor.permute(2, 0, 1)
        )

        # --------------------------------------------------
        # Convert ink mask to tensor
        # --------------------------------------------------

        mask_tensor = torch.from_numpy(
            binary
        ).float() / 255.0

        mask_tensor = (
            mask_tensor.unsqueeze(0)
        )

        return (
            image_tensor,
            mask_tensor
        )

    # --------------------------------------------------
    # Get one sample
    # --------------------------------------------------

    def __getitem__(self, index):

        row = self.df.iloc[index]

        sample_id = str(
            row["sample_id"]
        ).strip()

        # --------------------------------------------------
        # Find image
        # --------------------------------------------------

        image_path = self._find_image(row)

        # --------------------------------------------------
        # Load image (cropped to pure handwriting)
        # --------------------------------------------------

        form_id = str(
            row.get("form_id", sample_id)
        ).strip()

        image_tensor, mask_tensor = (
            self._load_image(
                image_path,
                form_id=form_id
            )
        )

        # --------------------------------------------------
        # Get 9 OpenCV features
        # --------------------------------------------------

        features = (
            row[self.FEATURE_COLUMNS]
            .values
            .astype(np.float32)
        )

        opencv_tensor = torch.tensor(
            features,
            dtype=torch.float32
        )

        # --------------------------------------------------
        # Return sample
        # --------------------------------------------------

        return {
            "image": image_tensor,

            "opencv_features":
                opencv_tensor,

            "ink_mask":
                mask_tensor,

            "sample_id":
                sample_id,

            "image_path":
                str(image_path),
        }


def crop_handwriting_region(
    image,
    form_id,
    padding=30,
    bboxes_csv=None
):
    """
    Crops an IAM form image to only the genuine handwriting region,
    discarding printed prompt headers, instructions, and name footers.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB/BGR).
    form_id : str
        IAM form identifier (e.g. 'a01-000u').
    padding : int
        Padding in pixels around handwriting bounds.
    bboxes_csv : str or Path, optional
        Path to form_handwriting_bboxes.csv.

    Returns
    -------
    np.ndarray
        Cropped image containing only the handwriting.
    """

    project_root = Path(__file__).resolve().parent.parent
    bbox_path = (
        Path(bboxes_csv)
        if bboxes_csv
        else project_root / "data" / "metadata" / "form_handwriting_bboxes.csv"
    )

    if not bbox_path.exists():
        return image

    bbox_df = pd.read_csv(bbox_path)
    match = bbox_df[
        bbox_df["form_id"] == str(form_id).strip()
    ]

    if match.empty:
        return image

    row = match.iloc[0]
    img_h, img_w = image.shape[:2]

    x1 = max(0, int(row["x1"]) - padding)
    y1 = max(0, int(row["y1"]) - padding)
    x2 = min(img_w, int(row["x2"]) + padding)
    y2 = min(img_h, int(row["y2"]) + padding)

    if y2 > y1 and x2 > x1:
        return image[y1:y2, x1:x2]

    return image


def extract_intra_page_cues(crop_gray: np.ndarray) -> np.ndarray:
    """
    Extract 8-D intra-page regional consistency cues from a native handwriting crop `crop_gray`.
    Normalized within the page (against the page's own patch median and IQR) to eliminate
    between-writer baseline variance while capturing localized motor and stroke anomalies:
      0: blur / edge sharpness drop (Laplacian-to-Sobel ratio)
      1: dropout / line-band void spike
      2: thickening / morphological terrace spike
      3: thickening / stroke radius spike
      4: local minimum plateau ratio
      5: thinning / faint stroke halo spike
      6: thinning / stroke radius drop
      7: local stroke intensity spike
    Purely image-based (no XML, no mask lookup, no parent lookup).
    """
    img_f = crop_gray.astype(np.float32)
    _, bin_inv = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (bin_inv > 0).astype(np.float32)

    lap = np.abs(cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)) * ink
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = (np.sqrt(gx * gx + gy * gy) + 1.0) * ink

    dist = cv2.distanceTransform(bin_inv, cv2.DIST_L2, 3) * ink
    dh = np.zeros_like(crop_gray, dtype=np.float32)
    dv = np.zeros_like(crop_gray, dtype=np.float32)
    dh[:, :-1] = np.abs(img_f[:, 1:] - img_f[:, :-1])
    dv[:-1, :] = np.abs(img_f[1:, :] - img_f[:-1, :])
    flat_ink = (((dh == 0) | (dv == 0)) & (crop_gray < 180) & (crop_gray > 5)).astype(np.float32) * ink

    er3 = cv2.erode(crop_gray, np.ones((3, 3), np.uint8)).astype(np.float32)
    dil3 = cv2.dilate(crop_gray, np.ones((3, 3), np.uint8)).astype(np.float32)
    is_lmin = ((img_f == er3) & ((dil3 - er3) > 15.0)).astype(np.float32) * ink
    dil5_ink = (cv2.dilate(bin_inv, np.ones((5, 5), np.uint8)) > 0).astype(np.float32)
    faint_halo = ((img_f > 185.0) & (img_f < 245.0) & ((dil3 - er3) > 12.0)).astype(np.float32) * dil5_ink

    row_ink = ink.mean(axis=1, keepdims=True)
    active_line_mask = (row_ink > 0.015).astype(np.float32)
    pure_white_in_line = ((crop_gray == 255).astype(np.float32)) * active_line_mask

    win = (80, 80)
    st = 24
    ink_sum = cv2.boxFilter(ink, ddepth=-1, ksize=win, normalize=False)[::st, ::st]
    halo_sum = cv2.boxFilter(dil5_ink, ddepth=-1, ksize=win, normalize=False)[::st, ::st]
    line_sum = cv2.boxFilter(active_line_mask * np.ones_like(ink), ddepth=-1, ksize=win, normalize=False)[::st, ::st]

    valid_line = line_sum > (0.5 * win[0] * win[1])
    if np.sum(valid_line) >= 4:
        w_white = cv2.boxFilter(pure_white_in_line, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid_line] / (line_sum[valid_line] + 1e-5)
        w_med = np.median(w_white)
        w_iqr = np.percentile(w_white, 75) - np.percentile(w_white, 25) + 1e-4
        z_white_max = float(np.sort((w_white - w_med) / w_iqr)[-2:].mean())
    else:
        z_white_max = 0.0

    valid = ink_sum > 100.0
    if np.sum(valid) < 6:
        return np.zeros(8, dtype=np.float32)
    denom = ink_sum[valid] + 1e-5
    h_denom = halo_sum[valid] + 1e-5

    p_lap = cv2.boxFilter(lap, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid] / (
        cv2.boxFilter(grad, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid] + 1e-5
    )
    p_flat = cv2.boxFilter(flat_ink, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid] / denom
    p_dist = cv2.boxFilter(dist, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid] / denom
    p_lmin = cv2.boxFilter(is_lmin, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid] / denom
    p_halo = cv2.boxFilter(faint_halo, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid] / h_denom
    p_gray = cv2.boxFilter(img_f * ink, ddepth=-1, ksize=win, normalize=False)[::st, ::st][valid] / denom

    P = np.stack([p_lap, p_flat, p_dist, p_lmin, p_halo, p_gray], axis=1)
    med = np.median(P, axis=0)
    iqr = np.percentile(P, 75, axis=0) - np.percentile(P, 25, axis=0) + 1e-4
    Z = (P - med) / iqr
    q_hi = np.sort(Z, axis=0)[-2:].mean(axis=0)
    q_lo = np.sort(-Z, axis=0)[-2:].mean(axis=0)

    return np.array([
        q_lo[0],
        z_white_max,
        q_hi[1],
        q_hi[2],
        q_hi[3],
        q_hi[4],
        q_lo[2],
        q_hi[5],
    ], dtype=np.float32)