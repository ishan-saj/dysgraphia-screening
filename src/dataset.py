from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


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
        self.handwriting_bboxes = {}

        if self.crop_handwriting_only:
            project_root = Path(__file__).resolve().parent.parent
            bbox_path = (
                Path(bboxes_csv)
                if bboxes_csv
                else project_root / "data" / "metadata" / "form_handwriting_bboxes.csv"
            )
            if bbox_path.exists():
                bbox_df = pd.read_csv(bbox_path)
                for _, r in bbox_df.iterrows():
                    self.handwriting_bboxes[str(r["form_id"]).strip()] = (
                        int(r["x1"]),
                        int(r["y1"]),
                        int(r["x2"]),
                        int(r["y2"]),
                    )
                print(
                    f"Loaded {len(self.handwriting_bboxes)} handwriting bounding boxes for cropping."
                )
            else:
                print(
                    f"Notice: Bbox file not found at {bbox_path}. Running without handwriting crop."
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
        lookup_key = str(form_id).strip() if form_id else None
        if (
            self.crop_handwriting_only
            and lookup_key
            and lookup_key in self.handwriting_bboxes
        ):
            x1, y1, x2, y2 = self.handwriting_bboxes[lookup_key]
            img_h, img_w = image.shape[:2]
            crop_x1 = max(0, x1 - self.padding)
            crop_y1 = max(0, y1 - self.padding)
            crop_x2 = min(img_w, x2 + self.padding)
            crop_y2 = min(img_h, y2 + self.padding)
            if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                image = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # --------------------------------------------------
        # Resize to 224 × 224
        # --------------------------------------------------

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_AREA
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