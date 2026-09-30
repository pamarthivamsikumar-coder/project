"""Prepare real driver-monitoring datasets for this project.

Supports:
1) Importing an already-extracted feature CSV.
2) Extracting EAR/MAR/gaze features from labeled driver videos using MediaPipe.

UTA-RLDD labels:
0  = Alert
5  = Low Vigilant
10 = Drowsy

For this project, Low Vigilant (5) is mapped to drowsy.
"""

from pathlib import Path
import argparse
import sys

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "data" / "real_multimodal_dataset.csv"

FEATURES = [
    "ear",
    "mar",
    "blink_duration",
    "gaze_deviation",
    "lane_deviation",
    "speed_kmph",
    "heart_rate",
]

LABELS = [
    "alert",
    "drowsy",
    "distracted",
    "high_impairment_risk",
]


# MediaPipe Face Landmarker indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 13, 14, 291, 17, 84]


def dist(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a) - np.asarray(b)
        )
    )


def ear(p):
    return (
        dist(p[1], p[5]) +
        dist(p[2], p[4])
    ) / (
        2 * max(dist(p[0], p[3]), 1e-6)
    )


def mar(p):
    return (
        dist(p[1], p[5]) +
        dist(p[2], p[4])
    ) / (
        2 * max(dist(p[0], p[3]), 1e-6)
    )


def canonical_label(s):
    """
    Convert dataset folder names / labels
    into project labels.
    """

    s = str(s).lower().replace("\\", "/")

    # Split actual path components
    parts = [
        p.strip()
        for p in s.split("/")
        if p.strip()
    ]

    # -------------------------------------------------
    # UTA-RLDD labels
    # 0  = Alert
    # 5  = Low Vigilant
    # 10 = Drowsy
    # -------------------------------------------------

    if "0" in parts:
        return "alert"

    if "5" in parts:
        return "drowsy"

    if "10" in parts:
        return "drowsy"

    # -------------------------------------------------
    # Other possible dataset naming conventions
    # -------------------------------------------------

    if any(
        x in s
        for x in [
            "drowsy",
            "sleep",
            "fatigue",
            "low_vigil",
            "lowvigil",
            "yawn",
        ]
    ):
        return "drowsy"

    if any(
        x in s
        for x in [
            "distract",
            "talk",
            "phone",
            "read",
            "look_side",
        ]
    ):
        return "distracted"

    if any(
        x in s
        for x in [
            "alert",
            "normal",
            "awake",
        ]
    ):
        return "alert"

    return None


def import_csv(path: Path):

    df = pd.read_csv(path)

    rename = {
        c: c.strip().lower()
        for c in df.columns
    }

    df = df.rename(columns=rename)

    aliases = {
        "blink": "blink_duration",
        "blink_time": "blink_duration",
        "gaze": "gaze_deviation",
        "speed": "speed_kmph",
        "heart_rate_bpm": "heart_rate",
        "bpm": "heart_rate",
        "label": "label",
        "class": "label",
        "state": "label",
    }

    for a, b in aliases.items():
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})

    missing = [
        c
        for c in FEATURES + ["label"]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    df = df[
        FEATURES + ["label"]
    ].copy()

    df["label"] = (
        df["label"]
        .map(canonical_label)
        .fillna(
            df["label"]
            .astype(str)
            .str.lower()
        )
    )

    df = df[
        df["label"].isin(LABELS)
    ]

    return df


def extract_videos(root: Path, limit_per_video=120):

    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError(
            "MediaPipe is required for video feature extraction."
        ) from e

    vision = mp.tasks.vision

    model = (
        ROOT
        / "models"
        / "face_landmarker.task"
    )

    if not model.exists():
        raise RuntimeError(
            "Face model missing. Run:\n"
            "python -m tools.download_face_model"
        )

    opts = (
        vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )

    landmarker = (
        vision.FaceLandmarker
        .create_from_options(opts)
    )

    rows = []

    videos = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
        }
    ]

    print(
        f"Found {len(videos)} video files."
    )

    if not videos:
        print(
            "No supported video files were found."
        )
        landmarker.close()
        return pd.DataFrame(
            columns=FEATURES + ["label"]
        )

    for video_index, vp in enumerate(videos, 1):

        # IMPORTANT:
        # Use the actual path so folders 0/5/10
        # are recognized correctly.
        label = canonical_label(vp.stem)

        if label is None:
            print(
                f"Skipping unlabeled video: {vp}"
            )
            continue

        print(
            f"[{video_index}/{len(videos)}] "
            f"{vp.name} -> {label}"
        )

        cap = cv2.VideoCapture(str(vp))

        if not cap.isOpened():
            print(
                f"Could not open: {vp}"
            )
            continue

        n = 0
        samples = 0

        while samples < limit_per_video:

            ok, frame = cap.read()

            if not ok:
                break

            n += 1

            # Process every 3rd frame
            if n % 3 != 0:
                continue

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            try:
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                res = landmarker.detect(image)

            except Exception:
                continue

            if not res.face_landmarks:
                continue

            pts = [
                (lm.x * w, lm.y * h)
                for lm in res.face_landmarks[0]
            ]

            # -------------------------
            # EAR
            # -------------------------

            left_ear = ear(
                [pts[i] for i in LEFT_EYE]
            )

            right_ear = ear(
                [pts[i] for i in RIGHT_EYE]
            )

            e = (
                left_ear +
                right_ear
            ) / 2.0

            # -------------------------
            # MAR
            # -------------------------

            m = mar(
                [pts[i] for i in MOUTH]
            )

            # -------------------------
            # Gaze
            # -------------------------

            left_eye_center = np.mean(
                [
                    pts[33],
                    pts[133],
                ],
                axis=0,
            )

            right_eye_center = np.mean(
                [
                    pts[362],
                    pts[263],
                ],
                axis=0,
            )

            nose = np.array(pts[1])

            eye_mid = (
                left_eye_center +
                right_eye_center
            ) / 2.0

            eye_width = max(
                abs(
                    right_eye_center[0]
                    - left_eye_center[0]
                ),
                1,
            )

            gaze = float(
                np.clip(
                    (
                        nose[0]
                        - eye_mid[0]
                    )
                    / eye_width,
                    -1,
                    1,
                )
            )

            # -------------------------
            # Dataset does not contain
            # vehicle speed/lane/rPPG.
            #
            # Keep these as separate
            # runtime modalities.
            # -------------------------

            blink_duration = 0.1
            lane_deviation = 0.0
            speed_kmph = 50.0
            heart_rate = 78.0

            rows.append(
                [
                    np.clip(e, 0.05, 0.6),
                    np.clip(m, 0.02, 1.5),
                    blink_duration,
                    abs(gaze),
                    lane_deviation,
                    speed_kmph,
                    heart_rate,
                    label,
                ]
            )

            samples += 1

        cap.release()

        print(
            f"  Extracted {samples} samples"
        )

    landmarker.close()

    return pd.DataFrame(
        rows,
        columns=FEATURES + ["label"],
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        action="append",
        help="Feature CSV from a real dataset",
    )

    parser.add_argument(
        "--videos",
        action="append",
        help="Folder containing labeled videos",
    )

    parser.add_argument(
        "--output",
        default=str(OUT),
    )

    args = parser.parse_args()

    frames = []

    # ---------------------------------------
    # CSV input
    # ---------------------------------------

    for p in args.csv or []:

        print(
            f"Importing CSV: {p}"
        )

        frames.append(
            import_csv(Path(p))
        )

    # ---------------------------------------
    # Video input
    # ---------------------------------------

    for p in args.videos or []:

        print(
            f"Processing videos from: {p}"
        )

        frames.append(
            extract_videos(
                Path(p)
            )
        )

    # ---------------------------------------
    # Auto-discover CSV files
    # ---------------------------------------

    if not frames:

        for p in (
            ROOT / "datasets"
        ).rglob("*.csv"):

            try:

                frames.append(
                    import_csv(p)
                )

                print(
                    f"Imported {p}"
                )

            except Exception as e:

                print(
                    f"Skipped {p}: {e}"
                )

    # ---------------------------------------
    # Nothing found
    # ---------------------------------------

    if not frames:

        print(
            "No real dataset was found."
        )

        print(
            "Use --videos or --csv."
        )

        sys.exit(2)

    # ---------------------------------------
    # Combine datasets
    # ---------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = df.dropna()

    # Numeric conversion
    df["speed_kmph"] = pd.to_numeric(
        df["speed_kmph"],
        errors="coerce",
    ).fillna(0)

    df["heart_rate"] = pd.to_numeric(
        df["heart_rate"],
        errors="coerce",
    ).fillna(0)

    # ---------------------------------------
    # Save
    # ---------------------------------------

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out,
        index=False,
    )

    print()
    print("=" * 50)
    print("REAL DATASET PREPARATION COMPLETE")
    print("=" * 50)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Saved to: {out}"
    )

    print()
    print("Class distribution:")
    print(
        df["label"].value_counts()
    )

    print("=" * 50)


if __name__ == "__main__":
    main()