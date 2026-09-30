from pathlib import Path
import argparse
import cv2
import pandas as pd


def get_label(video_path):
    folder = video_path.parent.name.lower()

    if "microsleep" in folder:
        return "microsleep"

    if "yawning" in folder:
        return "yawning"

    return "unknown"


def process_video(video_path, label, sample_every=30):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return []

    rows = []
    frame_no = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_no % sample_every == 0:
            rows.append({
                "video": video_path.name,
                "path": str(video_path),
                "frame": frame_no,
                "label": label,
                "width": frame.shape[1],
                "height": frame.shape[0]
            })

        frame_no += 1

    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True)
    parser.add_argument(
        "--output",
        default="data/nitymed_dataset.csv"
    )

    args = parser.parse_args()

    root = Path(args.videos)

    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    videos = sorted(
        list(root.rglob("*.mp4")) +
        list(root.rglob("*.MP4"))
    )

    if not videos:
        raise SystemExit(f"No MP4 videos found in {root}")

    print(f"Found {len(videos)} videos")

    all_rows = []

    for i, video in enumerate(videos, 1):

        label = get_label(video)

        print(
            f"[{i}/{len(videos)}] "
            f"{video.name} -> {label}"
        )

        rows = process_video(video, label)
        all_rows.extend(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(output, index=False)

    print()
    print("====================================")
    print("NITYMED PROCESSING COMPLETED")
    print("====================================")
    print("Videos:", len(videos))
    print("Rows:", len(df))
    print()
    print("Class distribution:")
    print(df["label"].value_counts())
    print()
    print("Saved:", output)
    print("====================================")


if __name__ == "__main__":
    main()