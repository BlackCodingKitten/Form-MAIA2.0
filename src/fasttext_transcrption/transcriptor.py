import os
import csv
from pathlib import Path

# Use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Use a local writable Hugging Face cache
HF_CACHE = Path.cwd() / ".hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")

from faster_whisper import WhisperModel


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_FOLDER = Path("data/video")
OUTPUT_CSV = Path("data/transcription/transcriptions.csv")

VIDEO_EXTENSIONS = {".mp4"}

MODEL_NAME = "large-v3"

# Use "it" for Italian or None for automatic language detection
LANGUAGE = "it"


# ============================================================
# LOAD THE MODEL ON GPU 0
# ============================================================

model = WhisperModel(
    MODEL_NAME,
    device="cuda",
    device_index=0,
    compute_type="float16",
    download_root=str(HF_CACHE / "hub"),
)


def find_videos(folder: Path) -> list[Path]:
    """Return all supported video files found recursively."""

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def transcribe_video(video_path: Path) -> str:
    """Transcribe the audio track of a video."""

    segments, _ = model.transcribe(
        str(video_path),
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )


def main() -> None:
    if not VIDEO_FOLDER.exists():
        raise FileNotFoundError(
            f"Video folder not found: {VIDEO_FOLDER}"
        )

    videos = find_videos(VIDEO_FOLDER)

    if not videos:
        print(f"No videos found in: {VIDEO_FOLDER}")
        return

    print(f"Videos found: {len(videos)}")

    # Dictionary structure:
    # {"video001.mp4": "transcription...", ...}
    transcriptions: dict[str, str] = {}

    for index, video_path in enumerate(videos, start=1):
        print(
            f"[{index}/{len(videos)}] "
            f"Transcribing {video_path.name}"
        )

        try:
            transcription = transcribe_video(video_path)
            transcriptions[video_path.name] = transcription

            if transcription:
                print("Transcription completed.")
            else:
                print("No speech detected.")

        except Exception as error:
            print(f"Transcription error: {error}")
            transcriptions[video_path.name] = f"ERROR: {error}"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=["video_name", "transcription"],
            delimiter=";",
            # quotechar='',
            # quoting=csv.QUOTE_ALL,
        )

        writer.writeheader()

        writer.writerows(
            {
                "video_name": video_name,
                "transcription": transcription,
            }
            for video_name, transcription in transcriptions.items()
        )

    print(f"\nCSV created: {OUTPUT_CSV}")
   


if __name__ == "__main__":
    main()