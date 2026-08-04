import subprocess
from pathlib import Path
from typing import Optional, Union


INPUT_FOLDER = Path("data/video")
OUTPUT_FOLDER = Path("data/audio")
VIDEO_EXTENSIONS = {".mp4"}


def mp4_to_mp3(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    bitrate: str = "192k",
) -> Path:
    """Extract the audio from an MP4 file and save it as MP3."""

    input_path = Path(input_path)
    output_path = (
        input_path.with_suffix(".mp3")
        if output_path is None
        else Path(output_path)
    )

    # Create the output directory if necessary
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(output_path),
        ],
        check=True,
    )

    return output_path


def find_videos(folder: Path) -> list[Path]:
    """Return all supported video files found recursively."""

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    videos = find_videos(INPUT_FOLDER)

    if not videos:
        print(f"No videos found in: {INPUT_FOLDER}")
        return

    print(f"Videos found: {len(videos)}")

    for index, video_path in enumerate(videos, start=1):
        # Preserve the relative folder structure and video file name
        relative_path = video_path.relative_to(INPUT_FOLDER)
        audio_path = OUTPUT_FOLDER / relative_path.with_suffix(".mp3")

        print(
            f"[{index}/{len(videos)}] "
            f"{video_path.name} -> {audio_path.name}"
        )

        try:
            mp4_to_mp3(video_path, audio_path)
            print("Audio extraction completed.")

        except subprocess.CalledProcessError as error:
            print(f"FFmpeg error: {error}")

    print(f"\nAudio files created in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()