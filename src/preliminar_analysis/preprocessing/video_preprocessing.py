
import argparse
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
from scenedetect import ContentDetector, detect

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def save_json(data, output_path):
    """Write a Python object to a formatted JSON file."""
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_directory(directory):
    """Delete and recreate a directory to avoid stale output files."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def parse_fraction(value):
    """Convert an ffprobe fraction such as 25/1 into a float."""
    return float(Fraction(value)) if value and value != "0/0" else 0.0


def probe_video(video_path):
    """Read video and audio metadata with ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    video_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]

    duration = float(video_stream.get("duration") or probe["format"].get("duration") or 0)
    fps = parse_fraction(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))

    return {
        "file_name": video_path.name,
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "frame_count": int(video_stream["nb_frames"]) if video_stream.get("nb_frames") else None,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "video_codec": video_stream.get("codec_name"),
        "pixel_format": video_stream.get("pix_fmt"),
        "audio_present": bool(audio_streams),
        "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
    }


def resize_frame(frame, maximum_width):
    """Resize a frame only when it is wider than the configured limit."""
    if maximum_width and frame.shape[1] > maximum_width:
        scale = maximum_width / frame.shape[1]
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def extract_frames(video_path, timestamps, output_directory, prefix, maximum_width=1280):
    """Extract frames at requested timestamps and return a JSON-ready manifest."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    for index, timestamp in enumerate(timestamps):
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = capture.read()
        if not success:
            continue

        frame = resize_frame(frame, maximum_width)
        file_name = f"{prefix}_{index:04d}_t{timestamp:08.3f}.jpg"
        file_path = output_directory / file_name
        cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        frames.append({
            "frame_id": f"{prefix}_{index:04d}",
            "timestamp": round(timestamp, 3),
            "file": str(file_path),
        })

    capture.release()
    return frames


def uniform_timestamps(duration, frame_count):
    """Return timestamps located at the centre of equal temporal bins."""
    return [((index + 0.5) * duration) / frame_count for index in range(frame_count)]


def dense_timestamps(duration, sampling_fps):
    """Return regularly spaced timestamps for the dense tracking stream."""
    return [index / sampling_fps for index in range(math.ceil(duration * sampling_fps)) if index / sampling_fps < duration]


def detect_shots(video_path, fps, threshold, minimum_scene_seconds):
    """Detect shot boundaries with PySceneDetect ContentDetector."""
    detector = ContentDetector(
        threshold=threshold,
        min_scene_len=max(1, round(minimum_scene_seconds * fps)),
    )
    scene_list = detect(str(video_path), detector, start_in_scene=True, show_progress=False)

    return [
        {
            "shot_id": f"shot_{index:03d}",
            "start_time": round(start.get_seconds(), 3),
            "end_time": round(end.get_seconds(), 3),
        }
        for index, (start, end) in enumerate(scene_list)
    ]


def build_segments(shots, duration, minimum_duration, target_duration, maximum_duration, overlap):
    """Build 2-6 second segments, preferring detected shot boundaries near the target duration."""
    cuts = sorted({0.0, duration, *[shot["start_time"] for shot in shots], *[shot["end_time"] for shot in shots]})
    segments = []
    start = 0.0

    while duration - start > maximum_duration:
        candidates = [cut for cut in cuts if start + minimum_duration <= cut <= start + maximum_duration]
        end = min(candidates, key=lambda cut: abs(cut - (start + target_duration))) if candidates else start + target_duration
        segments.append((start, end))
        start = end

    if duration - start < minimum_duration and segments:
        segments[-1] = (segments[-1][0], duration)
    else:
        segments.append((start, duration))

    output = []
    for index, (raw_start, end) in enumerate(segments):
        start = raw_start if index == 0 else max(0.0, raw_start - overlap)
        output.append({
            "segment_id": f"segment_{index:03d}",
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(end - start, 3),
            "source_shots": [
                shot["shot_id"]
                for shot in shots
                if max(start, shot["start_time"]) < min(end, shot["end_time"])
            ],
        })

    return output


def segment_timestamps(start, end, frame_count):
    """Return equally spaced timestamps inside a temporal segment."""
    duration = end - start
    return [start + ((index + 0.5) * duration) / frame_count for index in range(frame_count)]


def process_video(video_path, output_root, arguments):
    """Run the complete preprocessing pipeline for one video."""
    video_output = output_root / video_path.stem
    global_directory = video_output / "global_frames"
    dense_directory = video_output / "dense_frames"
    segment_root = video_output / "segment_frames"

    video_output.mkdir(parents=True, exist_ok=True)
    reset_directory(global_directory)
    reset_directory(dense_directory)
    reset_directory(segment_root)

    metadata = probe_video(video_path)
    metadata["video_id"] = video_path.stem

    if metadata["duration"] <= 0 or metadata["fps"] <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    global_frames = extract_frames(
        video_path,
        uniform_timestamps(metadata["duration"], arguments.global_frames),
        global_directory,
        "global",
        arguments.maximum_width,
    )

    dense_frames = extract_frames(
        video_path,
        dense_timestamps(metadata["duration"], arguments.dense_fps),
        dense_directory,
        "dense",
        arguments.maximum_width,
    )

    shots = detect_shots(
        video_path,
        metadata["fps"],
        arguments.scene_threshold,
        arguments.minimum_scene_seconds,
    )

    segments = build_segments(
        shots,
        metadata["duration"],
        arguments.minimum_segment_seconds,
        arguments.target_segment_seconds,
        arguments.maximum_segment_seconds,
        arguments.segment_overlap_seconds,
    )

    for segment in segments:
        segment_directory = segment_root / segment["segment_id"]
        segment_directory.mkdir(parents=True, exist_ok=True)
        segment["frames"] = extract_frames(
            video_path,
            segment_timestamps(segment["start_time"], segment["end_time"], arguments.frames_per_segment),
            segment_directory,
            segment["segment_id"],
            arguments.maximum_width,
        )

    save_json(metadata, video_output / "metadata.json")
    save_json({"video_id": video_path.stem, "global_frames": global_frames, "dense_frames": dense_frames}, video_output / "frames.json")
    save_json({"video_id": video_path.stem, "shots": shots}, video_output / "shots.json")
    save_json({"video_id": video_path.stem, "segments": segments}, video_output / "segments.json")

    print(f"Processed: {video_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for MAIA temporal analysis.")
    parser.add_argument("input_directory", type=Path, help="Directory containing input videos.")
    parser.add_argument("output_directory", type=Path, help="Directory where preprocessing results are saved.")
    parser.add_argument("--global-frames", type=int, default=32)
    parser.add_argument("--dense-fps", type=float, default=4.0)
    parser.add_argument("--scene-threshold", type=float, default=27.0)
    parser.add_argument("--minimum-scene-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-segment-seconds", type=float, default=2.0)
    parser.add_argument("--target-segment-seconds", type=float, default=4.0)
    parser.add_argument("--maximum-segment-seconds", type=float, default=6.0)
    parser.add_argument("--segment-overlap-seconds", type=float, default=0.5)
    parser.add_argument("--frames-per-segment", type=int, default=8)
    parser.add_argument("--maximum-width", type=int, default=1280)
    arguments = parser.parse_args()

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    video_paths = sorted(path for path in arguments.input_directory.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)

    if not video_paths:
        raise FileNotFoundError(f"No videos found in {arguments.input_directory}")

    for video_path in video_paths:
        try:
            process_video(video_path, arguments.output_directory, arguments)
        except Exception as error:
            print(f"Failed: {video_path.name} -> {error}")


if __name__ == "__main__":
    main()