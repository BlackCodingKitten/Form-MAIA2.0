import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

# Use the two free 43-GB GPUs. Inside the process they become cuda:0 and cuda:1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
VIDEO_DIR = Path("data/video")
OUTPUT_FILE = Path("data/semantic/qwen3_omni_30b_semantic.json")

WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 3.0
VIDEO_FPS = 2  # 4 s x 2 fps = at most 8 frames/window
MAX_NEW_TOKENS = 1200

SEMANTIC_PROMPT = """Analyze ONLY the supplied audio-video window.
Return one valid JSON object and nothing else.
Do not invent information that is not supported by the visual or acoustic evidence.
Use concise natural-language descriptions.

Required schema:
{
  "entities": ["..."],
  "actions": ["..."],
  "events": ["..."],
  "spatial_relations": ["..."],
  "state_changes": ["..."],
  "temporal_relations": ["..."],
  "causal_hypotheses": ["..."]
}

Rules:
- entities: people, objects, places, visible or clearly audible agents/sources.
- actions: directly observable actions.
- events: meaningful occurrences in this window, combining actors/actions/objects when possible.
- spatial_relations: only relations supported by the video.
- state_changes: observable changes from one state to another.
- temporal_relations: ordering/overlap among events visible or audible in this same window.
- causal_hypotheses: include only causality strongly suggested by the local evidence; otherwise [].
- If a field has no reliable information, return [].
"""


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True)
    return float(out.strip())


def make_window(src: Path, dst: Path, start: float, length: float):
    # Re-encode so every 4-s window has exactly ~2 fps while preserving its audio.
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
        "-vf", f"fps={VIDEO_FPS}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(dst)
    ])


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise


def load_results():
    if not OUTPUT_FILE.exists():
        return {}
    with OUTPUT_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    return {item["video"]: item for item in data}


def save_results(results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)


print(f"Loading {MODEL_ID} on visible GPUs {os.environ['CUDA_VISIBLE_DEVICES']}...")
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "41GiB", 1: "41GiB", "cpu": "120GiB"},
    attn_implementation="flash_attention_2",
)
model.disable_talker()  # text only; saves about 10 GB of GPU memory
model.eval()
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)

results = load_results()
videos = sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"})

for video_index, video in enumerate(videos, 1):
    if video.name in results:
        print(f"[{video_index}/{len(videos)}] SKIP {video.name}")
        continue

    print(f"[{video_index}/{len(videos)}] {video.name}")
    video_duration = duration(video)
    windows = []

    with tempfile.TemporaryDirectory(prefix="qwen3_semantic_") as tmp:
        start = 0.0
        window_id = 0

        while start < video_duration:
            end = min(start + WINDOW_SECONDS, video_duration)
            clip = Path(tmp) / f"window_{window_id:03d}.mp4"
            make_window(video, clip, start, end - start)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(clip)},
                    {"type": "text", "text": SEMANTIC_PROMPT},
                ],
            }]

            chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, video_inputs = process_mm_info(messages, use_audio_in_video=True)
            inputs = processor(
                text=chat_text,
                audio=audios,
                images=images,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            with torch.inference_mode():
                generated, _ = model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=True,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

            answer = processor.batch_decode(
                generated.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            try:
                semantic = parse_json(answer)
                error = None
            except Exception as exc:
                semantic = None
                error = f"{type(exc).__name__}: {exc}"

            windows.append({
                "window_id": window_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "semantic": semantic,
                "error": error,
                "raw_output": None if semantic is not None else answer,
            })

            print(f"    window {window_id}: {start:.1f}-{end:.1f}s" + (" OK" if semantic is not None else " ERROR"))
            start += STRIDE_SECONDS
            window_id += 1

    results[video.name] = {
        "video": video.name,
        "duration": round(video_duration, 3),
        "model": MODEL_ID,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "fps": VIDEO_FPS,
        "windows": windows,
    }
    save_results(results)  # checkpoint after every video

print(f"Done: {len(results)} videos -> {OUTPUT_FILE}")
