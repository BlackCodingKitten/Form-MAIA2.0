import json
import os
import re
from pathlib import Path

# Same two GPUs used by the semantic stage; run this script after semantic extraction.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
INPUT_FILE = Path("data/semantic/qwen3_omni_30b_semantic.json")
OUTPUT_FILE = Path("data/events/qwen3_omni_30b_events.json")
MAX_NEW_TOKENS = 1800

EVENT_PROMPT = """You are given the semantic analysis of overlapping windows from ONE video.
The windows are 4 seconds long with a 3-second stride, so adjacent windows share content.
Consolidate them into a single chronological event representation.

Return one valid JSON object and nothing else, using exactly this schema:
{
  "events": [
    {
      "event_id": "E1",
      "start": 0.0,
      "end": 0.0,
      "description": "...",
      "entities": ["..."],
      "actions": ["..."],
      "state_changes": ["..."],
      "evidence_windows": [0]
    }
  ],
  "temporal_relations": [
    {"event_1": "E1", "relation": "before|after|overlaps|during|same_event", "event_2": "E2"}
  ]
}

Rules:
- Merge duplicate descriptions caused by overlapping windows.
- One real-world occurrence must correspond to one event, even if it appears in multiple windows.
- Use the window timestamps as evidence; do not fabricate precise times finer than the available windows.
- Keep distinct repeated actions as distinct events when the evidence shows they happened more than once.
- Preserve state changes inside the event that causes them.
- Temporal relations must be supported by timestamps or explicit semantic evidence.
- Do NOT perform the later inference-analysis stage here: do not add intentions, explanations, hidden causes, or unsupported causal conclusions.
- If no reliable event is available, return {"events": [], "temporal_relations": []}.

WINDOW DATA:
"""


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise


def load_existing():
    if not OUTPUT_FILE.exists():
        return {}
    with OUTPUT_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    return {item["video"]: item for item in data}


def save(results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)


with INPUT_FILE.open(encoding="utf-8") as f:
    semantic_videos = json.load(f)

print(f"Loading {MODEL_ID} on visible GPUs {os.environ['CUDA_VISIBLE_DEVICES']}...")
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "41GiB", 1: "41GiB", "cpu": "120GiB"},
    attn_implementation="flash_attention_2",
)
model.disable_talker()
model.eval()
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)

results = load_existing()

for i, item in enumerate(semantic_videos, 1):
    video_name = item["video"]
    if video_name in results:
        print(f"[{i}/{len(semantic_videos)}] SKIP {video_name}")
        continue

    print(f"[{i}/{len(semantic_videos)}] {video_name}")

    usable_windows = [
        {
            "window_id": w["window_id"],
            "start": w["start"],
            "end": w["end"],
            "semantic": w["semantic"],
        }
        for w in item["windows"]
        if w.get("semantic") is not None
    ]

    prompt = EVENT_PROMPT + json.dumps(usable_windows, ensure_ascii=False, separators=(",", ":"))
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=chat_text, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated, _ = model.generate(
            **inputs,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    answer = processor.batch_decode(
        generated.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    try:
        event_data = parse_json(answer)
        error = None
    except Exception as exc:
        event_data = None
        error = f"{type(exc).__name__}: {exc}"

    results[video_name] = {
        "video": video_name,
        "model": MODEL_ID,
        "events": None if event_data is None else event_data.get("events", []),
        "temporal_relations": None if event_data is None else event_data.get("temporal_relations", []),
        "error": error,
        "raw_output": None if event_data is not None else answer,
    }
    save(results)

print(f"Done: {len(results)} videos -> {OUTPUT_FILE}")
