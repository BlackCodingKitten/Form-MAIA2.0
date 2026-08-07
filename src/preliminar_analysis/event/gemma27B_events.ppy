from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "google/gemma-3-27b-it"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("Nessun oggetto JSON trovato nella risposta.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                return json.loads(candidate)

    # Ultimo tentativo con json-repair, se installato.
    try:
        from json_repair import repair_json

        repaired = repair_json(text[start:], return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    raise ValueError("Oggetto JSON incompleto nella risposta del modello.")


def compact_semantic_input(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Mantiene soltanto le informazioni utili all'event detection.
    Evita di passare al modello metadati e testo superfluo.
    """
    compact_segments = []

    for segment in payload.get("segments", []):
        compact_segments.append(
            {
                "segment_id": segment.get("segment_id"),
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "entities": segment.get("entities", []),
                "actions": segment.get("actions", []),
                "events": segment.get("events", []),
                "state_changes": segment.get("state_changes", []),
                "temporal_relations": segment.get("temporal_relations", []),
                "spatial_relations": segment.get("spatial_relations", []),
            }
        )

    return {
        "id_video": payload.get("id_video"),
        "model": payload.get("model"),
        "segments": compact_segments,
    }


def build_event_prompt(payload: dict[str, Any]) -> str:
    schema = {
        "id_video": "string",
        "events": [
            {
                "event_id": "E0001",
                "description": "string",
                "start_time": 0.0,
                "end_time": 0.0,
                "participants": ["string"],
                "evidence_segments": ["segment_id"],
                "evidence_type": "observed|inferred|uncertain",
                "confidence": 0.0,
            }
        ],
        "temporal_relations": [
            {
                "first_event": "E0001",
                "relation": "before|after|overlaps|simultaneous|during",
                "second_event": "E0002",
                "confidence": 0.0,
            }
        ],
    }

    return (
        "You are performing event detection for a scientific video-analysis pipeline.\n"
        "The input is the semantic analysis produced by the SAME model, Gemma-3-27B-it.\n"
        "Do not use information from Qwen, OneLLM, Gemma-3n, or any other model.\n\n"
        "TASK:\n"
        "1. Identify the distinct events represented across the temporal segments.\n"
        "2. Merge duplicate descriptions of the same event caused by overlapping windows.\n"
        "3. Preserve genuinely distinct consecutive events.\n"
        "4. Assign stable event IDs in temporal order: E0001, E0002, ...\n"
        "5. Recover start/end times only from the supplied semantic evidence.\n"
        "6. List participants only when supported by the semantic evidence.\n"
        "7. Mark evidence_type='observed' for directly described events; "
        "'inferred' only when the semantic input itself explicitly marks an inference; "
        "otherwise use 'uncertain'.\n"
        "8. Create temporal relations only when supported by timestamps or explicit "
        "temporal relations in the input.\n"
        "9. Do not invent events, causes, intentions, or missing actions.\n"
        "10. Return ONLY one valid JSON object. No markdown and no explanation.\n\n"
        "OUTPUT SCHEMA:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "SEMANTIC INPUT:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


class Gemma27EventInferencer:
    def __init__(
        self,
        model_id: str,
        devices: str,
        max_memory_gib: int,
        max_new_tokens: int,
    ) -> None:
        # Esempio: --devices 1,2.
        # Nel processo diventano cuda:0 e cuda:1.
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Gemma3ForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Installa transformers, accelerate, bitsandbytes e torch."
            ) from error

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.physical_devices = [
            item.strip()
            for item in devices.split(",")
            if item.strip()
        ]

        if not self.physical_devices:
            raise ValueError("Specifica almeno una GPU con --devices.")

        if torch.cuda.device_count() != len(self.physical_devices):
            raise RuntimeError(
                f"GPU fisiche richieste: {self.physical_devices}; "
                f"GPU visibili a PyTorch: {torch.cuda.device_count()}."
            )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        max_memory = {
            index: f"{max_memory_gib}GiB"
            for index in range(len(self.physical_devices))
        }

        print(
            f"Caricamento {model_id} in 4-bit sulle GPU fisiche "
            f"{','.join(self.physical_devices)}"
        )

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="balanced",
            max_memory=max_memory,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.input_device = torch.device("cuda:0")

        print("Device map:")
        for module_name, device in self.model.hf_device_map.items():
            print(f"  {module_name or '<root>'}: {device}")

    def __call__(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a scientific event extraction system. "
                            "Return only valid JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.input_device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[-1]

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated = output[0][input_length:]

        return self.processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


def analyze_video(
    semantic_path: Path,
    output_directory: Path,
    inferencer: Gemma27EventInferencer,
    overwrite: bool,
    keep_raw: bool,
) -> dict[str, Any]:
    semantic_payload = json.loads(
        semantic_path.read_text(encoding="utf-8")
    )

    video_id = str(
        semantic_payload.get("id_video")
        or semantic_path.stem.removesuffix("_semantic")
    )

    output_path = output_directory / f"{video_id}_events.json"

    if output_path.exists() and not overwrite:
        return {
            "id_video": video_id,
            "status": "skipped",
            "numero_eventi": None,
            "output": str(output_path),
        }

    compact_payload = compact_semantic_input(semantic_payload)
    prompt = build_event_prompt(compact_payload)

    raw_response = inferencer(prompt)
    result = extract_json(raw_response)

    result["id_video"] = video_id
    result["model"] = DEFAULT_MODEL
    result["source_semantic_file"] = str(semantic_path)

    if keep_raw:
        result["raw_response"] = raw_response

    write_json(output_path, result)

    return {
        "id_video": video_id,
        "status": "completed",
        "numero_eventi": len(result.get("events", [])),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Event detection con google/gemma-3-27b-it "
            "distribuito su più GPU."
        )
    )

    parser.add_argument(
        "semantic_directory",
        nargs="?",
        type=Path,
        default="data/preliminar_analysis/semantic_analysis/gemma27",
    )
    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default="data/preliminar_analysis/event_analysis/gemma27",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)

    parser.add_argument(
        "--devices",
        default="1,2",
        help="GPU fisiche separate da virgola. Default: 1,2.",
    )

    parser.add_argument(
        "--max-memory-gib",
        type=int,
        default=40,
        help="Memoria massima utilizzabile per GPU.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Numero massimo di video; 0 = tutti.",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")

    args = parser.parse_args()

    semantic_files = sorted(
        args.semantic_directory.glob("*_semantic.json")
    )

    if args.limit_videos > 0:
        semantic_files = semantic_files[: args.limit_videos]

    if not semantic_files:
        parser.error(
            f"Nessun *_semantic.json trovato in {args.semantic_directory}"
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    inferencer = Gemma27EventInferencer(
        model_id=args.model,
        devices=args.devices,
        max_memory_gib=args.max_memory_gib,
        max_new_tokens=args.max_new_tokens,
    )

    rows = []

    for index, semantic_path in enumerate(semantic_files, start=1):
        print(
            f"[{index}/{len(semantic_files)}] "
            f"Event detection di {semantic_path.name}"
        )

        try:
            rows.append(
                analyze_video(
                    semantic_path=semantic_path,
                    output_directory=args.output_directory,
                    inferencer=inferencer,
                    overwrite=args.overwrite,
                    keep_raw=args.keep_raw,
                )
            )

        except Exception as error:
            print(
                f"Errore durante {semantic_path.name}: "
                f"{type(error).__name__}: {error}"
            )

            rows.append(
                {
                    "id_video": semantic_path.stem,
                    "status": "failed",
                    "numero_eventi": None,
                    "output": f"{type(error).__name__}: {error}",
                }
            )

    write_csv(
        args.output_directory / "riepilogo_eventi.csv",
        rows,
    )

    print(f"Risultati salvati in: {args.output_directory}")


if __name__ == "__main__":
    main()