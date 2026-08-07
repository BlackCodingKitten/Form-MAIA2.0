from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEMANTIC_CATEGORIES = (
    "entities",
    "actions",
    "events",
    "spatial_relations",
    "state_changes",
    "temporal_relations",
    "causal_hypotheses",
)


@dataclass(frozen=True)
class TemporalWindow:
    """A temporally ordered subset of dense frames."""

    segment_id: str
    start_time: float
    end_time: float
    frame_paths: tuple[Path, ...]

    @property
    def evidence_frames(self) -> list[str]:
        return [path.name for path in self.frame_paths]


class SemanticOutputError(RuntimeError):
    """Raised when a model response cannot be converted to the common schema."""


def get_timestamp(image_path: Path) -> float:
    """Extract a timestamp from names such as dense_0001_t0000.250.jpg."""
    try:
        return float(image_path.stem.rsplit("_t", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"Nome frame non valido, timestamp assente: {image_path.name}"
        ) from error


def discover_video_directories(preprocessing_directory: Path) -> list[Path]:
    """Find video directories that contain a dense_frames subdirectory."""
    if not preprocessing_directory.exists():
        raise FileNotFoundError(
            f"Directory di preprocessing inesistente: {preprocessing_directory}"
        )

    return sorted(
        path
        for path in preprocessing_directory.iterdir()
        if path.is_dir() and (path / "dense_frames").is_dir()
    )


def list_dense_frames(video_directory: Path) -> list[Path]:
    """Return dense frames sorted by their timestamp."""
    dense_directory = video_directory / "dense_frames"
    frames = [
        path
        for path in dense_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(frames, key=get_timestamp)


def _select_evenly(items: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0:
        raise ValueError("max_frames deve essere maggiore di zero.")
    if len(items) <= maximum:
        return items
    if maximum == 1:
        return [items[len(items) // 2]]

    indexes = {
        round(index * (len(items) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return [items[index] for index in sorted(indexes)]


def build_temporal_windows(
    frame_paths: list[Path],
    window_seconds: float,
    stride_seconds: float,
    max_frames: int,
) -> list[TemporalWindow]:
    """Build overlapping windows while preserving the original timestamps."""
    if not frame_paths:
        return []
    if window_seconds <= 0:
        raise ValueError("window_seconds deve essere maggiore di zero.")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds deve essere maggiore di zero.")

    timestamped = [(get_timestamp(path), path) for path in frame_paths]
    first_timestamp = timestamped[0][0]
    last_timestamp = timestamped[-1][0]

    windows: list[TemporalWindow] = []
    previous_selection: tuple[Path, ...] | None = None
    nominal_start = first_timestamp
    index = 0

    while nominal_start <= last_timestamp + 1e-9:
        nominal_end = nominal_start + window_seconds
        candidates = [
            path
            for timestamp, path in timestamped
            if nominal_start <= timestamp < nominal_end
        ]

        if candidates:
            selected = tuple(_select_evenly(candidates, max_frames))
            if selected != previous_selection:
                actual_start = get_timestamp(selected[0])
                actual_end = get_timestamp(selected[-1])
                windows.append(
                    TemporalWindow(
                        segment_id=f"segment_{index:04d}",
                        start_time=round(actual_start, 3),
                        end_time=round(actual_end, 3),
                        frame_paths=selected,
                    )
                )
                previous_selection = selected
                index += 1

        nominal_start += stride_seconds

    return windows


def build_semantic_prompt(video_id: str, window: TemporalWindow) -> str:
    """Create the shared extraction prompt used by all three models."""
    frame_map = "\n".join(
        f'- frame {index}: "{path.name}", timestamp={get_timestamp(path):.3f}s'
        for index, path in enumerate(window.frame_paths, start=1)
    )

    return f"""
Analizza esclusivamente i frame forniti, nell'ordine temporale indicato.
Video: {video_id}
Intervallo osservato: {window.start_time:.3f}s - {window.end_time:.3f}s

Mappa dei frame:
{frame_map}

Obiettivo: estrarre una rappresentazione semantica verificabile del segmento.
Regole obbligatorie:
1. Non inventare dettagli non visibili.
2. Usa evidence_type="observed" per fatti direttamente visibili.
3. Usa evidence_type="inferred" soltanto per inferenze inevitabili o ipotesi causali.
4. Per ogni elemento indica confidence tra 0 e 1 e i nomi esatti dei frame che lo supportano.
5. Le etichette delle entità devono essere brevi, al singolare e in italiano.
6. Non confondere una successione temporale con una relazione causale.
7. Se una categoria non contiene elementi, restituisci una lista vuota.
8. Restituisci esclusivamente un oggetto JSON valido, senza Markdown e senza spiegazioni esterne.

Schema richiesto:
{{
  "entities": [
    {{
      "entity_id": "e1",
      "label": "persona",
      "role": "ruolo visibile o null",
      "attributes": ["attributo osservabile"],
      "start_time": {window.start_time:.3f},
      "end_time": {window.end_time:.3f},
      "evidence_frames": ["nome_frame.jpg"],
      "evidence_type": "observed",
      "confidence": 0.0
    }}
  ],
  "actions": [
    {{
      "description": "azione osservata",
      "actor": "e1 o descrizione",
      "object": "e2, descrizione o null",
      "start_time": 0.0,
      "end_time": 0.0,
      "evidence_frames": ["nome_frame.jpg"],
      "evidence_type": "observed",
      "confidence": 0.0
    }}
  ],
  "events": [
    {{
      "description": "evento temporalmente esteso",
      "participants": ["e1"],
      "start_time": 0.0,
      "end_time": 0.0,
      "evidence_frames": ["nome_frame.jpg"],
      "evidence_type": "observed",
      "confidence": 0.0
    }}
  ],
  "spatial_relations": [
    {{
      "subject": "e1 o descrizione",
      "relation": "davanti a|dietro a|sopra|sotto|dentro|fuori|a sinistra di|a destra di|vicino a|lontano da|su|tra",
      "object": "e2 o descrizione",
      "start_time": 0.0,
      "end_time": 0.0,
      "evidence_frames": ["nome_frame.jpg"],
      "evidence_type": "observed",
      "confidence": 0.0
    }}
  ],
  "state_changes": [
    {{
      "entity": "e1 o descrizione",
      "before": "stato precedente",
      "after": "stato successivo",
      "start_time": 0.0,
      "end_time": 0.0,
      "evidence_frames": ["nome_frame.jpg"],
      "evidence_type": "observed",
      "confidence": 0.0
    }}
  ],
  "temporal_relations": [
    {{
      "first_event": "evento A",
      "relation": "before|after|overlaps|during|starts|finishes|simultaneous",
      "second_event": "evento B",
      "evidence_frames": ["nome_frame.jpg"],
      "evidence_type": "observed",
      "confidence": 0.0
    }}
  ],
  "causal_hypotheses": [
    {{
      "cause": "causa candidata",
      "effect": "effetto osservato",
      "start_time": 0.0,
      "end_time": 0.0,
      "evidence_frames": ["nome_frame.jpg"],
      "evidence_type": "inferred",
      "confidence": 0.0
    }}
  ]
}}
""".strip()


def _extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise SemanticOutputError("La risposta non contiene un oggetto JSON.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise SemanticOutputError("Oggetto JSON incompleto nella risposta del modello.")


def parse_model_json(response: str) -> dict[str, Any]:
    """Parse a model response, optionally using json-repair when installed."""
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidate = _extract_balanced_json(cleaned)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        try:
            from json_repair import repair_json  # type: ignore

            parsed = json.loads(repair_json(candidate))
        except Exception as repair_error:
            raise SemanticOutputError(
                f"JSON non valido: {first_error}"
            ) from repair_error

    if not isinstance(parsed, dict):
        raise SemanticOutputError("La radice della risposta deve essere un oggetto JSON.")
    return parsed


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def normalize_semantic_output(
    payload: dict[str, Any],
    window: TemporalWindow,
) -> dict[str, Any]:
    """Enforce the common schema without inventing missing semantic content."""
    valid_evidence = set(window.evidence_frames)
    normalized: dict[str, Any] = {
        "segment_id": window.segment_id,
        "start_time": window.start_time,
        "end_time": window.end_time,
        "input_frames": window.evidence_frames,
    }

    for category in SEMANTIC_CATEGORIES:
        raw_items = payload.get(category, [])
        if not isinstance(raw_items, list):
            raw_items = []

        normalized_items: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue

            item = dict(raw_item)
            item["start_time"] = round(
                max(
                    window.start_time,
                    min(
                        window.end_time,
                        _safe_float(item.get("start_time"), window.start_time),
                    ),
                ),
                3,
            )
            item["end_time"] = round(
                max(
                    item["start_time"],
                    min(
                        window.end_time,
                        _safe_float(item.get("end_time"), window.end_time),
                    ),
                ),
                3,
            )

            confidence = _safe_float(item.get("confidence"), 0.0)
            item["confidence"] = round(max(0.0, min(1.0, confidence)), 4)

            evidence_type = str(item.get("evidence_type", "observed")).lower()
            if evidence_type not in {"observed", "inferred", "uncertain"}:
                evidence_type = "uncertain"
            if category == "causal_hypotheses" and evidence_type == "observed":
                evidence_type = "inferred"
            item["evidence_type"] = evidence_type

            evidence_frames = item.get("evidence_frames", [])
            if not isinstance(evidence_frames, list):
                evidence_frames = []
            item["evidence_frames"] = [
                str(name)
                for name in evidence_frames
                if str(name) in valid_evidence
            ]

            normalized_items.append(item)

        normalized[category] = normalized_items

    return normalized


def count_semantic_items(segment: dict[str, Any]) -> int:
    return sum(len(segment.get(category, [])) for category in SEMANTIC_CATEGORIES)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_video_with_inferencer(
    *,
    model_name: str,
    video_directory: Path,
    output_directory: Path,
    window_seconds: float,
    stride_seconds: float,
    max_frames: int,
    inferencer: Any,
    overwrite: bool,
    keep_raw: bool,
) -> dict[str, Any]:
    """Shared loop used by Qwen and Gemma extractors."""
    output_path = output_directory / f"{video_directory.name}_semantic.json"
    if output_path.exists() and not overwrite:
        return {
            "id_video": video_directory.name,
            "status": "skipped",
            "numero_segmenti": None,
            "numero_elementi": None,
            "numero_errori": None,
            "output": str(output_path),
        }

    frames = list_dense_frames(video_directory)
    if not frames:
        raise FileNotFoundError("Nessun dense frame trovato.")

    windows = build_temporal_windows(
        frames,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        max_frames=max_frames,
    )
    if not windows:
        raise RuntimeError("Non è stato possibile costruire finestre temporali.")

    segments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for window_index, window in enumerate(windows, start=1):
        print(
            f"  [{window_index}/{len(windows)}] {window.segment_id} "
            f"({window.start_time:.3f}-{window.end_time:.3f}s, "
            f"{len(window.frame_paths)} frame)"
        )
        prompt = build_semantic_prompt(video_directory.name, window)
        response = ""
        try:
            response = inferencer(window.frame_paths, prompt)
            parsed = parse_model_json(response)
            segment = normalize_semantic_output(parsed, window)
            if keep_raw:
                segment["raw_response"] = response
            segments.append(segment)
        except Exception as error:
            error_record = {
                "segment_id": window.segment_id,
                "start_time": window.start_time,
                "end_time": window.end_time,
                "input_frames": window.evidence_frames,
                "error": f"{type(error).__name__}: {error}",
            }
            if response:
                error_record["raw_response"] = response
            errors.append(error_record)
            print(f"    Errore: {error_record['error']}")

    result = {
        "id_video": video_directory.name,
        "model": model_name,
        "configuration": {
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "max_frames_per_window": max_frames,
            "input_type": "ordered_dense_frames",
        },
        "numero_segmenti": len(segments),
        "numero_errori": len(errors),
        "segments": segments,
        "errors": errors,
    }
    write_json(output_path, result)

    return {
        "id_video": video_directory.name,
        "status": "completed",
        "numero_segmenti": len(segments),
        "numero_elementi": sum(count_semantic_items(item) for item in segments),
        "numero_errori": len(errors),
        "output": str(output_path),
    }


def estimate_video_fps(frame_paths: Iterable[Path]) -> float:
    timestamps = [get_timestamp(path) for path in frame_paths]
    differences = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    if not differences:
        return 2.0
    median_delta = statistics.median(differences)
    return max(1.0, min(8.0, 1.0 / median_delta))
