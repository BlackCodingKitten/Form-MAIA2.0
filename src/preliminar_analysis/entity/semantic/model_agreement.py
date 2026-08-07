from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from semantic_common import SEMANTIC_CATEGORIES, write_csv, write_json


DEFAULT_MODEL_DIRECTORIES = {
    "qwen": "qwen",
    "onellm": "onellm",
    "gemma": "gemma",
}

STOPWORDS = {
    "a", "ad", "al", "alla", "alle", "allo", "ai", "agli",
    "da", "dal", "dalla", "dalle", "dallo", "dei", "del", "della",
    "delle", "dello", "di", "e", "ed", "gli", "i", "il", "in",
    "la", "le", "lo", "nel", "nella", "nelle", "nello", "o", "per",
    "su", "sul", "sulla", "sulle", "sullo", "tra", "un", "una", "uno",
}


def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(character for character in text if not unicodedata.combining(character))
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(token for token in tokens if token not in STOPWORDS)


def item_text(category: str, item: dict[str, Any]) -> str:
    if category == "entities":
        return " ".join(
            str(value)
            for value in (item.get("label"), item.get("role"))
            if value
        )
    if category in {"actions", "events"}:
        return str(item.get("description", ""))
    if category == "spatial_relations":
        return " ".join(
            str(item.get(key, ""))
            for key in ("subject", "relation", "object")
        )
    if category == "state_changes":
        return " ".join(
            str(item.get(key, ""))
            for key in ("entity", "before", "after")
        )
    if category == "temporal_relations":
        return " ".join(
            str(item.get(key, ""))
            for key in ("first_event", "relation", "second_event")
        )
    if category == "causal_hypotheses":
        return " ".join(
            str(item.get(key, ""))
            for key in ("cause", "effect")
        )
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def similarity(first: str, second: str) -> float:
    first_normalized = canonicalize(first)
    second_normalized = canonicalize(second)
    if not first_normalized or not second_normalized:
        return 0.0
    if first_normalized == second_normalized:
        return 1.0

    first_tokens = set(first_normalized.split())
    second_tokens = set(second_normalized.split())
    union = first_tokens | second_tokens
    jaccard = len(first_tokens & second_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, first_normalized, second_normalized).ratio()
    return round(0.55 * sequence + 0.45 * jaccard, 4)


def load_video_outputs(
    semantic_root: Path,
    model_directories: dict[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    outputs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model_name, directory_name in model_directories.items():
        directory = semantic_root / directory_name
        if not directory.exists():
            print(f"Directory modello assente, ignorata: {directory}")
            continue
        for path in sorted(directory.glob("*_semantic.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            video_id = str(payload.get("id_video") or path.stem.removesuffix("_semantic"))
            outputs[video_id][model_name] = payload
    return outputs


def cluster_items(
    category: str,
    model_items: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []

    ordered_models = sorted(model_items)
    for model_name in ordered_models:
        for item in model_items[model_name]:
            text = item_text(category, item)
            best_cluster = None
            best_score = 0.0

            for cluster in clusters:
                if model_name in cluster["items_by_model"]:
                    continue
                score = max(
                    similarity(text, existing_text)
                    for existing_text in cluster["texts_by_model"].values()
                )
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if best_cluster is not None and best_score >= threshold:
                best_cluster["items_by_model"][model_name] = item
                best_cluster["texts_by_model"][model_name] = text
                best_cluster["matching_scores"].append(best_score)
            else:
                clusters.append(
                    {
                        "items_by_model": {model_name: item},
                        "texts_by_model": {model_name: text},
                        "matching_scores": [],
                    }
                )

    available_models = len(model_items)
    results = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        support_models = sorted(cluster["items_by_model"])
        support_count = len(support_models)
        if support_count == available_models and available_models >= 2:
            status = "unanimous"
        elif support_count >= 2:
            status = "majority"
        else:
            status = "single_model"

        candidates = list(cluster["items_by_model"].values())
        representative = max(
            candidates,
            key=lambda item: float(item.get("confidence", 0.0) or 0.0),
        )
        mean_similarity = (
            sum(cluster["matching_scores"]) / len(cluster["matching_scores"])
            if cluster["matching_scores"]
            else None
        )

        results.append(
            {
                "cluster_id": f"{category}_{cluster_index:04d}",
                "status": status,
                "support_models": support_models,
                "support_count": support_count,
                "available_models": available_models,
                "agreement_ratio": round(support_count / available_models, 4),
                "mean_matching_similarity": (
                    round(mean_similarity, 4)
                    if mean_similarity is not None
                    else None
                ),
                "canonical_text": canonicalize(item_text(category, representative)),
                "representative": representative,
                "items_by_model": cluster["items_by_model"],
            }
        )

    results.sort(
        key=lambda item: (
            -item["support_count"],
            item["canonical_text"],
        )
    )
    return results


def analyze_video(
    video_id: str,
    outputs: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    segment_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model_name, payload in outputs.items():
        for segment in payload.get("segments", []):
            segment_id = str(segment.get("segment_id", ""))
            if segment_id:
                segment_map[segment_id][model_name] = segment

    analyzed_segments = []
    for segment_id in sorted(segment_map):
        model_segments = segment_map[segment_id]
        available_models = sorted(model_segments)
        first_segment = next(iter(model_segments.values()))
        category_results = {}

        for category in SEMANTIC_CATEGORIES:
            model_items = {
                model_name: [
                    item
                    for item in segment.get(category, [])
                    if isinstance(item, dict)
                ]
                for model_name, segment in model_segments.items()
            }
            category_results[category] = cluster_items(
                category,
                model_items,
                threshold,
            )

        analyzed_segments.append(
            {
                "segment_id": segment_id,
                "start_time": first_segment.get("start_time"),
                "end_time": first_segment.get("end_time"),
                "input_frames": first_segment.get("input_frames", []),
                "available_models": available_models,
                "categories": category_results,
            }
        )

    return {
        "id_video": video_id,
        "models": sorted(outputs),
        "similarity_threshold": threshold,
        "numero_segmenti": len(analyzed_segments),
        "segments": analyzed_segments,
    }


def summarize_video(result: dict[str, Any]) -> dict[str, Any]:
    counts = defaultdict(int)
    for segment in result["segments"]:
        for clusters in segment["categories"].values():
            for cluster in clusters:
                counts[cluster["status"]] += 1

    total = sum(counts.values())
    consensus = counts["unanimous"] + counts["majority"]
    return {
        "id_video": result["id_video"],
        "modelli": ";".join(result["models"]),
        "numero_segmenti": result["numero_segmenti"],
        "cluster_totali": total,
        "cluster_unanimi": counts["unanimous"],
        "cluster_maggioranza": counts["majority"],
        "cluster_singolo_modello": counts["single_model"],
        "quota_consenso": round(consensus / total, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Confronta gli output semantici di Qwen, OneLLM e Gemma "
            "senza usare la maggioranza come ground truth."
        )
    )
    parser.add_argument(
        "semantic_root",
        nargs="?",
        default="data/preliminar_analysis/semantic_analysis",
        type=Path,
    )
    parser.add_argument(
        "output_directory",
        nargs="?",
        default="data/preliminar_analysis/semantic_analysis/agreement",
        type=Path,
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.62,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = load_video_outputs(
        args.semantic_root,
        DEFAULT_MODEL_DIRECTORIES,
    )
    if not outputs:
        parser.error("Nessun output semantico trovato.")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_id, model_outputs in sorted(outputs.items()):
        if len(model_outputs) < 2:
            print(
                f"{video_id}: presente un solo modello, impossibile "
                "calcolare accordo inter-modello."
            )
            continue

        output_path = args.output_directory / f"{video_id}_agreement.json"
        if output_path.exists() and not args.overwrite:
            print(f"{video_id}: output già presente, ignorato.")
            continue

        print(
            f"Analisi accordo di {video_id}: "
            f"{', '.join(sorted(model_outputs))}"
        )
        result = analyze_video(
            video_id,
            model_outputs,
            args.similarity_threshold,
        )
        write_json(output_path, result)
        rows.append(summarize_video(result))

    write_csv(args.output_directory / "riepilogo_accordo.csv", rows)
    print(f"Risultati salvati in: {args.output_directory}")


if __name__ == "__main__":
    main()
