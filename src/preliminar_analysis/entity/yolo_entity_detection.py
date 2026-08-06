import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_timestamp(image_path: Path) -> float:
    """Extract the timestamp from names like dense_0001_t0000.250.jpg."""
    return float(image_path.stem.rsplit("_t", 1)[1])


def write_csv(output_path: Path, rows: list[dict]) -> None:
    """Write a CSV file using the first row keys as column names."""
    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def process_video(
    model_path: str,
    video_directory: Path,
    output_directory: Path,
    confidence: float,
    device: str,
) -> tuple[dict, list[dict]]:
    """Run YOLO and ByteTrack on the dense frames of one video."""
    frame_paths = sorted(
        path
        for path in (video_directory / "dense_frames").iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not frame_paths:
        raise FileNotFoundError("Nessun dense frame trovato.")

    # A new model instance starts a new tracker for each video.
    model = YOLO(model_path)
    frames = []
    tracks = {}

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Frame non leggibile, ignorato: {frame_path.name}")
            continue

        # ByteTrack performs detection and tracking in a single step.
        result = model.track(
            source=image,
            tracker="bytetrack.yaml",
            persist=True,
            conf=confidence,
            device=device,
            verbose=False,
        )[0]

        boxes = result.boxes.xyxy.cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        track_ids = (
            result.boxes.id.int().cpu().tolist()
            if result.boxes.id is not None
            else [None] * len(boxes)
        )

        timestamp = get_timestamp(frame_path)
        detections = []

        for box, class_id, score, track_id in zip(
            boxes,
            class_ids,
            scores,
            track_ids,
        ):
            class_name = result.names[class_id]

            detection = {
                "id_traccia": track_id,
                "id_classe": class_id,
                "nome_classe": class_name,
                "confidenza": round(score, 4),
                "riquadro_xyxy": [round(value, 2) for value in box],
            }
            detections.append(detection)

            if track_id is None:
                continue

            if track_id not in tracks:
                tracks[track_id] = {
                    "id_traccia": track_id,
                    "id_classe": class_id,
                    "nome_classe": class_name,
                    "prima_comparsa": timestamp,
                    "ultima_comparsa": timestamp,
                    "numero_osservazioni": 0,
                    "somma_confidenze": 0.0,
                    "osservazioni": [],
                }

            tracks[track_id]["ultima_comparsa"] = timestamp
            tracks[track_id]["numero_osservazioni"] += 1
            tracks[track_id]["somma_confidenze"] += score
            tracks[track_id]["osservazioni"].append({
                "id_frame": frame_path.stem.split("_t", 1)[0],
                "timestamp": timestamp,
                "confidenza": round(score, 4),
                "riquadro_xyxy": [round(value, 2) for value in box],
            })

        frames.append({
            "id_frame": frame_path.stem.split("_t", 1)[0],
            "nome_file": frame_path.name,
            "timestamp": timestamp,
            "numero_rilevamenti": len(detections),
            "rilevamenti": detections,
        })

    track_list = []

    for track in tracks.values():
        track["durata_traccia"] = round(
            track["ultima_comparsa"] - track["prima_comparsa"],
            3,
        )
        track["confidenza_media"] = round(
            track["somma_confidenze"] / track["numero_osservazioni"],
            4,
        )
        del track["somma_confidenze"]
        track_list.append(track)

    track_list.sort(key=lambda track: track["id_traccia"])

    result = {
        "id_video": video_directory.name,
        "modello": model_path,
        "tracker": "ByteTrack",
        "dispositivo": f"CUDA:{device}",
        "soglia_confidenza": confidence,
        "numero_frame": len(frames),
        "numero_tracce": len(track_list),
        "frame": frames,
        "tracce": track_list,
    }

    output_path = output_directory / f"{video_directory.name}_tracciamento.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    video_summary = {
        "id_video": video_directory.name,
        "numero_frame": len(frames),
        "numero_tracce": len(track_list),
        "numero_rilevamenti": sum(
            frame["numero_rilevamenti"] for frame in frames
        ),
    }

    track_summary = [
        {
            "id_video": video_directory.name,
            "id_traccia": track["id_traccia"],
            "nome_classe": track["nome_classe"],
            "prima_comparsa": track["prima_comparsa"],
            "ultima_comparsa": track["ultima_comparsa"],
            "durata_traccia": track["durata_traccia"],
            "numero_osservazioni": track["numero_osservazioni"],
            "confidenza_media": track["confidenza_media"],
        }
        for track in track_list
    ]

    return video_summary, track_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Esegue YOLO e ByteTrack sui dense frame."
    )
    parser.add_argument(
        "preprocessing_directory",
        default="data/preliminar_analysis/preprocessing",
        type=Path,
        help="Directory che contiene una cartella per ogni video.",
    )
    parser.add_argument(
        "output_directory",
        default="data/preliminar_analysis/entity_detection",
        type=Path,
        help="Directory in cui salvare i risultati.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Modello YOLO. Predefinito: yolo11n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidenza minima. Predefinita: 0.25",
    )
    parser.add_argument(
        "--device",
        default="3",
        help="Dispositivo CUDA. Predefinito: 3",
    )
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)

    video_directories = sorted(
        path
        for path in args.preprocessing_directory.iterdir()
        if path.is_dir() and (path / "dense_frames").is_dir()
    )

    if not video_directories:
        parser.error("Non sono state trovate cartelle dense_frames.")

    video_rows = []
    track_rows = []

    for index, video_directory in enumerate(video_directories, 1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Analisi di {video_directory.name} su CUDA:{args.device}"
        )

        try:
            video_summary, track_summary = process_video(
                args.model,
                video_directory,
                args.output_directory,
                args.confidence,
                args.device,
            )
            video_rows.append(video_summary)
            track_rows.extend(track_summary)
        except Exception as error:
            print(
                f"Errore durante l'analisi di "
                f"{video_directory.name}: {error}"
            )

    write_csv(args.output_directory / "riepilogo_video.csv", video_rows)
    write_csv(args.output_directory / "riepilogo_tracce.csv", track_rows)

    print(
        f"Analisi completata. Risultati salvati in: "
        f"{args.output_directory}"
    )


if __name__ == "__main__":
    main()